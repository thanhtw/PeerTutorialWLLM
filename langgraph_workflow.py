"""
LangGraph Workflow for Java Peer Review Training System.

This module implements the code review workflow as a LangGraph graph.
"""

__all__ = ['JavaCodeReviewGraph']

import logging
import os
import random
import re
from typing import Dict, List, Any, Tuple
from langgraph.graph import StateGraph, END
from state_schema import WorkflowState, CodeSnippet, ReviewAttempt

# Import domain-specific components
from core.code_generator import CodeGenerator
from core.student_response_evaluator import StudentResponseEvaluator
from core.feedback_manager import FeedbackManager
from data.json_error_repository import JsonErrorRepository
from llm_manager import LLMManager

from utils.code_utils import (
    create_code_generation_prompt, 
    extract_code_from_response,   
    get_error_count_for_difficulty,
    generate_comparison_report,
    strip_error_annotations
)

from utils.enhanced_error_tracking import enrich_error_information

from utils.code_evaluation_agent import CodeEvaluationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JavaCodeReviewGraph:
    """
    LangGraph implementation of the Java Code Review workflow.
    """
    
    def __init__(self, llm_manager: LLMManager = None):
        """Initialize the graph with domain components."""
        # Initialize LLM Manager if not provided
        self.llm_manager = llm_manager or LLMManager()
        
        # Initialize repositories
        self.error_repository = JsonErrorRepository()
        
        # Initialize domain objects
        self._initialize_domain_objects()
        
        # Create the graph
        self.workflow = self._build_graph()
    
    def _initialize_domain_objects(self):
        """Initialize domain objects with LLMs if available."""
        # Check Ollama connection
        connection_status, _ = self.llm_manager.check_ollama_connection()
        
        if connection_status:
            # Initialize models for each component
            generative_model = self.llm_manager.initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
            review_model = self.llm_manager.initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
            summary_model = self.llm_manager.initialize_model_from_env("SUMMARY_MODEL", "SUMMARY_TEMPERATURE")
            compare_model = self.llm_manager.initialize_model_from_env("COMPARE_MODEL", "COMPARE_TEMPERATURE")
            
            # Initialize domain objects with models
            self.code_generator = CodeGenerator(generative_model) if generative_model else CodeGenerator()
            self.evaluator = StudentResponseEvaluator(review_model) if review_model else StudentResponseEvaluator()
            self.feedback_manager = FeedbackManager(self.evaluator)
            
            # Initialize the new code evaluation agent
            try:                
                self.code_evaluation_agent = CodeEvaluationAgent()
                logger.info("Successfully initialized Code Evaluation Agent")
            except Exception as e:
                logger.error(f"Error initializing Code Evaluation Agent: {str(e)}")
                self.code_evaluation_agent = None
        else:
            # Initialize domain objects without LLMs
            self.code_generator = CodeGenerator()
            self.evaluator = StudentResponseEvaluator()
            self.feedback_manager = FeedbackManager(self.evaluator)
            
            # Initialize the new code evaluation agent without LLM
            try:
                from utils.code_evaluation_agent import CodeEvaluationAgent
                self.code_evaluation_agent = CodeEvaluationAgent()
                logger.info("Successfully initialized Code Evaluation Agent (without LLM)")
            except Exception as e:
                logger.error(f"Error initializing Code Evaluation Agent: {str(e)}")
                self.code_evaluation_agent = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with code evaluation and regeneration."""
        # Create a new graph with our state schema
        workflow = StateGraph(WorkflowState)
        
        # Define nodes
        workflow.add_node("generate_code", self.generate_code_node)
        workflow.add_node("evaluate_code", self.evaluate_code_node) 
        workflow.add_node("regenerate_code", self.regenerate_code_node)
        workflow.add_node("review_code", self.review_code_node)
        workflow.add_node("analyze_review", self.analyze_review_node)
        workflow.add_node("generate_summary", self.generate_summary_node)
        
        # Define edges
        workflow.add_edge("generate_code", "evaluate_code")  # First go to evaluation
        workflow.add_edge("regenerate_code", "evaluate_code")  # Regeneration also goes to evaluation
        
        # Conditional edge based on evaluation result
        workflow.add_conditional_edges(
            "evaluate_code",
            self.should_regenerate_or_review,
            {
                "regenerate_code": "regenerate_code",
                "review_code": "review_code"
            }
        )
        
        workflow.add_edge("review_code", "analyze_review")
        
        # Conditional edges from analyze_review
        workflow.add_conditional_edges(
            "analyze_review",
            self.should_continue_review,
            {
                "continue_review": "review_code",
                "generate_summary": "generate_summary"
            }
        )
        
        workflow.add_edge("generate_summary", END)
        
        # Set the entry point
        workflow.set_entry_point("generate_code")
        
        return workflow

    def generate_code_node(self, state: WorkflowState) -> WorkflowState:
        """Generate Java code with errors node with enhanced debugging for all modes."""
        try:
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            selected_error_categories = state.selected_error_categories
            state.evaluation_attempts = 0
            state.evaluation_result = None
            state.code_generation_feedback = None
            
            # Print the selected categories for debugging
            print("\n========== GENERATE_CODE_NODE ==========")
            print(f"Code Length: {code_length}")
            print(f"Difficulty Level: {difficulty_level}")
            print(f"Selected Error Categories: {selected_error_categories}")
            
            # Check if we have valid selected categories
            if not selected_error_categories or (
                not selected_error_categories.get("build", []) and 
                not selected_error_categories.get("checkstyle", [])
            ):
                # Instead of using defaults, require explicit selection
                state.error = "No error categories selected. Please select at least one problem area or error category before generating code."
                return state
                    
            # Generate code with errors
            # Get errors from selected categories
            selected_errors, basic_problem_descriptions = self.error_repository.get_errors_for_llm(
                selected_categories=selected_error_categories,
                count=get_error_count_for_difficulty(difficulty_level),
                difficulty=difficulty_level
            )
            
            # Enhanced debugging: Print detailed information about selected errors
            print(f"\n========== SELECTED ERRORS FOR CODE GENERATION ==========")
            print(f"Total Selected Errors Count: {len(selected_errors)}")
            
            if selected_errors:
                print("\n--- DETAILED ERROR LISTING ---")
                for i, error in enumerate(selected_errors, 1):
                    print(f"  {i}. Type: {error.get('type', 'Unknown')}")
                    print(f"     Name: {error.get('name', 'Unknown')}")
                    print(f"     Category: {error.get('category', 'Unknown')}")
                    print(f"     Description: {error.get('description', 'Unknown')}")
                    if 'implementation_guide' in error:
                        print(f"     Implementation Guide: {error.get('implementation_guide', '')[:100]}..." 
                            if len(error.get('implementation_guide', '')) > 100 
                            else error.get('implementation_guide', ''))
                    print()
            
            # Generate code with selected errors and get enhanced error information
            # Now returns both annotated and clean versions
            annotated_code, clean_code, enhanced_errors, detailed_problems = self._generate_code_with_errors(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=selected_errors
            )
            
            # Create code snippet object with enhanced information
            code_snippet = CodeSnippet(
                code=annotated_code,  # Store annotated version with error comments
                clean_code=clean_code,  # Store clean version without error comments
                known_problems=detailed_problems,  # Use the detailed problems instead
                raw_errors={
                    "build": [e for e in enhanced_errors if e["type"] == "build"],
                    "checkstyle": [e for e in enhanced_errors if e["type"] == "checkstyle"]
                },
                # Add the enhanced error information
                enhanced_errors=enhanced_errors
            )
            
            # Print debug information after generation
            print("\n========== GENERATED CODE SNIPPET ==========")
            print(f"Annotated Code Length: {len(annotated_code)} characters, {len(annotated_code.splitlines())} lines")
            print(f"Clean Code Length: {len(clean_code)} characters, {len(clean_code.splitlines())} lines")
            print("\n========== KNOWN PROBLEMS ==========")
            for i, problem in enumerate(detailed_problems, 1):
                print(f"{i}. {problem}")
            print("\n========== ENHANCED ERRORS ==========")
            for i, error in enumerate(enhanced_errors, 1):
                print(f"{i}. Type: {error.get('type', 'Unknown')}")
                print(f"   Name: {error.get('name', 'Unknown')}")
                print(f"   Description: {error.get('description', 'Unknown')}")
                print(f"   Line: {error.get('line_number', 'Unknown')}")
                print(f"   Content: {error.get('line_content', 'Unknown')}")
                print()
            
            # Update state
            state.code_snippet = code_snippet
            state.current_step = "review"
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            import traceback
            traceback.print_exc()
            state.error = f"Error generating code: {str(e)}"
            return state
                
    # Update the generate_code_with_specific_errors method in langgraph_workflow.py

    def generate_code_with_specific_errors(self, state: WorkflowState, specific_errors: List[Dict[str, Any]]) -> WorkflowState:
        """Generate Java code with specific errors selected by the user."""
        try:
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            
            # Print the specific errors for debugging
            print("\n========== GENERATE_CODE_WITH_SPECIFIC_ERRORS ==========")
            print(f"Code Length: {code_length}")
            print(f"Difficulty Level: {difficulty_level}")
            print(f"Number of Specific Errors: {len(specific_errors)}")
            
            for i, error in enumerate(specific_errors):
                print(f"  {i+1}. {error.get('type', 'Unknown')} - {error.get('name', 'Unknown')} ({error.get('category', 'Unknown')})")
            
            # Initialize the evaluation agent if not already done
            if not hasattr(self, 'code_evaluation_agent') or self.code_evaluation_agent is None:
                try:
                    from utils.code_evaluation_agent import CodeEvaluationAgent
                    self.code_evaluation_agent = CodeEvaluationAgent()
                    logger.info("Created Code Evaluation Agent for specific errors mode")
                except Exception as e:
                    logger.error(f"Error initializing Code Evaluation Agent: {str(e)}")
            
            # Format problem descriptions
            problem_descriptions = []
            for error in specific_errors:
                error_type = error.get("type", "Unknown")
                name = error.get("name", "Unknown")
                description = error.get("description", "")
                category = error.get("category", "")
                
                if error_type == "build":
                    problem_descriptions.append(f"Build Error - {name}: {description} (Category: {category})")
                else:  # checkstyle
                    problem_descriptions.append(f"Checkstyle Error - {name}: {description} (Category: {category})")
            
            # Generate code with selected errors - now uses the enhanced method with evaluation agent
            annotated_code, clean_code, enhanced_errors, detailed_problems = self._generate_code_with_errors(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=specific_errors
            )
            
            # Use detailed_problems if available, otherwise fall back to basic problem_descriptions
            final_problems = detailed_problems if detailed_problems else problem_descriptions
            
            # Create code snippet object with both code versions
            code_snippet = CodeSnippet(
                code=annotated_code,  # Annotated version with error comments
                clean_code=clean_code,  # Clean version without error comments
                known_problems=final_problems,
                raw_errors={
                    "build": [e for e in specific_errors if e.get("type") == "build"],
                    "checkstyle": [e for e in specific_errors if e.get("type") == "checkstyle"]
                },
                enhanced_errors=enhanced_errors if enhanced_errors else []
            )

            # Print debug information after generation
            print("\n========== GENERATED CODE SNIPPET (SPECIFIC ERRORS) ==========")
            print(f"Annotated Code Length: {len(annotated_code)} characters, {len(annotated_code.splitlines())} lines")
            print(f"Clean Code Length: {len(clean_code)} characters, {len(clean_code.splitlines())} lines")
            print("\n========== KNOWN PROBLEMS ==========")
            for i, problem in enumerate(code_snippet.known_problems, 1):
                print(f"{i}. {problem}")
            print("\n========== RAW ERRORS ==========")
            print(f"Build Errors: {len([e for e in specific_errors if e.get('type') == 'build'])}")
            print(f"Checkstyle Errors: {len([e for e in specific_errors if e.get('type') == 'checkstyle'])}")
            
            # Print enhanced errors if available
            if enhanced_errors:
                print("\n========== ENHANCED ERRORS ==========")
                for i, error in enumerate(enhanced_errors, 1):
                    print(f"{i}. Type: {error.get('type', 'Unknown')}")
                    print(f"   Name: {error.get('name', 'Unknown')}")
                    print(f"   Description: {error.get('description', 'Unknown')}")
                    print(f"   Line: {error.get('line_number', 'Unknown')}")
                    print(f"   Content: {error.get('line_content', 'Unknown')}")
                    print()
            
            # Update state
            state.code_snippet = code_snippet
            state.current_step = "review"
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating code with specific errors: {str(e)}")
            import traceback
            traceback.print_exc()
            state.error = f"Error generating code with specific errors: {str(e)}"
            return state
    
    def review_code_node(self, state: WorkflowState) -> WorkflowState:
        """Review code node - this is a placeholder since user input happens in the UI."""
        # This node is primarily a placeholder since the actual review is submitted via the UI
        state.current_step = "review"
        return state
    
    def analyze_review_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze student review node."""
        try:
            # Get the latest review
            if not state.review_history:
                state.error = "No review submitted to analyze"
                return state
                    
            latest_review = state.review_history[-1]
            student_review = latest_review.student_review
            
            # Get code snippet
            if not state.code_snippet:
                state.error = "No code snippet available"
                return state
                    
            code_snippet = state.code_snippet.code
            known_problems = state.code_snippet.known_problems
            enhanced_errors = getattr(state.code_snippet, "enhanced_errors", None)
            
            # Analyze the student review with enhanced error information if available
            if enhanced_errors:
                analysis = self.evaluator._evaluate_with_llm(
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review,
                    enhanced_errors=enhanced_errors
                )
            else:
                analysis = self.evaluator.evaluate_review(
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review
                )
            
            # Update the review with analysis
            latest_review.analysis = analysis
            
            # Check if the review is sufficient
            review_sufficient = analysis.get("review_sufficient", False)
            state.review_sufficient = review_sufficient
            
            # Generate targeted guidance if needed
            if not review_sufficient and state.current_iteration < state.max_iterations:
                targeted_guidance = self.evaluator.generate_targeted_guidance(
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review,
                    review_analysis=analysis,
                    iteration_count=state.current_iteration,
                    max_iterations=state.max_iterations
                )
                latest_review.targeted_guidance = targeted_guidance
            
            # Increment iteration count
            state.current_iteration += 1
            
            # Update state
            state.current_step = "analyze"
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing review: {str(e)}")
            state.error = f"Error analyzing review: {str(e)}"
            return state
    
    def generate_summary_node(self, state: WorkflowState) -> WorkflowState:
        """Generate summary and comparison report node."""
        try:
            # Generate final feedback
            if not state.review_history:
                state.error = "No reviews available for summary"
                return state
                    
            # Generate feedback using the feedback manager
            all_reviews = [review.student_review for review in state.review_history]
            all_analyses = [review.analysis for review in state.review_history]
            
            # Initialize the feedback manager with the code snippet and known problems
            self.feedback_manager.start_new_review_session(
                code_snippet=state.code_snippet.code,
                known_problems=state.code_snippet.known_problems
            )
            
            # Load all reviews into the feedback manager
            for i, review_text in enumerate(all_reviews):
                if i < len(all_reviews) - 1:  # Skip the last one as we'll process it differently
                    self.feedback_manager.submit_review(review_text)
            
            # Submit the last review to get final analysis
            if all_reviews:
                self.feedback_manager.submit_review(all_reviews[-1])
            
            # Generate final feedback and comparison report
            final_feedback = self.feedback_manager.generate_final_feedback()
            
            # Safe version of comparison report generation
            comparison_report = self._generate_comparison_report(
                state.code_snippet.known_problems,
                state.review_history[-1].analysis if state.review_history else {}
            )
            
            # Update state
            state.review_summary = final_feedback
            state.comparison_report = comparison_report
            state.current_step = "complete"
            
            return state
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            state.error = f"Error generating summary: {str(e)}"
            return state
    
    # Conditional edge implementations
    def should_regenerate_or_review(self, state: WorkflowState) -> str:
        """
        Determine if we should regenerate code or move to review.
        This is used for the conditional edge from the evaluate_code node.
        
        Returns:
            "regenerate_code" if we need to regenerate code based on evaluation feedback
            "review_code" if the code is valid or we've reached max attempts
        """
        # Check if current step is explicitly set to regenerate
        if state.current_step == "regenerate":
            return "regenerate_code"
        
        # Check if evaluation passed
        if state.evaluation_result and state.evaluation_result.get("valid", False):
            return "review_code"
        
        # Check if we've reached max attempts
        if hasattr(state, 'evaluation_attempts') and state.evaluation_attempts >= state.max_evaluation_attempts:
            return "review_code"
        
        # Default to regenerate if we have an evaluation result but it's not valid
        if state.evaluation_result:
            return "regenerate_code"
        
        # If no evaluation result yet, move to review
        return "review_code"
    
    def should_continue_review(self, state: WorkflowState) -> str:
        """
        Determine if we should continue with another review iteration or generate summary.
        This is used for the conditional edge from the analyze_review node.
        
        Returns:
            "continue_review" if more review iterations are needed
            "generate_summary" if the review is sufficient or max iterations reached
        """
        # Check if we've reached max iterations
        if state.current_iteration > state.max_iterations:
            return "generate_summary"
        
        # Check if the review is sufficient
        if state.review_sufficient:
            return "generate_summary"
        
        # Otherwise, continue reviewing
        return "continue_review"

    def _generate_code_with_errors(self, code_length: str, difficulty_level: str, selected_errors: List[Dict[str, Any]]) -> Tuple[str, str, List[Dict[str, Any]], List[str]]:
        """
        Generate Java code with the selected errors, using the evaluation agent to ensure quality.
        
        Args:
            code_length: Length of code (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            selected_errors: List of selected errors to include in the code
        
        Returns:
            Tuple of (annotated_code, clean_code, enhanced_errors, detailed_problems)
        """
        
        # Import the validation function
        from utils.error_validation import validate_code_errors
        from utils.export_utils import export_prompt_response
        
        # Initialize the evaluation agent if not already done
        if not hasattr(self, 'code_evaluation_agent') or self.code_evaluation_agent is None:
            try:
                # Get an appropriate LLM for evaluation
                evaluation_llm = self.llm_manager.initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
                self.code_evaluation_agent = CodeEvaluationAgent(llm=evaluation_llm)
                logger.info("Created Code Evaluation Agent with LLM for error detection")
            except Exception as e:
                logger.error(f"Error initializing Code Evaluation Agent: {str(e)}")
                self.code_evaluation_agent = None
        
        # Use the local variable for clarity
        evaluation_agent = self.code_evaluation_agent
        
        # Validate we have proper error information
        if not selected_errors:
            logger.warning("No errors specified for code generation")
            selected_errors = []
        
        # Ensure all errors have the minimal required fields
        normalized_errors = []
        for error in selected_errors:
            normalized = error.copy()
            if "type" not in normalized and "category" in normalized:
                # Try to infer type from category
                category = normalized["category"].lower()
                if any(cat in category for cat in ["compile", "runtime", "logical", "warning"]):
                    normalized["type"] = "build"
                else:
                    normalized["type"] = "checkstyle"
            
            if "name" not in normalized:
                if "check_name" in normalized:
                    normalized["name"] = normalized["check_name"]
                elif "error_name" in normalized:
                    normalized["name"] = normalized["error_name"]
            
            # Ensure we have a description
            if "description" not in normalized:
                normalized["description"] = f"{normalized.get('type', 'Error')} of type {normalized.get('name', 'unknown')}"
            
            normalized_errors.append(normalized)
        
        # Use enhanced errors for generation
        selected_errors = normalized_errors
        
        # Use the code generator to create code with errors
        if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
            # Track attempts to generate proper code with errors
            max_attempts = 3  # Reduced to 3 attempts to ensure faster completion
            current_attempt = 0
            best_code = None
            best_validation = None
            
            while current_attempt < max_attempts:
                current_attempt += 1
                
                # Create a detailed prompt for the LLM - use improved prompt after first attempt
                if current_attempt == 1 or best_code is None:
                    prompt = create_code_generation_prompt(
                        code_length=code_length,
                        difficulty_level=difficulty_level,
                        selected_errors=selected_errors,
                        include_error_annotations=True  # Generate code with annotations
                    )
                else:
                    # Use evaluation agent to create an improved prompt based on previous attempt
                    try:
                        if evaluation_agent:
                            evaluation = evaluation_agent.evaluate_code(
                                best_code[1],  # Use the clean code from best attempt
                                selected_errors
                            )
                            
                            prompt = evaluation_agent.generate_improved_prompt(
                                best_code[1],  # Use the clean code from best attempt
                                selected_errors,
                                evaluation
                            )
                        else:
                            # Fallback if no evaluation agent
                            prompt = create_code_generation_prompt(
                                code_length=code_length,
                                difficulty_level=difficulty_level,
                                selected_errors=selected_errors,
                                include_error_annotations=True
                            )
                            prompt += f"\n\nPrevious attempt failed to implement all errors. Please try again with attempt {current_attempt}."
                    except Exception as e:
                        logger.error(f"Error generating improved prompt: {str(e)}")
                        # Fall back to standard prompt
                        prompt = create_code_generation_prompt(
                            code_length=code_length,
                            difficulty_level=difficulty_level,
                            selected_errors=selected_errors,
                            include_error_annotations=True
                        )
                        prompt += f"\n\nPrevious attempt failed to implement all errors. Please try again and ensure ALL requested errors are implemented."
                        
                        # Add emphasis on adding ALL requested errors
                        prompt += "\n\nMAKE SURE to implement ALL of the following errors:"
                        for error in selected_errors:
                            error_type = error.get("type", "").upper()
                            name = error.get("name", "")
                            prompt += f"\n- {error_type} - {name}"
                
                # Print the prompt for debugging
                print(f"\n========== CODE GENERATION PROMPT (Attempt {current_attempt}/{max_attempts}) ==========")
                print(prompt)
                
                # Generate code with errors
                response = self.code_generator.llm.invoke(prompt)
                
                # Export the prompt and response for debugging
                try:
                    from utils.export_utils import export_prompt_response
                    export_prompt_response(
                        prompt=prompt,
                        response=str(response),
                        operation_type=f"code_generation_attempt_{current_attempt}",
                        error_list=selected_errors
                    )
                except Exception as export_err:
                    logger.error(f"Error exporting prompt/response: {str(export_err)}")
                
                # Print the LLM response for debugging
                print(f"\n========== LLM RESPONSE (Attempt {current_attempt}/{max_attempts}) ==========")
                print(response)
                
                # Extract the code with annotations
                annotated_code = extract_code_from_response(response)
                
                if annotated_code and len(annotated_code.strip()) > 50:
                    # Create clean version by stripping ALL comments
                    clean_code = strip_error_annotations(annotated_code)
                    
                    # Debug output to check the cleaning process
                    print("\n========== CLEAN CODE GENERATION ==========")
                    print(f"Annotated Code Length: {len(annotated_code.splitlines())} lines")
                    print(f"Clean Code Length: {len(clean_code.splitlines())} lines")
                    print(f"Removed {len(annotated_code.splitlines()) - len(clean_code.splitlines())} comment lines")
                    
                    # Validate using LLM evaluation if available
                    try:
                        if evaluation_agent and evaluation_agent.llm:
                            # Use LLM-based validation
                            validation_results = evaluation_agent._evaluate_with_llm(clean_code, selected_errors)
                            
                            print("\n========== LLM CODE EVALUATION RESULTS ==========")
                            print(f"Valid: {validation_results['valid']}")
                            print(f"Found Errors: {validation_results['found_errors']}")
                            print(f"Missing Errors: {validation_results['missing_errors']}")
                            if "llm_feedback" in validation_results:
                                print("\nLLM Feedback:")
                                print(validation_results["llm_feedback"])
                            
                            # Check error locations if available
                            if "detailed_analysis" in validation_results:
                                found_errors = validation_results["detailed_analysis"].get("found_errors", [])
                                for error in found_errors:
                                    print(f"Found {error.get('error_type')} - {error.get('error_name')} at line {error.get('line_number')}")
                                    print(f"Code segment: {error.get('code_segment')}")
                                    print(f"Explanation: {error.get('explanation')}")
                        else:
                            # Fall back to regex-based validation
                            validation_results = validate_code_errors(clean_code, selected_errors)
                            
                            print("\n========== REGEX VALIDATION RESULTS ==========")
                            print(f"Valid: {validation_results['valid']}")
                            print(f"Found Errors: {validation_results['found_errors']}")
                            print(f"Missing Errors: {validation_results['missing_errors']}")
                        
                        # If all errors are implemented, we're done
                        if validation_results['valid']:
                            print(f"Valid code with all errors implemented (Attempt {current_attempt})")
                            # Enrich the error information using the clean code
                            enhanced_errors, detailed_problems = enrich_error_information(clean_code, selected_errors)
                            return annotated_code, clean_code, enhanced_errors, detailed_problems
                        
                        # Otherwise, store the best result so far (most errors implemented)
                        if best_validation is None or len(validation_results['found_errors']) > len(best_validation.get('found_errors', [])):
                            best_code = (annotated_code, clean_code)
                            best_validation = validation_results
                            
                        # If we've found at least 80% of the errors, that's good enough
                        # Only accept if we have at least one error
                        if (len(selected_errors) > 0 and 
                            len(validation_results['found_errors']) >= int(0.8 * len(selected_errors)) and
                            len(validation_results['found_errors']) > 0):
                            print(f"Found {len(validation_results['found_errors'])}/{len(selected_errors)} errors (>= 80%), accepting result")
                            # Enrich the error information using the clean code
                            enhanced_errors, detailed_problems = enrich_error_information(clean_code, selected_errors)
                            return annotated_code, clean_code, enhanced_errors, detailed_problems
                        
                        # Debug output for missing errors
                        print("\n========== MISSING ERRORS ==========")
                        for missing in validation_results.get('missing_errors', []):
                            print(f"- {missing}")
                            
                            # Find the corresponding error details
                            for error in selected_errors:
                                error_key = f"{error.get('type', '').upper()} - {error.get('name', '')}"
                                if error_key == missing:
                                    print(f"  Description: {error.get('description', 'No description')}")
                                    print(f"  Implementation guide: {error.get('implementation_guide', 'No guide')}")
                    
                    except Exception as e:
                        logger.error(f"Error in code validation: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Continue with next attempt
                        continue
            
            # If we've tried our maximum attempts but still haven't found all errors,
            # use the best result we found
            if best_code:
                print(f"Using best result after {max_attempts} attempts - found {len(best_validation['found_errors'])}/{len(selected_errors)} errors")
                annotated_code, clean_code = best_code
                
                # Enrich the error information using the clean code
                enhanced_errors, detailed_problems = enrich_error_information(clean_code, selected_errors)
                return annotated_code, clean_code, enhanced_errors, detailed_problems
        
        # Fallback: generate clean code and manually add errors
        try:
            base_code = self.code_generator.generate_java_code(
                code_length=code_length,
                difficulty_level=difficulty_level
            )
            
            # Print fallback generation for debugging
            print("\n========== FALLBACK CODE GENERATION ==========")
            print(base_code)
            
            # Create clean code (start with base code)
            clean_code = strip_error_annotations(base_code)
            
            # For fallback, manually add obvious errors and annotations
            modified_code = clean_code
            
            # Only add errors if we have any requested
            if selected_errors:
                print("\n========== MANUALLY ADDING ERRORS IN FALLBACK MODE ==========")
                for error in selected_errors:
                    error_type = error.get("type", "").lower() 
                    error_name = error.get("name", "")
                    
                    print(f"Adding error: {error_type.upper()} - {error_name}")
                    
                    # Add appropriate error code based on type
                    if error_type == "build":
                        # Handle common build error types
                        if "cannot find symbol" in error_name.lower():
                            modified_code = self._insert_in_main_method(modified_code, 
                                "// ERROR: BUILD - Cannot find symbol - Using undeclared variable\n        int result = undeclaredVar + 5;")
                        
                        elif "incompatible types" in error_name.lower():
                            modified_code = self._insert_in_main_method(modified_code,
                                "// ERROR: BUILD - Incompatible types - Assigning String to int\n        int value = \"hello\";")
                        
                        elif "missing return" in error_name.lower():
                            method_code = """
            // ERROR: BUILD - Missing return statement - Method missing return in some paths
            public int calculateValue(int input) {
                if (input > 0) {
                    return input * 2;
                }
                // Missing return statement for else case
            }
        """
                            modified_code = self._insert_in_class(modified_code, method_code)
                        
                        elif "null" in error_name.lower() or "nullpointer" in error_name.lower():
                            modified_code = self._insert_in_main_method(modified_code,
                                "// ERROR: BUILD - NullPointerException - Accessing method on null object\n        String str = null;\n        int length = str.length();")
                        
                        elif ("string" in error_name.lower() or "equals" in error_name.lower()) and "==" in error_name.lower():
                            modified_code = self._insert_in_main_method(modified_code,
                                "// ERROR: BUILD - String comparison using == - Using == instead of equals()\n        String s1 = \"hello\";\n        String s2 = \"h\" + \"ello\";\n        if (s1 == s2) {\n            System.out.println(\"Strings are same\");\n        }")
                        
                        else:
                            # Generic build error
                            modified_code = self._insert_in_main_method(modified_code,
                                f"// ERROR: BUILD - {error_name} - Generic implementation\n        // This code has a {error_name} error\n        throw new RuntimeException(\"Build error: {error_name}\");")
                    
                    elif error_type == "checkstyle":
                        # Handle common checkstyle error types
                        if "typename" in error_name.lower():
                            modified_code += """

        // ERROR: CHECKSTYLE - TypeName - Class name should start with uppercase
        class myClass {
            public void doSomething() {
                // Empty method
            }
        }
        """
                        
                        elif "membername" in error_name.lower():
                            modified_code = self._insert_in_class(modified_code,
                                "// ERROR: CHECKSTYLE - MemberName - Variable name should use lowerCamelCase\n    private String User_Name = \"John\";")
                        
                        elif "methodname" in error_name.lower():
                            modified_code = self._insert_in_class(modified_code,
                                "// ERROR: CHECKSTYLE - MethodName - Method name should use lowerCamelCase\n    public void PrintMessage() {\n        System.out.println(\"Hello\");\n    }")
                        
                        elif "whitespace" in error_name.lower():
                            modified_code = self._insert_in_main_method(modified_code,
                                "// ERROR: CHECKSTYLE - WhitespaceAround - Missing whitespace around operators\n        int x=5+3;")
                        
                        else:
                            # Generic checkstyle error - add empty code style comment
                            modified_code = self._insert_in_class(modified_code,
                                f"// ERROR: CHECKSTYLE - {error_name} - Generic implementation\n    // This violates {error_name} style guideline\n    public void badlyNamedMethod_withUnderscore() {{}}")
            
            # Use the modified code with added errors as both clean and annotated version
            clean_code = modified_code
            annotated_code = modified_code
            
            # Enrich the error information using the clean code
            enhanced_errors, detailed_problems = enrich_error_information(clean_code, selected_errors)
            
            return annotated_code, clean_code, enhanced_errors, detailed_problems
        
        except Exception as e:
            logger.error(f"Error in fallback code generation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Ultra-fallback - create minimal Java code with obvious errors
            annotated_code = """
        // Fallback Java code with errors
        public class Main {
            // ERROR: CHECKSTYLE - MemberName - Variable name should use lowerCamelCase
            private String User_Name = "John";
            
            public static void main(String[] args) {
                // ERROR: BUILD - NullPointerException - Accessing method on null object
                String str = null;
                int length = str.length();
                
                // ERROR: BUILD - Cannot find symbol - Using undeclared variable
                int result = undeclaredVar + 5;
            }
            
            // ERROR: BUILD - Missing return statement - Method missing return in some paths
            public int calculateValue(int input) {
                if (input > 0) {
                    return input * 2;
                }
                // Missing return statement for else case
            }
        }

        // ERROR: CHECKSTYLE - TypeName - Class name should start with uppercase
        class myClass {
            public void doSomething() {
                // Empty method
            }
        }
        """
            clean_code = """
        public class Main {
            private String User_Name = "John";
            
            public static void main(String[] args) {
                String str = null;
                int length = str.length();
                
                int result = undeclaredVar + 5;
            }
            
            public int calculateValue(int input) {
                if (input > 0) {
                    return input * 2;
                }
            }
        }

        class myClass {
            public void doSomething() {
            }
        }
        """
            
            # Create some basic enhanced errors
            error1 = {"type": "checkstyle", "name": "MemberName", "line_number": 3, "line_content": 'private String User_Name = "John";'}
            error2 = {"type": "build", "name": "NullPointerException", "line_number": 7, "line_content": "int length = str.length();"}
            error3 = {"type": "build", "name": "Cannot find symbol", "line_number": 9, "line_content": "int result = undeclaredVar + 5;"}
            enhanced_errors = [error1, error2, error3]
            
            # Create problem descriptions
            problem1 = "CHECKSTYLE ERROR - MemberName: Member variable names should use lowerCamelCase"
            problem2 = "BUILD ERROR - NullPointerException: Object accessed without null check"
            problem3 = "BUILD ERROR - Cannot find symbol: Using variable that hasn't been declared"
            detailed_problems = [problem1, problem2, problem3]
            
            return annotated_code, clean_code, enhanced_errors, detailed_problems

    def _insert_in_main_method(self, code: str, insertion: str) -> str:
        """
        Insert code into the main method of a Java class.
        
        Args:
            code: The original code
            insertion: The code to insert
            
        Returns:
            Updated code with the insertion
        """
        lines = code.splitlines()
        
        # Look for main method
        main_start = -1
        main_brace_count = 0
        found_open_brace = False
        
        for i, line in enumerate(lines):
            if "public static void main" in line:
                main_start = i
            
            if main_start != -1 and i >= main_start:
                if "{" in line:
                    found_open_brace = True
                    main_brace_count += line.count("{")
                if "}" in line:
                    main_brace_count -= line.count("}")
                
                # Insert after the opening brace of the main method
                if found_open_brace and main_brace_count == 1 and "public static void main" not in line:
                    # Insert after this line
                    lines.insert(i + 1, insertion)
                    return "\n".join(lines)
        
        # If no main method found, create one
        if main_start == -1:
            # Look for a class definition instead
            class_start = -1
            class_brace_count = 0
            
            for i, line in enumerate(lines):
                if "class" in line and "{" in line:
                    class_start = i
                    class_brace_count = 1
                    break
            
            if class_start != -1:
                # Find where to insert the main method in the class
                insert_idx = class_start + 1
                
                # Skip past any class-level fields or methods to find a good insertion point
                while insert_idx < len(lines) and class_brace_count > 0:
                    if "{" in lines[insert_idx]:
                        class_brace_count += lines[insert_idx].count("{")
                    if "}" in lines[insert_idx]:
                        class_brace_count -= lines[insert_idx].count("}")
                        
                        # If we're back to brace level 1, we're at class level
                        if class_brace_count == 1:
                            # Insert main method after this line
                            main_method = f"""
        public static void main(String[] args) {{
    {insertion}
        }}"""
                            lines.insert(insert_idx + 1, main_method)
                            return "\n".join(lines)
                    
                    insert_idx += 1
                
                # If we didn't find a good spot, just add the main method at the end of the class
                # Find the class closing brace
                for i in range(len(lines) - 1, class_start, -1):
                    if "}" in lines[i] and not "{" in lines[i]:
                        main_method = f"""
        public static void main(String[] args) {{
    {insertion}
        }}"""
                        lines.insert(i, main_method)
                        return "\n".join(lines)
        
        # If everything else fails, create a basic class with main method
        new_code = f"""public class Main {{
        public static void main(String[] args) {{
    {insertion}
        }}
    }}"""
        
        # Append to existing code if it exists
        if code.strip():
            return code + "\n\n" + new_code
        else:
            return new_code

    def _insert_in_class(self, code: str, insertion: str) -> str:
        """
        Insert code inside a Java class definition.
        
        Args:
            code: The original code
            insertion: The code to insert
            
        Returns:
            Updated code with the insertion
        """
        lines = code.splitlines()
        
        # Look for class definition
        class_start = -1
        
        for i, line in enumerate(lines):
            if "class" in line and "{" in line:
                class_start = i
                # Insert after the class declaration line
                lines.insert(i + 1, insertion)
                return "\n".join(lines)
        
        # If no class found, create one
        if class_start == -1:
            new_code = f"""public class Main {{
    {insertion}

        public static void main(String[] args) {{
            System.out.println("Hello, World!");
        }}
    }}"""
            
            # Append to existing code if it exists
            if code.strip():
                return code + "\n\n" + new_code
            else:
                return new_code
        
    def _generate_comparison_report(self, known_problems: List[str], review_analysis: Dict[str, Any]) -> str:
        """Generate a comparison report between student review and known problems."""
        return generate_comparison_report(known_problems, review_analysis)
    
    # Public interface methods
    def get_all_error_categories(self) -> Dict[str, List[str]]:
        """Get all available error categories."""
        return self.error_repository.get_all_categories()
    
    def submit_review(self, state: WorkflowState, student_review: str) -> WorkflowState:
        """
        Submit a student review and update the state.
        This method is called from the UI when a student submits a review.
        """
        # Create a new review attempt
        review_attempt = ReviewAttempt(
            student_review=student_review,
            iteration_number=state.current_iteration,
            analysis={},
            targeted_guidance=None
        )
        
        # Add to review history
        state.review_history.append(review_attempt)
        
        # Run the state through the analyze_review node
        updated_state = self.analyze_review_node(state)
        
        return updated_state
    
    def reset(self) -> WorkflowState:
        """Reset the workflow to initial state."""
        return WorkflowState()
    
    def regenerate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Regenerate code based on evaluation feedback.
        Uses the feedback from code evaluation to improve code generation.
        """
        try:
            logger.info(f"Starting code regeneration (Attempt {state.evaluation_attempts})")
            
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            selected_error_categories = state.selected_error_categories
            
            # Print regeneration info for debugging
            print(f"\n========== REGENERATING CODE (Attempt {state.evaluation_attempts}) ==========")
            print(f"Using feedback from evaluation to improve code generation")
            
            # Get errors from selected categories or use the ones from previous attempt
            requested_errors = []
            if hasattr(state.code_snippet, "raw_errors"):
                for error_type in state.code_snippet.raw_errors:
                    requested_errors.extend(state.code_snippet.raw_errors[error_type])
            else:
                # Fallback to getting errors from categories if raw_errors not available
                selected_errors, _ = self.error_repository.get_errors_for_llm(
                    selected_categories=selected_error_categories,
                    count=get_error_count_for_difficulty(difficulty_level),
                    difficulty=difficulty_level
                )
                requested_errors = selected_errors
            
            # Use the code generation feedback to generate improved code
            feedback_prompt = state.code_generation_feedback
            
            # Export the feedback prompt
            try:
                from utils.export_utils import export_prompt_response
                export_prompt_response(
                    prompt=feedback_prompt,
                    response="",
                    operation_type=f"regeneration_prompt_attempt_{state.evaluation_attempts}",
                    error_list=requested_errors
                )
            except Exception as export_err:
                logger.error(f"Error exporting regeneration prompt: {str(export_err)}")
            
            # Log the prompt for debugging
            print("\n========== REGENERATION PROMPT ==========")
            print(feedback_prompt[:500] + "..." if len(feedback_prompt) > 500 else feedback_prompt)
            
            # Generate code with feedback prompt
            if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
                # Generate code using the improved prompt
                response = self.code_generator.llm.invoke(feedback_prompt)
                
                # Export the response
                try:
                    from utils.export_utils import export_prompt_response
                    export_prompt_response(
                        prompt=feedback_prompt,
                        response=str(response),
                        operation_type=f"regeneration_response_attempt_{state.evaluation_attempts}",
                        error_list=requested_errors
                    )
                except Exception as export_err:
                    logger.error(f"Error exporting regeneration response: {str(export_err)}")
                
                # Extract the code with annotations
                annotated_code = extract_code_from_response(response)
                
                # Create clean version by stripping annotations
                clean_code = strip_error_annotations(annotated_code)
                
                # Evaluate the code immediately to ensure it contains the requested errors
                # This is the key improvement to ensure evaluation completes within each attempt
                if self.code_evaluation_agent:
                    immediate_evaluation = self.code_evaluation_agent.evaluate_code(
                        clean_code, requested_errors
                    )
                    
                    # Export the immediate evaluation results
                    try:
                        from utils.export_utils import export_prompt_response
                        export_prompt_response(
                            prompt="", 
                            response="", 
                            operation_type=f"immediate_evaluation_attempt_{state.evaluation_attempts}",
                            error_list=requested_errors,
                            evaluation_result=immediate_evaluation
                        )
                    except Exception as export_err:
                        logger.error(f"Error exporting immediate evaluation: {str(export_err)}")
                    
                    # Print immediate evaluation results for debugging
                    print("\n========== IMMEDIATE EVALUATION RESULTS ==========")
                    print(f"Valid: {immediate_evaluation.get('valid', False)}")
                    print(f"Found Errors: {len(immediate_evaluation.get('found_errors', []))} of {len(requested_errors)}")
                    print(f"Missing Errors: {immediate_evaluation.get('missing_errors', [])}")
                
                # Enrich the error information 
                enhanced_errors, detailed_problems = enrich_error_information(
                    clean_code, requested_errors
                )
                
                # Create updated code snippet
                state.code_snippet = CodeSnippet(
                    code=annotated_code,
                    clean_code=clean_code,
                    known_problems=detailed_problems,
                    raw_errors={
                        "build": [e for e in requested_errors if e.get("type") == "build"],
                        "checkstyle": [e for e in requested_errors if e.get("type") == "checkstyle"]
                    },
                    enhanced_errors=enhanced_errors
                )
                
                # Store the evaluation result in the state for immediate feedback
                if self.code_evaluation_agent:
                    state.evaluation_result = immediate_evaluation
                
                # Move to evaluation step again
                state.current_step = "evaluate"
                logger.info(f"Code regenerated successfully on attempt {state.evaluation_attempts}")
                
                return state
            else:
                # If no LLM available, fall back to standard generation
                logger.warning("No LLM available for regeneration. Falling back to standard generation.")
                return self.generate_code_node(state)
                
        except Exception as e:
            logger.error(f"Error regenerating code: {str(e)}")
            import traceback
            traceback.print_exc()
            state.error = f"Error regenerating code: {str(e)}"
            return state
        
    def evaluate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate generated code to ensure it contains the requested errors.
        Acts as a dedicated node in the LangGraph workflow.
        Uses LLM for more accurate evaluation when available.
        """
        try:
            logger.info("Starting code evaluation node")
            
            # Get code snippet from state
            if not state.code_snippet:
                state.error = "No code snippet available for evaluation"
                return state
                
            # Get the clean code (without annotations)
            code = state.code_snippet.clean_code
            
            # Get requested errors from state
            requested_errors = []
            if hasattr(state.code_snippet, "raw_errors"):
                for error_type in state.code_snippet.raw_errors:
                    requested_errors.extend(state.code_snippet.raw_errors[error_type])
            
            # Initialize the evaluation agent if not already done
            if not hasattr(self, 'code_evaluation_agent') or self.code_evaluation_agent is None:
                try:
                    from utils.code_evaluation_agent import CodeEvaluationAgent
                    
                    # Get the appropriate LLM for code evaluation
                    evaluation_llm = self.llm_manager.initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
                    
                    # Create the code evaluation agent with the LLM
                    self.code_evaluation_agent = CodeEvaluationAgent(llm=evaluation_llm)
                    logger.info("Created Code Evaluation Agent with LLM for evaluation node")
                except Exception as e:
                    logger.error(f"Error initializing Code Evaluation Agent: {str(e)}")
                    state.error = f"Error initializing Code Evaluation Agent: {str(e)}"
                    return state
            
            # Evaluate the code
            evaluation_result = self.code_evaluation_agent.evaluate_code(
                code, requested_errors
            )
            
            # Update state with evaluation results
            state.evaluation_result = evaluation_result
            state.evaluation_attempts += 1
            
            # Log evaluation results
            logger.info(f"Code evaluation complete: {len(evaluation_result.get('found_errors', []))} " +
                    f"of {len(requested_errors)} errors implemented")
            
            # If evaluation passed (all errors implemented), move to review
            if evaluation_result["valid"]:
                state.current_step = "review"
                logger.info("All errors successfully implemented, proceeding to review")
            else:
                # Generate feedback for code regeneration
                feedback = self.code_evaluation_agent.generate_improved_prompt(
                    code, requested_errors, evaluation_result
                )
                state.code_generation_feedback = feedback
                
                # Check if we've reached max attempts
                if state.evaluation_attempts >= state.max_evaluation_attempts:
                    # If we've reached max attempts, proceed to review anyway
                    state.current_step = "review"
                    logger.warning(f"Reached maximum evaluation attempts ({state.max_evaluation_attempts}). Proceeding to review.")
                else:
                    # Otherwise, set the step to regenerate code
                    state.current_step = "regenerate"
                    logger.info(f"Evaluation attempt {state.evaluation_attempts}: Feedback generated for regeneration")
            
            return state
            
        except Exception as e:
            logger.error(f"Error evaluating code: {str(e)}")
            import traceback
            traceback.print_exc()
            state.error = f"Error evaluating code: {str(e)}"
            return state
        
   