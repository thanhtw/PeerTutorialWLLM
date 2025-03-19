"""
LangGraph Workflow for Java Peer Review Training System.

This module implements the code review workflow as a LangGraph graph.
"""

__all__ = ['JavaCodeReviewGraph']

import logging
import os
import random
import re
from typing import Dict, List, Any, Annotated, TypedDict, Tuple, cast, Optional
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
    add_error_comments,
    get_error_count_for_difficulty,
    generate_comparison_report,
    strip_error_annotations
)
from utils.enhanced_error_tracking import enrich_error_information

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
        else:
            # Initialize domain objects without LLMs
            self.code_generator = CodeGenerator()
            self.evaluator = StudentResponseEvaluator()
            self.feedback_manager = FeedbackManager(self.evaluator)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create a new graph with our state schema
        workflow = StateGraph(WorkflowState)
        
        # Define nodes
        workflow.add_node("generate_code", self.generate_code_node)
        workflow.add_node("review_code", self.review_code_node)
        workflow.add_node("analyze_review", self.analyze_review_node)
        workflow.add_node("generate_summary", self.generate_summary_node)
        
        # Define edges
        workflow.add_edge("generate_code", "review_code")
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
    
    # Node implementations
    # Update the generate_code_node method in langgraph_workflow.py

    def generate_code_node(self, state: WorkflowState) -> WorkflowState:
        """Generate Java code with errors node with enhanced debugging for all modes."""
        try:
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            selected_error_categories = state.selected_error_categories
            
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
            
            # Generate code with selected errors - now returns both versions
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
    def should_continue_review(self, state: WorkflowState) -> str:
        """
        Determine if we should continue with another review iteration or generate summary.
        """
        # Check if we've reached max iterations
        if state.current_iteration > state.max_iterations:
            return "generate_summary"
        
        # Check if the review is sufficient
        if state.review_sufficient:
            return "generate_summary"
        
        # Otherwise, continue reviewing
        return "continue_review"
    
    # Helper methods
    # Replace the _generate_code_with_errors method in langgraph_workflow.py

    # In langgraph_workflow.py, update the _generate_code_with_errors method:

    def _generate_code_with_errors(self, code_length: str, difficulty_level: str, selected_errors: List[Dict[str, Any]]) -> Tuple[str, str, List[Dict[str, Any]], List[str]]:
        """
        Generate Java code with the selected errors, returning both annotated and clean versions.
        
        Args:
            code_length: Length of code (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            selected_errors: List of selected errors to include in the code
        
        Returns:
            Tuple of (annotated_code, clean_code, enhanced_errors, detailed_problems)
        """
        # Use the code generator to create code with errors
        if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
            # Create a detailed prompt for the LLM
            prompt = create_code_generation_prompt(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=selected_errors,
                include_error_annotations=True  # Generate code with annotations
            )
            
            # Print the prompt for debugging
            print("\n========== CODE GENERATION PROMPT ==========")
            print(prompt)
            
            # Generate code with errors
            response = self.code_generator.llm.invoke(prompt)
            
            # Print the LLM response for debugging
            print("\n========== LLM RESPONSE ==========")
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
                
                # Enrich the error information using the clean code
                enhanced_errors, detailed_problems = enrich_error_information(clean_code, selected_errors)
                
                return annotated_code, clean_code, enhanced_errors, detailed_problems
        
        # Fallback: generate clean code and manually note errors
        base_code = self.code_generator.generate_java_code(
            code_length=code_length,
            difficulty_level=difficulty_level
        )
        
        # Print fallback generation for debugging
        print("\n========== FALLBACK CODE GENERATION ==========")
        print(base_code)
        
        # Create clean code (same as base code in this case, but without comments)
        clean_code = strip_error_annotations(base_code)
        
        # For fallback, keep the base code as annotated version
        annotated_code = base_code
        
        # Even for fallback, try to enrich the error information
        enhanced_errors, detailed_problems = enrich_error_information(clean_code, selected_errors)
        
        return annotated_code, clean_code, enhanced_errors, detailed_problems
        
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