"""
LangGraph Workflow for Java Peer Review Training System.

This module implements the code review workflow as a LangGraph graph.
The workflow includes:
1. Code generation with different modes (standard, advanced, specific)
2. Code evaluation in a loop with regeneration until requirements are met or max attempts
3. Student review with analysis and feedback
"""

import logging
import os
from typing import Dict, List, Any
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
logging.basicConfig(level=logging.INFO)
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
            
            # Initialize the code evaluation agent
            try:                
                self.code_evaluation_agent = CodeEvaluationAgent(llm=generative_model)
                logger.info("Successfully initialized Code Evaluation Agent")
            except Exception as e:
                logger.error(f"Error initializing Code Evaluation Agent: {str(e)}")
                self.code_evaluation_agent = None
        else:
            # Initialize domain objects without LLMs
            self.code_generator = CodeGenerator()
            self.evaluator = StudentResponseEvaluator()
            self.feedback_manager = FeedbackManager(self.evaluator)
            
            # Initialize the code evaluation agent without LLM
            try:
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
        
        # Conditional edge based on evaluation result - the key decision point
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
        """Generate Java code with errors node with support for all modes."""
        try:
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            selected_error_categories = state.selected_error_categories
            
            # Get error selection mode
            error_selection_mode = getattr(state, "error_selection_mode", "standard")
            
            # Initialize counters
            state.evaluation_attempts = 0
            state.evaluation_result = None
            state.code_generation_feedback = None
            
            # Print debug information
            print(f"\n========== GENERATE_CODE_NODE (Mode: {error_selection_mode}) ==========")
            print(f"Code Length: {code_length}")
            print(f"Difficulty Level: {difficulty_level}")
            print(f"Selected Error Categories: {selected_error_categories}")
            
            # Handle different modes
            if error_selection_mode == "specific" and hasattr(state, "selected_specific_errors"):
                # Specific mode - use exact errors
                selected_errors = state.selected_specific_errors
                print(f"Specific Mode: {len(selected_errors)} errors selected")
                
            else:
                # Standard or Advanced mode - use categories
                if not selected_error_categories or (
                    not selected_error_categories.get("build", []) and 
                    not selected_error_categories.get("checkstyle", [])
                ):
                    # Require explicit selection
                    state.error = "No error categories selected. Please select at least one problem area or error category."
                    return state
                
                # Get errors from selected categories
                selected_errors, problem_descriptions = self.error_repository.get_errors_for_llm(
                    selected_categories=selected_error_categories,
                    count=get_error_count_for_difficulty(difficulty_level),
                    difficulty=difficulty_level
                )
                
                print(f"Category Mode: {len(selected_errors)} errors selected from categories")
            
            # Generate code with the selected errors
            annotated_code, clean_code, enhanced_errors, detailed_problems = self._generate_code_with_errors(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=selected_errors
            )
            
            # Create code snippet object
            code_snippet = CodeSnippet(
                code=annotated_code,  # Store annotated version with error comments
                clean_code=clean_code,  # Store clean version without error comments
                known_problems=detailed_problems,  # Use detailed problems
                raw_errors={
                    "build": [e for e in selected_errors if e.get("type", "") == "build"],
                    "checkstyle": [e for e in selected_errors if e.get("type", "") == "checkstyle"]
                },
                enhanced_errors=enhanced_errors
            )
            
            # Update state
            state.code_snippet = code_snippet
            state.current_step = "evaluate"  # Go to evaluation first
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            import traceback
            traceback.print_exc()
            state.error = f"Error generating code: {str(e)}"
            return state
    
    def generate_code_with_specific_errors(self, state: WorkflowState, specific_errors: List[Dict[str, Any]]) -> WorkflowState:
        """Generate Java code with specific errors selected by the user."""
        try:
            # Set the error selection mode
            state.error_selection_mode = "specific"
            state.selected_specific_errors = specific_errors
            
            # Use the standard generation method
            return self.generate_code_node(state)
            
        except Exception as e:
            logger.error(f"Error generating code with specific errors: {str(e)}")
            import traceback
            traceback.print_exc()
            state.error = f"Error generating code with specific errors: {str(e)}"
            return state
    
    def evaluate_code_node(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate generated code to ensure it contains the requested errors.
        Uses only LLM-based evaluation, with no regex fallback.
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
                    # Get the appropriate LLM for code evaluation
                    evaluation_llm = self.llm_manager.initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
                    
                    # Create the code evaluation agent with the LLM
                    self.code_evaluation_agent = CodeEvaluationAgent(llm=evaluation_llm)
                    logger.info("Created Code Evaluation Agent with LLM for evaluation node")
                except Exception as e:
                    logger.error(f"Error initializing Code Evaluation Agent: {str(e)}")
                    state.error = f"Error initializing Code Evaluation Agent: {str(e)}"
                    return state
            
            # Evaluate the code using only LLM-based evaluation
            if self.code_evaluation_agent.llm:
                evaluation_result = self.code_evaluation_agent._evaluate_with_llm(
                    code, requested_errors
                )
            else:
                logger.error("No LLM available for evaluation")
                evaluation_result = {
                    "valid": False,
                    "found_errors": [],
                    "missing_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                                    for error in requested_errors],
                    "error_locations": {},
                    "llm_feedback": "No LLM available for evaluation"
                }
            
            # Update state with evaluation results
            state.evaluation_result = evaluation_result
            state.evaluation_attempts += 1
            
            # Log evaluation results
            logger.info(f"Code evaluation complete: {len(evaluation_result.get('found_errors', []))} " +
                    f"of {len(requested_errors)} errors implemented")
            
            # If evaluation passed (all errors implemented), move to review
            if evaluation_result.get("valid", False):
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
            
            # Get errors from previous attempt
            requested_errors = []
            if hasattr(state.code_snippet, "raw_errors"):
                for error_type in state.code_snippet.raw_errors:
                    requested_errors.extend(state.code_snippet.raw_errors[error_type])
            
            # Use the code generation feedback to generate improved code
            feedback_prompt = state.code_generation_feedback
            
            # Log the prompt for debugging
            print("\n========== REGENERATION PROMPT ==========")
            print(feedback_prompt[:500] + "..." if len(feedback_prompt) > 500 else feedback_prompt)
            
            # Generate code with feedback prompt
            if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
                # Generate code using the improved prompt
                response = self.code_generator.llm.invoke(feedback_prompt)
                
                # Extract the code with annotations
                annotated_code = extract_code_from_response(response)
                
                # Create clean version by stripping annotations
                clean_code = strip_error_annotations(annotated_code)
                
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
                
                # Move to evaluation step again
                state.current_step = "evaluate"
                logger.info(f"Code regenerated successfully on attempt {state.evaluation_attempts}")
                
                return state
            else:
                # If no LLM available, fall back to standard generation
                logger.warning("No LLM available for regeneration")
                state.error = "No LLM available for code regeneration"
                return state
                
        except Exception as e:
            logger.error(f"Error regenerating code: {str(e)}")
            import traceback
            traceback.print_exc()
            state.error = f"Error regenerating code: {str(e)}"
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
            
            # Generate comparison report
            comparison_report = generate_comparison_report(
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
        This is the decision point matching the diagram (Max Interaction or Fixed Requirement).
        
        Returns:
            "regenerate_code" if we need to regenerate code based on evaluation feedback
            "review_code" if the code is valid or we've reached max attempts
        """
        # Check if current step is explicitly set to regenerate
        if state.current_step == "regenerate":
            return "regenerate_code"
        
        # Check if evaluation passed (Fixed Requirement)
        if state.evaluation_result and state.evaluation_result.get("valid", False):
            return "review_code"
        
        # Check if we've reached max attempts (Max Interaction)
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

    def _generate_code_with_errors(self, code_length: str, difficulty_level: str, selected_errors: List[Dict[str, Any]]) -> tuple:
        """
        Generate Java code with the selected errors, using the LLM to ensure quality.
        
        Args:
            code_length: Length of code (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            selected_errors: List of selected errors to include in the code
        
        Returns:
            Tuple of (annotated_code, clean_code, enhanced_errors, detailed_problems)
        """
        # Ensure valid errors
        if not selected_errors:
            logger.warning("No errors specified for code generation")
            selected_errors = []
        
        # Create a detailed prompt for the LLM
        prompt = create_code_generation_prompt(
            code_length=code_length,
            difficulty_level=difficulty_level,
            selected_errors=selected_errors,
            include_error_annotations=True  # Generate code with annotations
        )
        
        try:
            # Generate code with errors
            if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
                # Get response from LLM
                response = self.code_generator.llm.invoke(prompt)
                
                # Extract the code with annotations
                annotated_code = extract_code_from_response(response)
                
                # Create clean version by stripping annotations
                clean_code = strip_error_annotations(annotated_code)
                
                # Enrich the error information
                enhanced_errors, detailed_problems = enrich_error_information(
                    clean_code, selected_errors
                )
                
                return annotated_code, clean_code, enhanced_errors, detailed_problems
            else:
                # Fallback if no LLM available
                logger.warning("No LLM available for code generation")
                return (
                    "// No LLM available for code generation\npublic class Fallback {}",
                    "public class Fallback {}",
                    [],
                    ["No code generated - LLM not available"]
                )
                
        except Exception as e:
            logger.error(f"Error in code generation: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return minimal fallback
            return (
                f"// Error during code generation: {str(e)}\npublic class ErrorFallback {{}}",
                "public class ErrorFallback {}",
                [],
                [f"Error during code generation: {str(e)}"]
            )
    
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