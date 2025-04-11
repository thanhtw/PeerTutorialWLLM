"""
Improved LangGraph Workflow for Java Peer Review Training System with MCP integration.

This module implements the enhanced code review workflow as a LangGraph graph,
using the Model Control Protocol (MCP) for managing model interactions.
"""

import logging
import os
from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
from state_schema import WorkflowState, CodeSnippet, ReviewAttempt

# Import MCP client
from mcp_client import MCPClient

# Import domain-specific components
from core.student_response_evaluator import StudentResponseEvaluator
from core.feedback_manager import FeedbackManager
from data.json_error_repository import JsonErrorRepository
from llm_manager import LLMManager

from utils.improved_export_utils import export_prompt_response, generate_session_id

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPJavaCodeReviewGraph:
    """
    Enhanced LangGraph implementation of the Java Code Review workflow using MCP.
    
    This implementation uses the Model Control Protocol for model interactions,
    providing better separation of concerns and flexibility in model management.
    """
    
    def __init__(self, mcp_client: MCPClient = None, llm_manager: LLMManager = None):
        """Initialize the graph with MCP client and domain components."""
        # Initialize MCP client if not provided
        self.mcp_client = mcp_client or MCPClient()
        
        # Initialize LLM Manager if not provided
        self.llm_manager = llm_manager or LLMManager()
        
        # Initialize repositories
        self.error_repository = JsonErrorRepository()
        
        # Generate a unique workflow ID (session_id for tracking)
        self.workflow_id = generate_session_id()
        logger.info(f"Initialized workflow with ID: {self.workflow_id}")
        
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
            review_model = self.llm_manager.initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
            
            # Initialize domain objects with models
            self.evaluator = StudentResponseEvaluator(review_model) if review_model else StudentResponseEvaluator()
            self.feedback_manager = FeedbackManager(self.evaluator)
            
            logger.info("Successfully initialized domain components with LLMs")
        else:
            # Initialize domain objects without LLMs
            self.evaluator = StudentResponseEvaluator()
            self.feedback_manager = FeedbackManager(self.evaluator)
            
            logger.warning("Initialized components without LLMs due to connection issues")
    
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
        """Generate Java code with errors using MCP."""
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
            
            # Log debug information
            logger.info(f"Generating code with mode: {error_selection_mode}")
            logger.info(f"Parameters: length={code_length}, difficulty={difficulty_level}")
            
            # Handle different modes
            if error_selection_mode == "specific" and hasattr(state, "selected_specific_errors"):
                # Specific mode - use exact errors
                selected_errors = state.selected_specific_errors
                logger.info(f"Specific Mode: {len(selected_errors)} errors selected")
                
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
                from utils.code_utils import get_error_count_for_difficulty
                selected_errors, problem_descriptions = self.error_repository.get_errors_for_llm(
                    selected_categories=selected_error_categories,
                    count=get_error_count_for_difficulty(difficulty_level),
                    difficulty=difficulty_level
                )
                
                logger.info(f"Category Mode: {len(selected_errors)} errors selected from categories")
            
            # Generate code with the selected errors using MCP
            domain = getattr(state, "domain", "student_management")
            
            # Call MCP to generate code
            annotated_code = self.mcp_client.generate_java_code(
                code_length=code_length,
                difficulty_level=difficulty_level,
                error_types=selected_errors,
                domain=domain
            )
            
            # Process the result
            from utils.code_utils import strip_error_annotations
            clean_code = strip_error_annotations(annotated_code)
            
            # Extract problem descriptions
            detailed_problems = []
            for line in annotated_code.splitlines():
                if "// ERROR:" in line:
                    error_desc = line.split("// ERROR:")[1].strip()
                    detailed_problems.append(error_desc)
            
            # If no error annotations found, create from error names
            if not detailed_problems:
                for error in selected_errors:
                    error_type = error.get("type", "").upper()
                    name = error.get("name", "")
                    description = error.get("description", "")
                    detailed_problems.append(f"{error_type} ERROR - {name}: {description}")
            
            # Create code snippet object
            code_snippet = CodeSnippet(
                code=annotated_code,  # Store annotated version with error comments
                clean_code=clean_code,  # Store clean version without error comments
                known_problems=detailed_problems,  # Use detailed problems
                raw_errors={
                    "build": [e for e in selected_errors if e.get("type", "") == "build"],
                    "checkstyle": [e for e in selected_errors if e.get("type", "") == "checkstyle"]
                },
                enhanced_errors=selected_errors  # Store selected errors directly
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
        Uses MCP for evaluation.
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
            
            # Evaluate the code using MCP
            evaluation_result = self.mcp_client.evaluate_java_code(
                code=code, 
                requested_errors=requested_errors
            )
            
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
        Regenerate code based on evaluation feedback using MCP.
        """
        try:
            logger.info(f"Starting code regeneration (Attempt {state.evaluation_attempts})")
            
            # Since we don't have direct regeneration in MCP yet, we'll do generation again with the same errors
            # This is a simplification - a true implementation would pass the feedback to MCP
            
            # Get the original parameters
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            domain = getattr(state, "domain", "student_management")
            
            # Get errors from previous attempt
            requested_errors = []
            if hasattr(state.code_snippet, "raw_errors"):
                for error_type in state.code_snippet.raw_errors:
                    requested_errors.extend(state.code_snippet.raw_errors[error_type])
            
            # Generate new code with the same errors
            annotated_code = self.mcp_client.generate_java_code(
                code_length=code_length,
                difficulty_level=difficulty_level,
                error_types=requested_errors,
                domain=domain
            )
            
            # Process the result
            from utils.code_utils import strip_error_annotations
            clean_code = strip_error_annotations(annotated_code)
            
            # Extract problem descriptions
            detailed_problems = []
            for line in annotated_code.splitlines():
                if "// ERROR:" in line:
                    error_desc = line.split("// ERROR:")[1].strip()
                    detailed_problems.append(error_desc)
            
            # If no error annotations found, use previous problems
            if not detailed_problems and hasattr(state.code_snippet, "known_problems"):
                detailed_problems = state.code_snippet.known_problems
            
            # Create updated code snippet
            state.code_snippet = CodeSnippet(
                code=annotated_code,
                clean_code=clean_code,
                known_problems=detailed_problems,
                raw_errors={
                    "build": [e for e in requested_errors if e.get("type") == "build"],
                    "checkstyle": [e for e in requested_errors if e.get("type") == "checkstyle"]
                },
                enhanced_errors=requested_errors  # Store selected errors directly
            )
            
            # Move to evaluation step again
            state.current_step = "evaluate"
            logger.info(f"Code regenerated successfully on attempt {state.evaluation_attempts}")
            
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
        """Analyze student review node using MCP."""
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
            
            # Analyze the student review using MCP
            analysis = self.mcp_client.analyze_student_review(
                code_snippet=code_snippet,
                known_problems=known_problems,
                student_review=student_review
            )
            
            # Update the review with analysis
            latest_review.analysis = analysis
            
            # Export the review analysis if export_debug is enabled
            export_prompt_response(
                prompt=student_review, 
                response=str(analysis), 
                operation_type="review_analysis",
                session_id=self.workflow_id
            )
            
            # Check if the review is sufficient
            review_sufficient = analysis.get("review_sufficient", False)
            state.review_sufficient = review_sufficient
            
            # Generate targeted guidance if needed
            if not review_sufficient and state.current_iteration < state.max_iterations:
                targeted_guidance = self.mcp_client.generate_targeted_guidance(
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review,
                    review_analysis=analysis,
                    iteration_count=state.current_iteration,
                    max_iterations=state.max_iterations
                )
                latest_review.targeted_guidance = targeted_guidance
                
                # Export the guidance if export_debug is enabled
                export_prompt_response(
                    prompt="", 
                    response=targeted_guidance, 
                    operation_type="targeted_guidance",
                    session_id=self.workflow_id
                )
            
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
        """Generate summary and comparison report node using MCP."""
        try:
            # Generate final feedback
            if not state.review_history:
                state.error = "No reviews available for summary"
                return state
                    
            # Create the review history in the expected format
            review_history = [review.dict() for review in state.review_history]
            
            # Generate feedback using MCP
            final_feedback = self.mcp_client.generate_final_feedback(
                code_snippet=state.code_snippet.code,
                known_problems=state.code_snippet.known_problems,
                review_history=review_history
            )
            
            # Generate comparison report
            comparison_report = self.mcp_client.generate_comparison_report(
                known_problems=state.code_snippet.known_problems,
                review_analysis=state.review_history[-1].analysis if state.review_history else {}
            )
            
            # Export the summary if export_debug is enabled
            export_prompt_response(
                prompt="", 
                response=final_feedback + "\n\n" + comparison_report, 
                operation_type="summary_report",
                session_id=self.workflow_id
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
        
        # Export the review submission if export_debug is enabled
        export_prompt_response(
            prompt="", 
            response=student_review, 
            operation_type="review_submission",
            session_id=self.workflow_id
        )
        
        # Run the state through the analyze_review node
        updated_state = self.analyze_review_node(state)
        
        return updated_state
    
    def reset(self) -> WorkflowState:
        """Reset the workflow to initial state with a new session ID."""
        # Generate a new workflow ID
        self.workflow_id = generate_session_id()
        logger.info(f"Reset workflow with new ID: {self.workflow_id}")
        
        return WorkflowState()