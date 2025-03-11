"""
LangGraph Workflow for Java Peer Review Training System.

This module implements the code review workflow as a LangGraph graph.
"""

__all__ = ['JavaCodeReviewGraph']

import logging
from typing import Dict, List, Any, Annotated, TypedDict, cast
from langgraph.graph import StateGraph, END
from state_schema import WorkflowState, CodeSnippet, ReviewAttempt

# Import domain-specific components
from core.code_generator import CodeGenerator
from core.student_response_evaluator import StudentResponseEvaluator
from core.feedback_manager import FeedbackManager
from data.json_error_repository import JsonErrorRepository
from llm_manager import LLMManager

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
    def generate_code_node(self, state: WorkflowState) -> WorkflowState:
        """Generate Java code with errors node."""
        try:
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            selected_error_categories = state.selected_error_categories
            
            # Generate code with errors
            selected_errors = []
            problem_descriptions = []
            
            # Get errors from selected categories
            errors = self.error_repository.get_errors_by_categories(selected_error_categories)
            
            # Generate code with selected errors
            code_with_errors = self._generate_code_with_errors(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=errors
            )
                        
            # Create problem descriptions
            for error_type, error_list in errors.items():
                for error in error_list:
                    if error_type == "build":
                        name = error.get("error_name", "Unknown error")
                        description = error.get("description", "")
                        category = error.get("category", "")
                        problem_descriptions.append(f"Build Error - {name}: {description} (Category: {category})")
                    else:  # checkstyle
                        name = error.get("check_name", "Unknown check")
                        description = error.get("description", "")
                        category = error.get("category", "")
                        problem_descriptions.append(f"Checkstyle Error - {name}: {description} (Category: {category})")
            
            # Create code snippet object
            code_snippet = CodeSnippet(
                code=code_with_errors,
                known_problems=problem_descriptions,
                raw_errors=errors  # This is now correctly typed as Dict[str, List[Dict[str, Any]]]
            )
            
            # Update state
            state.code_snippet = code_snippet
            state.current_step = "review"
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            state.error = f"Error generating code: {str(e)}"
            return state
    
    def review_code_node(self, state: WorkflowState) -> WorkflowState:
        """Review code node - this is a placeholder since user input happens in the UI."""
        # This node is primarily a placeholder since the actual review is submitted via the UI
        # The UI will call the submit_review method directly
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
            
            # Analyze the student review
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
            comparison_report = self._safe_generate_comparison_report(
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
    
    def _safe_generate_comparison_report(self, known_problems: List[str], review_analysis: Dict[str, Any]) -> str:
        """
        Safely generate a comparison report between student review and known problems.
        Handles type checking to prevent errors.
        
        Args:
            known_problems: List of known problems
            review_analysis: Analysis of the student review
            
        Returns:
            Comparison report text
        """
        try:
            # Create report header
            report = "# Detailed Comparison: Your Review vs. Actual Issues\n\n"
            
            # Problems section
            report += "## Code Issues Analysis\n\n"
            
            # Safely extract data from review analysis
            identified_problems = review_analysis.get("identified_problems", [])
            missed_problems = review_analysis.get("missed_problems", [])
            false_positives = review_analysis.get("false_positives", [])
            
            # Ensure all problems are properly converted to strings
            known_problems_str = [str(p) if not isinstance(p, str) else p for p in known_problems]
            identified_problems_str = [str(p) if not isinstance(p, str) else p for p in identified_problems]
            missed_problems_str = [str(p) if not isinstance(p, str) else p for p in missed_problems]
            false_positives_str = [str(p) if not isinstance(p, str) else p for p in false_positives]
            
            # Issues found correctly
            if identified_problems_str:
                report += "### Issues You Identified Correctly\n\n"
                for i, problem in enumerate(identified_problems_str, 1):
                    report += f"**{i}. {problem}**\n\n"
                    report += "Great job finding this issue! "
                    report += "This demonstrates your understanding of this type of problem.\n\n"
            
            # Issues missed
            if missed_problems_str:
                report += "### Issues You Missed\n\n"
                for i, problem in enumerate(missed_problems_str, 1):
                    report += f"**{i}. {problem}**\n\n"
                    report += "You didn't identify this issue. "
                    
                    # Add some specific guidance based on the problem type
                    problem_lower = problem.lower()
                    if "null" in problem_lower:
                        report += "When reviewing code, always check for potential null references and proper null handling.\n\n"
                    elif "naming" in problem_lower or "convention" in problem_lower:
                        report += "Pay attention to naming conventions in Java. Classes should use UpperCamelCase, while methods and variables should use lowerCamelCase.\n\n"
                    elif "javadoc" in problem_lower or "comment" in problem_lower:
                        report += "Remember to check for proper documentation. Methods should have complete Javadoc comments with @param and @return tags where appropriate.\n\n"
                    elif "exception" in problem_lower or "throw" in problem_lower:
                        report += "Always verify that exceptions are either caught or declared in the method signature with 'throws'.\n\n"
                    elif "loop" in problem_lower or "condition" in problem_lower:
                        report += "Carefully examine loop conditions for off-by-one errors or potential infinite loops.\n\n"
                    else:
                        report += "This is something to look for in future code reviews.\n\n"
            
            # Calculate some metrics
            total_problems = len(known_problems_str)
            identified_count = len(identified_problems_str)
            missed_count = len(missed_problems_str)
            false_positive_count = len(false_positives_str)
            
            accuracy = (identified_count / total_problems * 100) if total_problems > 0 else 0
            
            # Overall assessment
            report += "### Overall Assessment\n\n"
            
            if accuracy >= 80:
                report += "**Excellent review!** You found most of the issues in the code.\n\n"
            elif accuracy >= 60:
                report += "**Good review.** You found many issues, but missed some important ones.\n\n"
            elif accuracy >= 40:
                report += "**Fair review.** You found some issues, but missed many important ones.\n\n"
            else:
                report += "**Needs improvement.** You missed most of the issues in the code.\n\n"
            
            report += f"- You identified {identified_count} out of {total_problems} issues ({accuracy:.1f}%)\n"
            report += f"- You missed {missed_count} issues\n"
            report += f"- You incorrectly identified {false_positive_count} non-issues\n\n"
            
            return report
        
        except Exception as e:
            logger.error(f"Error generating comparison report: {str(e)}")
            return "Error generating comparison report. Your review was processed, but we couldn't generate a detailed comparison."
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
    def _generate_code_with_errors(self, code_length: str, difficulty_level: str, selected_errors: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate Java code with the selected errors."""
        # Flatten the errors into a single list
        flat_errors = []
        for error_type, errors in selected_errors.items():
            for error in errors:
                flat_errors.append({
                    "type": error_type,
                    **error
                })
        
        # Use the code generator to create code with errors
        if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
            # Create a detailed prompt for the LLM
            prompt = self._create_code_generation_prompt(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=flat_errors
            )
            
            # Generate code with errors
            response = self.code_generator.llm.invoke(prompt)
            code = self._extract_code_from_response(response)
            
            if code and len(code.strip()) > 50:
                return code
        
        # Fallback: generate clean code and manually note errors
        base_code = self.code_generator.generate_java_code(
            code_length=code_length,
            difficulty_level=difficulty_level
        )

        return self._add_error_comments(base_code, flat_errors)
    
    def _create_code_generation_prompt(self, code_length: str, difficulty_level: str, selected_errors: List[Dict[str, Any]]) -> str:
        """Create a prompt for generating code with errors."""
        # Implementation similar to the original _create_direct_code_generation_prompt method
        # (code omitted for brevity)
        return "Detailed prompt for generating Java code with errors"
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response."""
        import re
        code_blocks = re.findall(r'```(?:java)?\s*(.*?)\s*```', response, re.DOTALL)
        
        if code_blocks:
            return max(code_blocks, key=len)
        
        return response.strip()
    
    def _add_error_comments(self, code: str, errors: List[Dict[str, Any]]) -> str:
        """Add error comments to code as fallback."""
        lines = code.split('\n')
        
        for i, error in enumerate(errors):
            error_type = error.get("type", "unknown")
            name = error.get("error_name", error.get("check_name", "unknown error"))
            description = error.get("description", "")
            
            position = min(5 + i * 3, len(lines) - 1)
            comment = f"// TODO: Fix {error_type} error: {name} - {description}"
            
            lines.insert(position, comment)
        
        return '\n'.join(lines)
    
    def _generate_comparison_report(self, known_problems: List[str], review_analysis: Dict[str, Any]) -> str:
        """Generate a comparison report between student review and known problems."""
        # Implementation similar to the original _generate_comparison_report method
        # (code omitted for brevity)
        report = "# Detailed Comparison: Your Review vs. Actual Issues\n\n"
        # ... rest of implementation
        return report
    
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