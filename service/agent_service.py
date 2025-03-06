"""
Agent Service module for Java Peer Review Training System.

This module provides the AgentService class which coordinates between
the UI, domain objects, and LLM manager to execute the code review workflow.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

# Import domain classes
from core.code_generator import CodeGenerator
from core.error_injector import ErrorInjector
from core.student_response_evaluator import StudentResponseEvaluator
from core.feedback_manager import FeedbackManager

# Import data access
from data.json_error_repository import JsonErrorRepository

# Import LLM manager
from llm_manager import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentService:
    """
    Service that coordinates the code review workflow.
    
    This class manages the interaction between UI components, domain objects,
    and the LLM manager to execute the code review training workflow.
    """
    
    def __init__(self, llm_manager: LLMManager = None):
        """
        Initialize the AgentService with dependencies.
        
        Args:
            llm_manager: LLM Manager for language model access
        """
        # Initialize LLM Manager if not provided
        self.llm_manager = llm_manager or LLMManager()
        
        # Initialize repositories
        self.error_repository = JsonErrorRepository()
        
        # Initialize domain objects without LLMs first
        self.code_generator = CodeGenerator()
        self.error_injector = ErrorInjector()
        self.evaluator = StudentResponseEvaluator()
        self.feedback_manager = FeedbackManager(self.evaluator)
        
        # Check if Ollama is available and initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM models for domain objects if Ollama is available."""
        # Check Ollama connection
        connection_status, message = self.llm_manager.check_ollama_connection()
        
        if connection_status:
            try:
                # Initialize models for each component
                generative_model = self.llm_manager.initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
                review_model = self.llm_manager.initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
                
                # Set models for domain objects
                if generative_model:
                    self.code_generator = CodeGenerator(generative_model)
                    self.error_injector = ErrorInjector(generative_model)
                
                if review_model:
                    self.evaluator = StudentResponseEvaluator(review_model)
                    self.feedback_manager = FeedbackManager(self.evaluator)
                
                logger.info("Successfully initialized models for domain objects")
                
            except Exception as e:
                logger.error(f"Error initializing models: {str(e)}")
        else:
            logger.warning(f"Ollama not available: {message}")
    
    def get_all_error_categories(self) -> Dict[str, List[str]]:
        """
        Get all available error categories.
        
        Returns:
            Dictionary with 'build' and 'checkstyle' categories
        """
        return self.error_repository.get_all_categories()
    
    def generate_code_with_errors(self, 
                                 code_length: str = "medium",
                                 difficulty_level: str = "medium",
                                 selected_error_categories: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """
        Generate Java code with intentional errors.
        
        Args:
            code_length: Length of code (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            selected_error_categories: Dictionary with 'build' and 'checkstyle' keys,
                                     each containing a list of selected categories
            
        Returns:
            Dictionary with generated code, known problems, and error information
        """
        try:
            # Generate base Java code without errors
            logger.info(f"Generating Java code: {code_length} length, {difficulty_level} difficulty")
            code_snippet = self.code_generator.generate_java_code(
                code_length=code_length,
                difficulty_level=difficulty_level,
                domain="student_management"  # Could be parameterized in the future
            )
            
            # Select errors to inject based on categories
            if selected_error_categories is None or not any(selected_error_categories.values()):
                # Use default categories if none specified
                selected_error_categories = {
                    "build": ["CompileTimeErrors", "RuntimeErrors", "LogicalErrors"],
                    "checkstyle": ["NamingConventionChecks", "WhitespaceAndFormattingChecks", "JavadocChecks"]
                }
            
            # Select errors from repository
            errors = self.error_injector.select_errors(
                selected_categories=selected_error_categories,
                difficulty_level=difficulty_level
            )
            
            # Inject errors into the code
            logger.info(f"Injecting {len(errors)} errors into code")
            code_with_errors, problem_descriptions = self.error_injector.inject_errors(
                code=code_snippet,
                errors=errors
            )
            
            # Start a new feedback session
            self.feedback_manager.start_new_review_session(
                code_snippet=code_with_errors,
                known_problems=problem_descriptions
            )
            
            return {
                "code_snippet": code_with_errors,
                "known_problems": problem_descriptions,
                "raw_errors": errors
            }
            
        except Exception as e:
            logger.error(f"Error generating code with errors: {str(e)}")
            return {
                "error": f"Error generating code: {str(e)}"
            }
    
    def process_student_review(self, student_review: str) -> Dict[str, Any]:
        """
        Process a student's review of code problems.
        
        Args:
            student_review: The student's review comments
            
        Returns:
            Dictionary with analysis results and next steps
        """
        try:
            # Submit the review to the feedback manager
            result = self.feedback_manager.submit_review(student_review)
            
            # Add some context information
            result["current_step"] = "wait_for_review" if result["next_steps"] == "iterate" else "summarize_review"
            
            # Generate summary and comparison if review is sufficient or max iterations reached
            if result["next_steps"] == "summarize":
                # Generate final feedback
                final_feedback = self.feedback_manager.generate_final_feedback()
                result["review_summary"] = final_feedback
                
                # Generate comparison report
                result["comparison_report"] = self._generate_comparison_report()
                
                # Mark as complete
                result["current_step"] = "complete"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing student review: {str(e)}")
            return {
                "error": f"Error processing review: {str(e)}"
            }
    
    def _generate_comparison_report(self) -> str:
        """
        Generate a comparison report between student review and known problems.
        
        Returns:
            Comparison report text
        """
        # Get the latest review
        latest_review = self.feedback_manager.get_latest_review()
        
        if not latest_review:
            return "No review data available for comparison."
        
        # Create a detailed comparison report
        report = "# Detailed Comparison: Your Review vs. Actual Issues\n\n"
        
        # Problems section
        report += "## Code Issues Analysis\n\n"
        
        # Get the problems and analysis
        known_problems = self.feedback_manager.known_problems
        review_analysis = latest_review["review_analysis"]
        
        identified_problems = review_analysis.get("identified_problems", [])
        missed_problems = review_analysis.get("missed_problems", [])
        false_positives = review_analysis.get("false_positives", [])
        
        # Issues found correctly
        if identified_problems:
            report += "### Issues You Identified Correctly\n\n"
            for i, problem in enumerate(identified_problems, 1):
                report += f"**{i}. {problem}**\n\n"
                report += "Great job finding this issue! "
                report += "This demonstrates your understanding of this type of problem.\n\n"
        
        # Issues missed
        if missed_problems:
            report += "### Issues You Missed\n\n"
            for i, problem in enumerate(missed_problems, 1):
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
        
        # False positives
        if false_positives:
            report += "### Issues You Incorrectly Identified\n\n"
            for i, problem in enumerate(false_positives, 1):
                report += f"**{i}. {problem}**\n\n"
                report += "This wasn't actually an issue in the code. "
                report += "Be careful not to flag correct code as problematic.\n\n"
        
        # Review patterns and advice
        report += "## Review Patterns and Advice\n\n"
        
        # Calculate some metrics
        total_problems = len(known_problems)
        identified_count = len(identified_problems)
        missed_count = len(missed_problems)
        false_positive_count = len(false_positives)
        
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
        
        # Improvement tips
        report += "### Tips for Improvement\n\n"
        
        # Based on missed problems, categorize what types of issues they missed
        missed_categories = set()
        for problem in missed_problems:
            problem_lower = problem.lower()
            
            if any(term in problem_lower for term in ["null", "nullpointer"]):
                missed_categories.add("Null pointer checks")
            elif any(term in problem_lower for term in ["array", "index", "bounds"]):
                missed_categories.add("Array bounds checking")
            elif any(term in problem_lower for term in ["exception", "throw", "catch"]):
                missed_categories.add("Exception handling")
            elif any(term in problem_lower for term in ["name", "convention", "identifier"]):
                missed_categories.add("Naming conventions")
            elif any(term in problem_lower for term in ["comment", "javadoc", "document"]):
                missed_categories.add("Code documentation")
            elif any(term in problem_lower for term in ["whitespace", "format", "indent"]):
                missed_categories.add("Code formatting")
            elif any(term in problem_lower for term in ["loop", "condition", "if", "while"]):
                missed_categories.add("Control flow")
            elif any(term in problem_lower for term in ["type", "cast", "conversion"]):
                missed_categories.add("Type safety")
            else:
                missed_categories.add("General code quality")
        
        if missed_categories:
            report += "Focus on improving these areas in future reviews:\n\n"
            for category in missed_categories:
                report += f"- **{category}**: "
                
                if category == "Null pointer checks":
                    report += "Always verify objects are not null before accessing their methods or properties.\n"
                elif category == "Array bounds checking":
                    report += "Check that array indices are within valid ranges to prevent ArrayIndexOutOfBoundsExceptions.\n"
                elif category == "Exception handling":
                    report += "Ensure exceptions are properly caught or declared in method signatures.\n"
                elif category == "Naming conventions":
                    report += "Verify that classes, methods, and variables follow Java naming conventions.\n"
                elif category == "Code documentation":
                    report += "Look for missing or incomplete Javadoc comments on classes and methods.\n"
                elif category == "Code formatting":
                    report += "Check for consistent whitespace, indentation, and brace placement.\n"
                elif category == "Control flow":
                    report += "Carefully examine loop conditions and if statements for logical errors.\n"
                elif category == "Type safety":
                    report += "Watch for incorrect type conversions or assignments between incompatible types.\n"
                else:
                    report += "Look for violations of general Java best practices and code quality standards.\n"
        
        # Add a conclusion
        report += "\n## Next Steps\n\n"
        report += "To improve your code review skills:\n\n"
        report += "1. **Practice systematically**: Develop a checklist of common issues to look for\n"
        report += "2. **Study Java best practices**: Learn more about Java coding standards and conventions\n"
        report += "3. **Read good code**: Examine well-written Java code to understand what quality looks like\n"
        report += "4. **Review with peers**: Discuss code reviews with others to gain different perspectives\n"
        
        return report
    
    def get_review_history(self) -> List[Dict[str, Any]]:
        """
        Get the review history.
        
        Returns:
            List of review iteration dictionaries
        """
        return self.feedback_manager.get_review_history()
    
    def reset_session(self):
        """Reset the current session."""
        self.feedback_manager.reset()