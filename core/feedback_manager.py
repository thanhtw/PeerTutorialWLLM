"""
Feedback Manager module for Java Peer Review Training System.

This module provides the FeedbackManager class which coordinates
the feedback loop between student reviews and AI-generated feedback.
"""

import logging
import datetime
from typing import List, Dict, Any, Optional, Tuple
from .student_response_evaluator import StudentResponseEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewIteration:
    """Class for storing a single review iteration."""
    
    def __init__(self, iteration_number: int, student_review: str, 
                 review_analysis: Dict[str, Any], targeted_guidance: Optional[str] = None, 
                 timestamp: Optional[str] = None):
        """
        Initialize a review iteration record.
        
        Args:
            iteration_number: The sequence number of this review iteration
            student_review: The student's review comments
            review_analysis: Analysis of the student review
            targeted_guidance: Guidance provided for the next iteration
            timestamp: Timestamp of this review iteration
        """
        self.iteration_number = iteration_number
        self.student_review = student_review
        self.review_analysis = review_analysis
        self.targeted_guidance = targeted_guidance
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration_number": self.iteration_number,
            "student_review": self.student_review,
            "review_analysis": self.review_analysis,
            "targeted_guidance": self.targeted_guidance,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewIteration':
        """Create a ReviewIteration from a dictionary."""
        return cls(
            iteration_number=data.get("iteration_number", 1),
            student_review=data.get("student_review", ""),
            review_analysis=data.get("review_analysis", {}),
            targeted_guidance=data.get("targeted_guidance"),
            timestamp=data.get("timestamp")
        )

class FeedbackManager:
    """
    Manages the feedback loop between student reviews and AI-generated feedback.
    
    This class coordinates the iterative process of student review, analysis,
    and feedback generation, maintaining the history of review iterations.
    """
    
    def __init__(self, evaluator: StudentResponseEvaluator,
                max_iterations: int = 3):
        """
        Initialize the FeedbackManager.
        
        Args:
            evaluator: StudentResponseEvaluator for analyzing reviews
            max_iterations: Maximum number of review iterations
        """
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.review_history = []
        self.current_iteration = 1
        self.review_sufficient = False
        self.code_snippet = ""
        self.known_problems = []
    
    def start_new_review_session(self, code_snippet: str, known_problems: List[str]):
        """
        Start a new review session.
        
        Args:
            code_snippet: The code snippet to be reviewed
            known_problems: List of known problems in the code
        """
        self.code_snippet = code_snippet
        self.known_problems = known_problems
        self.review_history = []
        self.current_iteration = 1
        self.review_sufficient = False
        logger.info("Started new review session")
    
    def submit_review(self, student_review: str) -> Dict[str, Any]:
        """
        Submit a student review for analysis.
        
        Args:
            student_review: The student's review comments
            
        Returns:
            Dictionary with analysis results and guidance
        """
        if not self.code_snippet or not self.known_problems:
            logger.error("Cannot submit review without active code snippet")
            return {
                "error": "No active code snippet to review"
            }
        
        # Analyze the student review
        logger.info(f"Analyzing student review (iteration {self.current_iteration})")
        review_analysis = self.evaluator.evaluate_review(
            code_snippet=self.code_snippet,
            known_problems=self.known_problems,
            student_review=student_review
        )
        
        # Check if the review is sufficient
        self.review_sufficient = review_analysis.get("review_sufficient", False)
        
        # Generate targeted guidance if needed
        targeted_guidance = None
        if not self.review_sufficient and self.current_iteration < self.max_iterations:
            logger.info(f"Generating guidance for iteration {self.current_iteration}")
            targeted_guidance = self.evaluator.generate_targeted_guidance(
                code_snippet=self.code_snippet,
                known_problems=self.known_problems,
                student_review=student_review,
                review_analysis=review_analysis,
                iteration_count=self.current_iteration,
                max_iterations=self.max_iterations
            )
        
        # Create a review iteration record
        current_iteration = ReviewIteration(
            iteration_number=self.current_iteration,
            student_review=student_review,
            review_analysis=review_analysis,
            targeted_guidance=targeted_guidance,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        # Add to review history
        self.review_history.append(current_iteration)
        
        # Prepare the result
        result = {
            "iteration_count": self.current_iteration,
            "review_analysis": review_analysis,
            "review_sufficient": self.review_sufficient,
            "targeted_guidance": targeted_guidance,
            "next_steps": "summarize" if self.review_sufficient or self.current_iteration >= self.max_iterations else "iterate"
        }
        
        # Increment iteration count for next review
        self.current_iteration += 1
        
        return result
    
    def get_review_history(self) -> List[Dict[str, Any]]:
        """
        Get the review history.
        
        Returns:
            List of review iteration dictionaries
        """
        return [iteration.to_dict() for iteration in self.review_history]
    
    def get_latest_review(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest review iteration.
        
        Returns:
            Dictionary of the latest review iteration or None if no reviews
        """
        if not self.review_history:
            return None
        
        return self.review_history[-1].to_dict()
    
    def generate_final_feedback(self) -> str:
        """
        Generate final feedback summary after the review process.
        
        Returns:
            Final feedback text
        """
        if not self.review_history:
            return "No reviews submitted yet."
        
        # Get the final/best review
        latest_review = self.review_history[-1]
        
        # Build a feedback summary
        feedback = "# Final Review Feedback\n\n"
        
        # Analysis stats
        analysis = latest_review.review_analysis
        identified_count = analysis.get("identified_count", 0)
        total_problems = analysis.get("total_problems", len(self.known_problems))
        identified_percentage = analysis.get("identified_percentage", 0)
        
        # Performance summary
        feedback += "## Review Performance\n\n"
        feedback += f"You identified {identified_count} out of {total_problems} issues "
        feedback += f"({identified_percentage:.1f}% accuracy).\n\n"
        
        # Strengths and areas for improvement
        feedback += "## Strengths\n\n"
        
        identified_problems = analysis.get("identified_problems", [])
        if identified_problems:
            feedback += "You correctly identified:\n\n"
            for problem in identified_problems:
                feedback += f"- {problem}\n"
        else:
            feedback += "Keep practicing to improve your error identification skills.\n"
        
        feedback += "\n## Areas for Improvement\n\n"
        
        missed_problems = analysis.get("missed_problems", [])
        if missed_problems:
            feedback += "You missed these important issues:\n\n"
            for problem in missed_problems:
                feedback += f"- {problem}\n"
        else:
            feedback += "Great job! You found all the issues.\n"
        
        # Progress across iterations
        if len(self.review_history) > 1:
            feedback += "\n## Progress Across Iterations\n\n"
            
            # Create a progress table
            feedback += "| Iteration | Issues Found | Accuracy |\n"
            feedback += "|-----------|--------------|----------|\n"
            
            for iteration in self.review_history:
                iter_num = iteration.iteration_number
                iter_analysis = iteration.review_analysis
                iter_found = iter_analysis.get("identified_count", 0)
                iter_accuracy = iter_analysis.get("identified_percentage", 0)
                
                feedback += f"| {iter_num} | {iter_found}/{total_problems} | {iter_accuracy:.1f}% |\n"
        
        # Tips for future reviews
        feedback += "\n## Tips for Future Code Reviews\n\n"
        
        # Derive tips from missed problems
        if missed_problems:
            problem_categories = set()
            
            # Simple categorization of missed problems
            for problem in missed_problems:
                problem_lower = problem.lower()
                
                if any(term in problem_lower for term in ["null", "nullpointer", "null pointer"]):
                    problem_categories.add("null pointer handling")
                elif any(term in problem_lower for term in ["array", "index", "bound"]):
                    problem_categories.add("array bounds checking")
                elif any(term in problem_lower for term in ["exception", "throw", "catch"]):
                    problem_categories.add("exception handling")
                elif any(term in problem_lower for term in ["comparison", "equals", "=="]):
                    problem_categories.add("object comparison")
                elif any(term in problem_lower for term in ["name", "convention", "style"]):
                    problem_categories.add("naming conventions")
                elif any(term in problem_lower for term in ["whitespace", "indent", "format"]):
                    problem_categories.add("code formatting")
                elif any(term in problem_lower for term in ["comment", "javadoc", "documentation"]):
                    problem_categories.add("code documentation")
                elif any(term in problem_lower for term in ["return", "type", "cast"]):
                    problem_categories.add("type safety")
                else:
                    problem_categories.add("general code quality")
            
            # Generate tips based on categories
            if problem_categories:
                feedback += "Based on your review, focus on these areas:\n\n"
                
                for category in problem_categories:
                    if category == "null pointer handling":
                        feedback += "- **Null Pointer Handling**: Always check if objects can be null before accessing their methods or properties.\n"
                    elif category == "array bounds checking":
                        feedback += "- **Array Bounds Checking**: Verify that array indices are within valid ranges before accessing elements.\n"
                    elif category == "exception handling":
                        feedback += "- **Exception Handling**: Look for code that might throw exceptions but doesn't handle them properly.\n"
                    elif category == "object comparison":
                        feedback += "- **Object Comparison**: Remember that `==` compares references while `equals()` compares content for objects like Strings.\n"
                    elif category == "naming conventions":
                        feedback += "- **Naming Conventions**: Verify that classes, methods, and variables follow Java naming conventions.\n"
                    elif category == "code formatting":
                        feedback += "- **Code Formatting**: Check for proper indentation, whitespace, and consistency in formatting.\n"
                    elif category == "code documentation":
                        feedback += "- **Documentation**: Ensure methods have proper Javadoc comments and parameters are documented.\n"
                    elif category == "type safety":
                        feedback += "- **Type Safety**: Look for improper type conversions or missing return statements.\n"
                    else:
                        feedback += "- **Code Quality**: Review for general coding best practices and potential bugs.\n"
        else:
            # Generic tips if no missed problems
            feedback += "- **Completeness**: Always check the entire codebase systematically.\n"
            feedback += "- **Methodology**: Develop a structured approach to code reviews.\n"
            feedback += "- **Documentation**: Look for clear and complete documentation.\n"
            feedback += "- **Best Practices**: Compare code against Java best practices.\n"
        
        return feedback
    
    def reset(self):
        """Reset the feedback manager to initial state."""
        self.review_history = []
        self.current_iteration = 1
        self.review_sufficient = False
        self.code_snippet = ""
        self.known_problems = []