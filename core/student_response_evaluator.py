"""
Student Response Evaluator module for Java Peer Review Training System.

This module provides the StudentResponseEvaluator class which analyzes
student reviews and provides feedback on how well they identified issues.
"""

import re
import logging
import json
import random  
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StudentResponseEvaluator:
    """
    Evaluates student code reviews against known problems in the code.
    
    This class analyzes how thoroughly and accurately a student identified 
    issues in a code snippet, providing detailed feedback and metrics.
    """
    
    def __init__(self, llm: BaseLanguageModel = None,
                 min_identified_percentage: float = 60.0):
        """
        Initialize the StudentResponseEvaluator.
        
        Args:
            llm: Language model to use for evaluation
            min_identified_percentage: Minimum percentage of problems that
                                     should be identified for a sufficient review
        """
        self.llm = llm
        self.min_identified_percentage = min_identified_percentage
    
    def evaluate_review(self, 
                         code_snippet: str,
                         known_problems: List[str],
                         student_review: str) -> Dict[str, Any]:
        """
        Evaluate a student's review of code problems.
        
        Args:
            code_snippet: The original code snippet with injected errors
            known_problems: List of known problems in the code
            student_review: The student's review comments
            
        Returns:
            Dictionary with analysis results
        """
        
        return self._evaluate_with_llm(code_snippet, known_problems, student_review)
        
    def _evaluate_with_llm(self, 
                      code_snippet: str,
                      known_problems: List[str],
                      student_review: str) -> Dict[str, Any]:
        """
        Evaluate a student's review using a language model.
        
        Args:
            code_snippet: The original code snippet with injected errors
            known_problems: List of known problems in the code
            student_review: The student's review comments
            
        Returns:
            Dictionary with analysis results
        """
        if not self.llm:
            logger.warning("No LLM provided, falling back to programmatic evaluation")
            return self._fallback_evaluation(known_problems)  # Add this fallback method
            
        # Create a detailed prompt for the LLM
        system_prompt = """You are an expert code review analyzer. When analyzing student reviews:
    1. Be thorough and accurate in your assessment
    2. Return your analysis in valid JSON format with proper escaping
    3. Provide constructive feedback that helps students improve
    4. Be precise in identifying which problems were found and which were missed
    5. Format your response as proper JSON
    """
        
        prompt = f"""
    Please analyze how well the student's review identifies the known problems in the code.

    ORIGINAL CODE:
    ```java
    {code_snippet}
    ```

    KNOWN PROBLEMS IN THE CODE:
    {self._format_list(known_problems)}

    STUDENT'S REVIEW:
    ```
    {student_review}
    ```

    Carefully analyze how thoroughly and accurately the student identified the known problems.

    For each known problem, determine if the student correctly identified it, partially identified it, or missed it completely.
    Consider semantic matches - students may use different wording but correctly identify the same issue.

    Return your analysis in this exact JSON format:
    ```json
    {{
    "identified_problems": ["Problem 1 they identified correctly", "Problem 2 they identified correctly"],
    "missed_problems": ["Problem 1 they missed", "Problem 2 they missed"],
    "false_positives": ["Non-issue 1 they incorrectly flagged", "Non-issue 2 they incorrectly flagged"],
    "accuracy_percentage": 75.0,
    "review_sufficient": true,
    "feedback": "Your general assessment of the review quality and advice for improvement"
    }}
    ```

    A review is considered "sufficient" if the student correctly identified at least {self.min_identified_percentage}% of the known problems.
    Be specific in your feedback about what types of issues they missed and how they can improve their code review skills.
    """
        
        try:
            # Get the evaluation from the LLM
            logger.info("Evaluating student review with LLM")
            response = self.llm.invoke(system_prompt + "\n\n" + prompt)
            
            # Make sure we have a response
            if not response:
                logger.error("LLM returned None or empty response for review evaluation")
                return self._fallback_evaluation(known_problems)
            
            # Extract JSON data from the response
            analysis_data = self._extract_json_from_text(response)
            
            # Make sure we have analysis data
            if not analysis_data or "error" in analysis_data:
                logger.error(f"Failed to extract valid analysis data: {analysis_data.get('error', 'Unknown error')}")
                return self._fallback_evaluation(known_problems)
            
            # Process the analysis data
            return self._process_analysis_data(analysis_data, known_problems)
            
        except Exception as e:
            logger.error(f"Error evaluating review with LLM: {str(e)}")
            return self._fallback_evaluation(known_problems)

# Add this new fallback method to core/student_response_evaluator.py
    def _fallback_evaluation(self, known_problems: List[str]) -> Dict[str, Any]:
        """
        Generate a fallback evaluation when the LLM fails.
        
        Args:
            known_problems: List of known problems in the code
            
        Returns:
            Basic evaluation dictionary
        """
        logger.warning("Using fallback evaluation due to LLM error")
        
        # Create a basic fallback evaluation
        return {
            "identified_problems": [],
            "missed_problems": known_problems,
            "false_positives": [],
            "accuracy_percentage": 0.0,
            "identified_percentage": 0.0,
            "identified_count": 0,
            "total_problems": len(known_problems),
            "review_sufficient": False,
            "feedback": "Your review needs improvement. Try to identify more issues in the code."
        }
            
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON data from LLM response text.
        
        Args:
            text: Text containing JSON data
            
        Returns:
            Extracted JSON data
        """
        # Handle None or empty text
        if not text:
            return {"error": "Empty response from LLM"}
        
        try:
            # Try to find JSON block with regex
            patterns = [
                r'```json\s*([\s\S]*?)```',  # JSON code block
                r'```\s*({[\s\S]*?})\s*```',  # Any JSON object in code block
                r'({[\s\S]*"identified_problems"[\s\S]*"missed_problems"[\s\S]*})',  # Look for our expected fields
                r'({[\s\S]*})',  # Any JSON-like structure
            ]
            
            # Try each pattern
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for match in matches:
                    try:
                        # Clean up the match
                        json_str = match.strip()
                        # Try to parse as JSON
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            
            # If standard methods fail, try to manually extract fields
            logger.warning("Could not extract JSON, attempting manual extraction")
            analysis = {}
            
            # Try to extract identified problems
            identified_match = re.search(r'"identified_problems"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if identified_match:
                try:
                    identified_str = identified_match.group(1)
                    analysis["identified_problems"] = json.loads(identified_str)
                except:
                    analysis["identified_problems"] = []
            else:
                analysis["identified_problems"] = []
            
            # Try to extract missed problems
            missed_match = re.search(r'"missed_problems"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if missed_match:
                try:
                    missed_str = missed_match.group(1)
                    analysis["missed_problems"] = json.loads(missed_str)
                except:
                    analysis["missed_problems"] = []
            else:
                analysis["missed_problems"] = []
            
            # Try to extract false positives
            false_pos_match = re.search(r'"false_positives"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if false_pos_match:
                try:
                    false_pos_str = false_pos_match.group(1)
                    analysis["false_positives"] = json.loads(false_pos_str)
                except:
                    analysis["false_positives"] = []
            else:
                analysis["false_positives"] = []
            
            # Try to extract accuracy percentage
            accuracy_match = re.search(r'"accuracy_percentage"\s*:\s*([0-9.]+)', text)
            if accuracy_match:
                try:
                    analysis["accuracy_percentage"] = float(accuracy_match.group(1))
                except:
                    analysis["accuracy_percentage"] = 0.0
            else:
                analysis["accuracy_percentage"] = 0.0
            
            # Try to extract review_sufficient
            sufficient_match = re.search(r'"review_sufficient"\s*:\s*(true|false)', text)
            if sufficient_match:
                analysis["review_sufficient"] = sufficient_match.group(1) == "true"
            else:
                analysis["review_sufficient"] = False
            
            # Try to extract feedback
            feedback_match = re.search(r'"feedback"\s*:\s*"(.*?)"', text)
            if feedback_match:
                analysis["feedback"] = feedback_match.group(1)
            else:
                analysis["feedback"] = "The analysis could not extract feedback."
            
            if analysis:
                return analysis
            
            # If all else fails, return an error object
            logger.error("Could not extract analysis data from LLM response")
            return {
                "error": "Could not parse JSON response",
                "raw_text": text[:500] + ("..." if len(text) > 500 else "")
            }
            
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return {
                "error": f"Error extracting JSON: {str(e)}",
                "raw_text": text[:500] + ("..." if len(text) > 500 else "")
            }
    
    def _process_analysis_data(self, 
                          analysis_data: Dict[str, Any],
                          known_problems: List[str]) -> Dict[str, Any]:
        """
        Process and validate analysis data from the LLM.
        
        Args:
            analysis_data: Analysis data from LLM
            known_problems: List of known problems for reference
            
        Returns:
            Processed and validated analysis data
        """
        # Handle None case
        if not analysis_data:
            return self._fallback_evaluation(known_problems)
        
        # Extract required fields with fallbacks
        identified_problems = analysis_data.get("identified_problems", [])
        missed_problems = analysis_data.get("missed_problems", [])
        false_positives = analysis_data.get("false_positives", [])
        
        # Ensure these are lists, not None
        identified_problems = identified_problems if isinstance(identified_problems, list) else []
        missed_problems = missed_problems if isinstance(missed_problems, list) else []
        false_positives = false_positives if isinstance(false_positives, list) else []
        
        try:
            accuracy_percentage = float(analysis_data.get("accuracy_percentage", 50.0))
        except (TypeError, ValueError):
            accuracy_percentage = 50.0
            
        feedback = analysis_data.get("feedback", "The analysis was partially completed.")
        
        # Determine if review is sufficient based on identified percentage
        identified_count = len(identified_problems)
        total_problems = len(known_problems)
        
        if total_problems > 0:
            identified_percentage = (identified_count / total_problems) * 100
        else:
            identified_percentage = 100.0
            
        # Check if model didn't provide review_sufficient field
        if "review_sufficient" not in analysis_data:
            review_sufficient = identified_percentage >= self.min_identified_percentage
        else:
            review_sufficient = analysis_data["review_sufficient"]
        
        # Provide more detailed feedback for insufficient reviews
        if not review_sufficient and feedback == "The analysis was partially completed.":
            if identified_percentage < 30:
                feedback = ("Your review missed most of the critical issues in the code. "
                            "Try to look more carefully for logic errors, style violations, "
                            "and potential runtime exceptions.")
            else:
                feedback = ("Your review found some issues but missed important problems. "
                            f"You identified {identified_percentage:.1f}% of the known issues. "
                            "Try to be more thorough in your next review.")
        
        return {
            "identified_problems": identified_problems,
            "missed_problems": missed_problems,
            "false_positives": false_positives,
            "accuracy_percentage": accuracy_percentage,
            "identified_percentage": identified_percentage,
            "identified_count": identified_count,
            "total_problems": total_problems,
            "review_sufficient": review_sufficient,
            "feedback": feedback
        }
   
    def _format_list(self, items: List[str]) -> str:
        """Format a list of items as a bullet list."""
        return "\n".join([f"- {item}" for item in items])
    
    def generate_targeted_guidance(self,
                                  code_snippet: str,
                                  known_problems: List[str],
                                  student_review: str,
                                  review_analysis: Dict[str, Any],
                                  iteration_count: int,
                                  max_iterations: int) -> str:
        """
        Generate targeted guidance for the student to improve their review.
        
        Args:
            code_snippet: The original code snippet with injected errors
            known_problems: List of known problems in the code
            student_review: The student's review comments
            review_analysis: Analysis of the student review
            iteration_count: Current iteration number
            max_iterations: Maximum number of iterations
            
        Returns:
            Targeted guidance text
        """
        
        return self._generate_guidance_with_llm(
                code_snippet, 
                known_problems, 
                student_review, 
                review_analysis, 
                iteration_count, 
                max_iterations
            )
            
    def _generate_guidance_with_llm(self,
                                   code_snippet: str,
                                   known_problems: List[str],
                                   student_review: str,
                                   review_analysis: Dict[str, Any],
                                   iteration_count: int,
                                   max_iterations: int) -> str:
        """
        Generate targeted guidance using a language model.
        
        Args:
            code_snippet: The original code snippet with injected errors
            known_problems: List of known problems in the code
            student_review: The student's review comments
            review_analysis: Analysis of the student review
            iteration_count: Current iteration number
            max_iterations: Maximum number of iterations
            
        Returns:
            Targeted guidance text
        """
        if not self.llm:
            logger.warning("No LLM provided, falling back to programmatic guidance generation")
            
        
        # Create a detailed prompt for the LLM
        system_prompt = """You are an expert Java programming mentor who provides constructive feedback to students.
Your guidance is:
1. Encouraging and supportive
2. Specific and actionable
3. Educational - teaching students how to find issues rather than just telling them what to find
4. Focused on developing their review skills
5. Balanced - acknowledging what they did well while guiding them to improve"""
        
        prompt = f"""
Please create targeted guidance for a student who has reviewed Java code but missed some important errors.

ORIGINAL JAVA CODE:
```java
{code_snippet}
```

KNOWN PROBLEMS IN THE CODE:
{self._format_list(known_problems)}

STUDENT'S REVIEW ATTEMPT #{iteration_count} of {max_iterations}:
```
{student_review}
```

PROBLEMS CORRECTLY IDENTIFIED BY THE STUDENT:
{self._format_list(review_analysis.get("identified_problems", []))}

PROBLEMS MISSED BY THE STUDENT:
{self._format_list(review_analysis.get("missed_problems", []))}

The student has identified {review_analysis.get("identified_count", 0)} out of {review_analysis.get("total_problems", len(known_problems))} issues ({review_analysis.get("identified_percentage", 0):.1f}%).

Create constructive guidance that:
1. Acknowledges what the student found correctly with specific praise
2. Provides hints about the types of errors they missed (without directly listing them all)
3. Suggests specific areas of the code to examine more carefully
4. Encourages them to look for particular Java error patterns they may have overlooked
5. If there are false positives, gently explain why those are not actually issues
6. End with specific questions that might help the student find the missed problems

The guidance should be educational and help the student improve their Java code review skills.
Focus on teaching them how to identify the types of issues they missed.

Be encouraging but specific. Help the student develop a more comprehensive approach to code review.
"""
        
        try:
            # Generate the guidance using the LLM
            logger.info(f"Generating targeted guidance for iteration {iteration_count}")
            guidance = self.llm.invoke(system_prompt + "\n\n" + prompt)
            
            return guidance
            
        except Exception as e:
            logger.error(f"Error generating guidance with LLM: {str(e)}")
            
    
    