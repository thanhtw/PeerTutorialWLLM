"""
Student Response Evaluator module for Java Peer Review Training System.

This module provides the StudentResponseEvaluator class which analyzes
student reviews and provides feedback on how well they identified issues.
"""

import re
import logging
import json
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
        if self.llm:
            return self._evaluate_with_llm(code_snippet, known_problems, student_review)
        else:
            return self._evaluate_programmatically(code_snippet, known_problems, student_review)
    
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
            return self._evaluate_programmatically(code_snippet, known_problems, student_review)
        
        # Create a detailed prompt for the LLM
        system_prompt = """You are an expert code review analyzer. When analyzing student reviews:
1. Be thorough and accurate in your assessment
2. Return your analysis in valid JSON format with proper escaping
3. Provide constructive feedback that helps students improve
4. Be precise in identifying which problems were found and which were missed
5. Format your response as proper JSON"""
        
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
            
            # Extract JSON data from the response
            analysis_data = self._extract_json_from_text(response)
            
            # Process the analysis data
            return self._process_analysis_data(analysis_data, known_problems)
            
        except Exception as e:
            logger.error(f"Error evaluating review with LLM: {str(e)}")
            return self._evaluate_programmatically(code_snippet, known_problems, student_review)
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON data from LLM response text.
        
        Args:
            text: Text containing JSON data
            
        Returns:
            Extracted JSON data
        """
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
            
            # Try to extract missed problems
            missed_match = re.search(r'"missed_problems"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if missed_match:
                try:
                    missed_str = missed_match.group(1)
                    analysis["missed_problems"] = json.loads(missed_str)
                except:
                    analysis["missed_problems"] = []
            
            # Try to extract false positives
            false_pos_match = re.search(r'"false_positives"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if false_pos_match:
                try:
                    false_pos_str = false_pos_match.group(1)
                    analysis["false_positives"] = json.loads(false_pos_str)
                except:
                    analysis["false_positives"] = []
            
            # Try to extract accuracy percentage
            accuracy_match = re.search(r'"accuracy_percentage"\s*:\s*([0-9.]+)', text)
            if accuracy_match:
                try:
                    analysis["accuracy_percentage"] = float(accuracy_match.group(1))
                except:
                    analysis["accuracy_percentage"] = 0.0
            
            # Try to extract review_sufficient
            sufficient_match = re.search(r'"review_sufficient"\s*:\s*(true|false)', text)
            if sufficient_match:
                analysis["review_sufficient"] = sufficient_match.group(1) == "true"
            
            # Try to extract feedback
            feedback_match = re.search(r'"feedback"\s*:\s*"(.*?)"', text)
            if feedback_match:
                analysis["feedback"] = feedback_match.group(1)
            
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
        # Extract required fields with fallbacks
        identified_problems = analysis_data.get("identified_problems", [])
        missed_problems = analysis_data.get("missed_problems", [])
        false_positives = analysis_data.get("false_positives", [])
        
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
    
    def _evaluate_programmatically(self,
                                  code_snippet: str,
                                  known_problems: List[str],
                                  student_review: str) -> Dict[str, Any]:
        """
        Evaluate a student's review using programmatic analysis as a fallback.
        
        Args:
            code_snippet: The original code snippet with injected errors
            known_problems: List of known problems in the code
            student_review: The student's review comments
            
        Returns:
            Dictionary with analysis results
        """
        identified_problems = []
        missed_problems = []
        
        # Prepare the student review for analysis
        review_lower = student_review.lower()
        review_sentences = self._extract_sentences(student_review)
        
        # For each known problem, check if it was identified
        for problem in known_problems:
            problem_lower = problem.lower()
            
            # Extract key terms from the problem description
            key_terms = self._extract_key_terms(problem_lower)
            
            # Check if any sentence contains enough key terms
            found = False
            for sentence in review_sentences:
                sentence_lower = sentence.lower()
                matched_terms = [term for term in key_terms if term in sentence_lower]
                
                # If we match enough terms, consider it identified
                if len(matched_terms) >= max(2, len(key_terms) // 2):
                    identified_problems.append(problem)
                    found = True
                    break
            
            if not found:
                missed_problems.append(problem)
        
        # Check for false positives
        false_positives = []
        for sentence in review_sentences:
            # If it looks like a problem statement
            if any(indicator in sentence.lower() for indicator in ["error", "issue", "problem", "bug", "incorrect"]):
                is_known = False
                for problem in known_problems:
                    problem_lower = problem.lower()
                    key_terms = self._extract_key_terms(problem_lower)
                    matched_terms = [term for term in key_terms if term in sentence.lower()]
                    
                    if len(matched_terms) >= max(2, len(key_terms) // 2):
                        is_known = True
                        break
                
                if not is_known:
                    # Likely a false positive
                    false_positives.append(sentence)
        
        # Calculate metrics
        identified_count = len(identified_problems)
        total_problems = len(known_problems)
        
        if total_problems > 0:
            identified_percentage = (identified_count / total_problems) * 100
            accuracy_percentage = identified_percentage
        else:
            identified_percentage = 100.0
            accuracy_percentage = 100.0
        
        review_sufficient = identified_percentage >= self.min_identified_percentage
        
        # Generate feedback
        if review_sufficient:
            if identified_percentage == 100:
                feedback = "Excellent review! You identified all the problems in the code."
            else:
                feedback = (f"Good review. You identified {identified_percentage:.1f}% of the issues. "
                           f"There were {len(missed_problems)} problems you missed.")
        else:
            feedback = (f"Your review needs improvement. You only identified {identified_percentage:.1f}% "
                       f"of the issues, while {self.min_identified_percentage}% is required. "
                       f"Try to look more carefully for all types of issues.")
        
        # Add advice on missed problems if any
        if missed_problems:
            problem_categories = self._categorize_problems(missed_problems)
            feedback += " Pay more attention to " + ", ".join(problem_categories) + "."
        
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
    
    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract individual sentences from text.
        
        Args:
            text: Text to extract sentences from
            
        Returns:
            List of sentences
        """
        # Basic sentence extraction using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Also treat list items as sentences
        list_items = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|\n|$)', text)
        if list_items:
            sentences.extend(list_items)
        
        # Also treat bullet points as sentences
        bullets = re.findall(r'[-*•]\s+(.*?)(?=[-*•]|\n|$)', text)
        if bullets:
            sentences.extend(bullets)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from a problem description.
        
        Args:
            text: Problem description text
            
        Returns:
            List of key terms
        """
        # Remove common words
        stop_words = ["the", "a", "an", "is", "are", "in", "on", "at", "and", "or", "but", "to", "for", "of", "with", "that", "this"]
        
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Include specific error name patterns
        error_patterns = re.findall(r'\b[A-Z][a-zA-Z]*(?:Error|Exception|Bug|Issue)\b', text)
        
        # Add specific Java terms that may not be filtered out
        java_terms = ["null", "equals", "==", "syntax", "static", "final", "import", "class", "method", "return"]
        for term in java_terms:
            if term in text:
                key_terms.append(term)
        
        # Add error patterns
        key_terms.extend([p.lower() for p in error_patterns])
        
        # Add phrases (for capturing multi-word concepts)
        phrases = [
            "null pointer", "array index", "out of bounds", "missing return", 
            "comparison error", "naming convention", "magic number", "whitespace", 
            "braces", "javadoc", "comments", "indentation", "unused import", 
            "uncaught exception"
        ]
        for phrase in phrases:
            if phrase in text:
                key_terms.append(phrase)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(key_terms))
    
    def _categorize_problems(self, problems: List[str]) -> List[str]:
        """
        Categorize problems into general categories.
        
        Args:
            problems: List of problem descriptions
            
        Returns:
            List of problem categories
        """
        categories = set()
        
        for problem in problems:
            problem_lower = problem.lower()
            
            if any(term in problem_lower for term in ["nullpointer", "null pointer", "null reference"]):
                categories.add("null pointer issues")
            elif any(term in problem_lower for term in ["array", "index", "bounds"]):
                categories.add("array indexing issues")
            elif any(term in problem_lower for term in ["naming", "variable name", "method name", "class name"]):
                categories.add("naming convention issues")
            elif any(term in problem_lower for term in ["whitespace", "indentation", "formatting"]):
                categories.add("formatting issues")
            elif any(term in problem_lower for term in ["javadoc", "comment", "documentation"]):
                categories.add("documentation issues")
            elif any(term in problem_lower for term in ["import", "package"]):
                categories.add("import-related issues")
            elif any(term in problem_lower for term in ["exception", "try", "catch", "throw"]):
                categories.add("exception handling")
            elif any(term in problem_lower for term in ["logic", "condition", "loop", "comparison"]):
                categories.add("logical errors")
            elif any(term in problem_lower for term in ["type", "cast", "conversion"]):
                categories.add("type-related issues")
            else:
                categories.add("code quality issues")
        
        return list(categories)
    
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
        if self.llm:
            return self._generate_guidance_with_llm(
                code_snippet, 
                known_problems, 
                student_review, 
                review_analysis, 
                iteration_count, 
                max_iterations
            )
        else:
            return self._generate_guidance_programmatically(
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
            return self._generate_guidance_programmatically(
                code_snippet, known_problems, student_review, review_analysis, iteration_count, max_iterations
            )
        
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
            return self._generate_guidance_programmatically(
                code_snippet, known_problems, student_review, review_analysis, iteration_count, max_iterations
            )
    
    def _generate_guidance_programmatically(self,
                                          code_snippet: str,
                                          known_problems: List[str],
                                          student_review: str,
                                          review_analysis: Dict[str, Any],
                                          iteration_count: int,
                                          max_iterations: int) -> str:
        """
        Generate targeted guidance programmatically as a fallback.
        
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
        identified_problems = review_analysis.get("identified_problems", [])
        missed_problems = review_analysis.get("missed_problems", [])
        identified_count = review_analysis.get("identified_count", 0)
        total_problems = review_analysis.get("total_problems", len(known_problems))
        identified_percentage = review_analysis.get("identified_percentage", 0)
        
        # Start with acknowledgment of what they found
        if identified_problems:
            guidance = f"# Review Feedback - Attempt {iteration_count} of {max_iterations}\n\n"
            guidance += "## What You Found Correctly\n\n"
            guidance += f"Good work identifying {identified_count} out of {total_problems} issues! "
            
            if identified_count == 1:
                guidance += f"You correctly spotted: {identified_problems[0]}.\n\n"
            else:
                guidance += "You correctly spotted these issues:\n"
                for problem in identified_problems[:3]:  # Limit to first 3 to avoid overwhelming
                    guidance += f"- {problem}\n"
                if len(identified_problems) > 3:
                    guidance += f"- And {len(identified_problems) - 3} more issue(s)\n"
                guidance += "\n"
        else:
            guidance = f"# Review Feedback - Attempt {iteration_count} of {max_iterations}\n\n"
            guidance += "You haven't identified any of the issues in the code yet. That's okay! Let's work on finding them.\n\n"
        
        # Provide hints about missed issues
        guidance += "## Areas to Focus On\n\n"
        
        # Categorize missed problems
        problem_categories = self._categorize_problems(missed_problems)
        
        guidance += "You should focus on these types of issues:\n\n"
        for category in problem_categories:
            guidance += f"- {category}\n"
        
        guidance += "\n"
        
        # Suggest specific code areas to examine
        guidance += "## Code Areas to Examine\n\n"
        
        # Find methods/lines that likely contain issues
        code_lines = code_snippet.split("\n")
        areas_to_check = []
        
        # Look for methods
        method_starts = []
        for i, line in enumerate(code_lines):
            if ("public" in line or "private" in line) and "(" in line and ")" in line and "{" in line:
                method_name = "unknown method"
                match = re.search(r'\b\w+\s+(\w+)\s*\(', line)
                if match:
                    method_name = match.group(1)
                method_starts.append((i, method_name))
        
        # Check if missed problems have keywords that might match methods
        for method_line, method_name in method_starts:
            for problem in missed_problems:
                problem_lower = problem.lower()
                if method_name.lower() in problem_lower:
                    areas_to_check.append(f"The {method_name} method (around line {method_line+1})")
                    break
        
        # Add general areas if we don't have specific matches
        if not areas_to_check:
            # Look for specific code constructs
            for problem_type, keywords in [
                ("class definition", ["class", "extends", "implements"]),
                ("field declarations", ["private", "protected", "public", ";", "="]),
                ("method signatures", ["public", "private", "void", "return", "("]),
                ("import statements", ["import"]),
                ("comments and documentation", ["/**", "/*", "*/"]),
                ("conditional statements", ["if", "else", "switch"]),
                ("loops", ["for", "while", "do"]),
                ("exception handling", ["try", "catch", "throw", "throws"]),
            ]:
                for i, line in enumerate(code_lines):
                    if any(keyword in line for keyword in keywords):
                        areas_to_check.append(f"The {problem_type} (around line {i+1})")
                        break
        
        # Add some areas to check
        if areas_to_check:
            guidance += "Take a closer look at:\n\n"
            for area in areas_to_check[:3]:  # Limit to 3 areas
                guidance += f"- {area}\n"
        else:
            guidance += "Make sure to carefully examine:\n\n"
            guidance += "- The class structure and declarations\n"
            guidance += "- Method implementations and logic\n"
            guidance += "- Variable declarations and usage\n"
        
        guidance += "\n"
        
        # Provide specific tips based on categories
        guidance += "## Review Tips\n\n"
        
        tips = []
        for category in problem_categories:
            if "null pointer" in category:
                tips.append("Check for variables that might be null before they're used")
            elif "array indexing" in category:
                tips.append("Look for array or list accesses that might go beyond their bounds")
            elif "naming convention" in category:
                tips.append("Verify that class, method, and variable names follow Java conventions")
            elif "formatting" in category:
                tips.append("Review indentation, spacing around operators, and brace placement")
            elif "documentation" in category:
                tips.append("Check if methods and classes have appropriate Javadoc comments")
            elif "import" in category:
                tips.append("Look for unused, redundant, or missing imports")
            elif "exception" in category:
                tips.append("Verify that exceptions are properly caught or declared in method signatures")
            elif "logical" in category:
                tips.append("Check loop conditions, if statements, and comparison operators")
            elif "type" in category:
                tips.append("Look for incompatible types in assignments and method parameters")
            else:
                tips.append("Review the code for common Java programming mistakes and style issues")
        
        # Add some general tips as well
        general_tips = [
            "Review Java's equality comparison: '==' for primitives, '.equals()' for objects",
            "Check for proper resource management (closing files, streams, etc.)",
            "Ensure proper access modifiers are used (public, private, protected)",
            "Verify that appropriate data structures are being used"
        ]
        
        # Select a couple of general tips
        selected_general_tips = random.sample(general_tips, min(2, len(general_tips)))
        
        # Combine specific and general tips
        all_tips = tips + selected_general_tips
        random.shuffle(all_tips)  # Mix them up
        
        for tip in all_tips[:5]:  # Limit to 5 tips
            guidance += f"- {tip}\n"
        
        guidance += "\n"
        
        # End with reflective questions
        guidance += "## Questions to Consider\n\n"
        
        questions = [
            "Are there any places where variables are used before being initialized?",
            "Do all methods return the appropriate types?",
            "Are there any potential null pointer exceptions?",
            "Are naming conventions consistent throughout the code?",
            "Is the code properly documented with Javadoc comments?",
            "Are there any logical errors in loop conditions or if statements?",
            "Could any exceptions be thrown but not caught?",
            "Are there any performance issues in the code?",
            "Do any methods or classes violate design principles?",
            "Are magic numbers used instead of named constants?"
        ]
        
        # Select questions based on missed problem categories
        selected_questions = []
        for category in problem_categories:
            if "null pointer" in category and "Are there any potential null pointer exceptions?" in questions:
                selected_questions.append("Are there any potential null pointer exceptions?")
            elif "array indexing" in category and "Are there any places where array indices might be out of bounds?" not in selected_questions:
                selected_questions.append("Are there any places where array indices might be out of bounds?")
            elif "naming convention" in category and "Are naming conventions consistent throughout the code?" in questions:
                selected_questions.append("Are naming conventions consistent throughout the code?")
            elif "documentation" in category and "Is the code properly documented with Javadoc comments?" in questions:
                selected_questions.append("Is the code properly documented with Javadoc comments?")
            elif "exception" in category and "Could any exceptions be thrown but not caught?" in questions:
                selected_questions.append("Could any exceptions be thrown but not caught?")
            elif "logical" in category and "Are there any logical errors in loop conditions or if statements?" in questions:
                selected_questions.append("Are there any logical errors in loop conditions or if statements?")
        
        # If we don't have enough specific questions, add some general ones
        remaining_questions = [q for q in questions if q not in selected_questions]
        while len(selected_questions) < 3 and remaining_questions:
            selected_questions.append(remaining_questions.pop(0))
        
        for question in selected_questions:
            guidance += f"- {question}\n"
        
        return guidance