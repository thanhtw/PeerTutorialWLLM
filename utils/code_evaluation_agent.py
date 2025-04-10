"""
Code Evaluation Agent for Java Peer Review Training System.

This module provides the CodeEvaluationAgent class which evaluates 
generated Java code to ensure it contains the required errors.
Uses an LLM for more accurate analysis when available.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel

from utils.error_validation import validate_code_errors, is_comment, is_primitive_or_common
from utils.export_utils import export_prompt_response  # Import the new utility

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeEvaluationAgent:
    """
    Agent for evaluating generated Java code to ensure it meets error requirements.
    
    This agent provides detailed feedback on how well the generated code
    implements the required errors, and suggests improvements for the
    code generator. Can use an LLM for more accurate evaluation.
    """
    
    def __init__(self, llm: BaseLanguageModel = None, export_debug: bool = True):
        """
        Initialize the CodeEvaluationAgent with optional LLM.
        
        Args:
            llm: Optional language model for evaluation
            export_debug: Whether to export prompts and responses to files
        """
        logger.info("Initializing Code Evaluation Agent")
        self.llm = llm
        self.export_debug = export_debug
        if llm:
            logger.info("LLM provided for code evaluation")
        else:
            logger.info("No LLM provided, will use regex-based validation")
    
    def evaluate_code(self, code: str, requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate generated code against requested errors with detailed feedback.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            
        Returns:
            Dictionary with evaluation results and feedback
        """
        try:
            logger.info(f"Evaluating code with {len(requested_errors)} requested errors")
            
            # Use LLM-based evaluation if available, otherwise fall back to regex-based validation
            if self.llm:
                validation_results = self._evaluate_with_llm(code, requested_errors)
                logger.info("Using LLM for code evaluation")
            else:
                validation_results = validate_code_errors(code, requested_errors)
                logger.info("Using regex-based validation (LLM not available)")
            
            # Create more detailed evaluation
            evaluation = {
                "valid": validation_results["valid"],
                "found_errors": validation_results["found_errors"],
                "missing_errors": validation_results["missing_errors"],
                "error_locations": validation_results["error_locations"],
                "feedback": self._generate_feedback(code, requested_errors, validation_results),
                "suggestions": self._generate_suggestions(code, requested_errors, validation_results)
            }
            
            # Include LLM feedback if available
            if "llm_feedback" in validation_results:
                evaluation["llm_feedback"] = validation_results["llm_feedback"]
            if "detailed_analysis" in validation_results:
                evaluation["detailed_analysis"] = validation_results["detailed_analysis"]
            
            logger.info(f"Evaluation complete: found {len(evaluation['found_errors'])} errors, missing {len(evaluation['missing_errors'])} errors")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating code: {str(e)}")
            # Return a safe fallback
            return {
                "valid": False,
                "found_errors": [],
                "missing_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                                for error in requested_errors],
                "error_locations": {},
                "feedback": f"Error during evaluation: {str(e)}",
                "suggestions": []
            }
    
    def _evaluate_with_llm(self, code: str, requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate generated code using an LLM to identify if requested errors are present.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.llm:
            logger.warning("No LLM provided for evaluation, falling back to regex-based validation")
            return validate_code_errors(code, requested_errors)
        
        # Format the requested errors for the prompt
        error_descriptions = []
        for i, error in enumerate(requested_errors, 1):
            error_type = error.get("type", "").upper()
            name = error.get("name", "")
            description = error.get("description", "")
            implementation_guide = error.get("implementation_guide", "")
            
            error_descriptions.append(f"{i}. {error_type} ERROR - {name}\n   Description: {description}\n   Implementation guide: {implementation_guide}")
        
        error_list = "\n\n".join(error_descriptions)
        
        # Create a prompt for the LLM
        prompt = f"""You are an expert Java code reviewer tasked with evaluating whether a code snippet correctly implements specific errors.

CODE SNIPPET:
```java
{code}
```

REQUESTED ERRORS TO IMPLEMENT:
{error_list}

Analyze the code carefully and determine if each requested error is properly implemented.
For each error, provide:
1. Whether it is implemented in the code (YES/NO)
2. The exact line number where the error occurs
3. The specific code segment that contains the error
4. A brief explanation of how the error is implemented or why it's missing

Return your analysis in the following JSON format:
```json
{{
  "found_errors": [
    {{
      "error_type": "BUILD",
      "error_name": "NullPointerException",
      "line_number": 42,
      "code_segment": "String str = null; int length = str.length();",
      "explanation": "This will throw NullPointerException because str is null when length() is called"
    }}
  ],
  "missing_errors": [
    {{
      "error_type": "CHECKSTYLE",
      "error_name": "MemberName",
      "explanation": "No variable names breaking member naming conventions were found in the code"
    }}
  ],
  "valid": false,
  "feedback": "The code successfully implements 2 out of 3 requested errors. The NullPointerException and StringComparison errors are correctly implemented, but the MemberName error is missing."
}}
```

Be thorough and precise in your analysis, ensuring that you identify exactly where and how each error is implemented.
"""
        
        try:
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Export the prompt and response if export_debug is enabled
            if self.export_debug:
                export_prompt_response(
                    prompt=prompt, 
                    response=str(response), 
                    operation_type="code_evaluation",
                    error_list=requested_errors
                )
            
            # Extract JSON from response
            import re
            import json
            
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
                try:
                    analysis = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from LLM response")
                    return validate_code_errors(code, requested_errors)
            else:
                # Try to find any JSON object in the response
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        analysis = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from LLM response")
                        return validate_code_errors(code, requested_errors)
                else:
                    logger.error("No JSON found in LLM response")
                    return validate_code_errors(code, requested_errors)
            
            # Process the analysis results
            found_errors = []
            missing_errors = []
            error_locations = {}
            
            # Extract found errors
            for error in analysis.get("found_errors", []):
                error_type = error.get("error_type", "")
                error_name = error.get("error_name", "")
                error_key = f"{error_type} - {error_name}"
                
                found_errors.append(error_key)
                error_locations[error_key] = error.get("line_number", 0)
            
            # Extract missing errors
            for error in analysis.get("missing_errors", []):
                error_type = error.get("error_type", "")
                error_name = error.get("error_name", "")
                error_key = f"{error_type} - {error_name}"
                
                missing_errors.append(error_key)
            
            # Check if we're missing any errors that weren't explicitly mentioned
            all_requested_keys = [f"{error.get('type', '').upper()} - {error.get('name', '')}" for error in requested_errors]
            for key in all_requested_keys:
                if key not in found_errors and key not in missing_errors:
                    missing_errors.append(key)
            
            # Create the validation result
            validation_results = {
                "valid": analysis.get("valid", False),
                "found_errors": found_errors,
                "missing_errors": missing_errors,
                "error_locations": error_locations,
                "llm_feedback": analysis.get("feedback", ""),
                "detailed_analysis": analysis  # Keep the full LLM analysis for detailed feedback
            }
            
            # Export the evaluation results if export_debug is enabled
            if self.export_debug:
                export_prompt_response(
                    prompt="", 
                    response="", 
                    operation_type="evaluation_results",
                    error_list=requested_errors,
                    evaluation_result=validation_results
                )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error evaluating code with LLM: {str(e)}")
            # Fall back to regex-based validation
            return validate_code_errors(code, requested_errors)
    
    def _generate_feedback(self, code: str, requested_errors: List[Dict[str, Any]], 
                         validation_results: Dict[str, Any]) -> str:
        """
        Generate detailed feedback on the implementation of errors.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            validation_results: Results from validation
            
        Returns:
            Detailed feedback string
        """
        # If LLM feedback is available, use it
        if "llm_feedback" in validation_results and validation_results["llm_feedback"]:
            return validation_results["llm_feedback"]
        
        # Otherwise, use the original feedback generation logic
        lines = code.splitlines()
        feedback = []
        
        # Provide feedback on correctly implemented errors
        if validation_results["found_errors"]:
            feedback.append("Successfully implemented errors:")
            for error_key in validation_results["found_errors"]:
                line_num = validation_results["error_locations"].get(error_key, 0)
                line_content = lines[line_num-1] if 0 < line_num <= len(lines) else "Unknown"
                feedback.append(f"- {error_key} (Line {line_num}: '{line_content.strip()}')")
        
        # Provide feedback on missing errors
        if validation_results["missing_errors"]:
            feedback.append("\nErrors that need implementation:")
            for error_key in validation_results["missing_errors"]:
                # Find the corresponding error details
                error_details = None
                for error in requested_errors:
                    if f"{error.get('type', '').upper()} - {error.get('name', '')}" == error_key:
                        error_details = error
                        break
                
                if error_details:
                    implementation_guide = error_details.get("implementation_guide", "No implementation guide available")
                    feedback.append(f"- {error_key}")
                    feedback.append(f"  Implementation guide: {implementation_guide}")
                else:
                    feedback.append(f"- {error_key} (Details not available)")
        
        # Overall assessment
        if validation_results["valid"]:
            feedback.append("\nAll requested errors have been successfully implemented in the code.")
        else:
            found_count = len(validation_results["found_errors"])
            total_count = len(requested_errors)
            feedback.append(f"\nImplemented {found_count} out of {total_count} requested errors "
                          f"({found_count/total_count*100:.1f}%).")
            
        return "\n".join(feedback)
    
    def _generate_suggestions(self, code: str, requested_errors: List[Dict[str, Any]], 
                            validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate specific suggestions for implementing missing errors.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            validation_results: Results from validation
            
        Returns:
            List of suggestion dictionaries
        """
        suggestions = []
        
        # If we have detailed analysis from LLM, use it to generate better suggestions
        if "detailed_analysis" in validation_results:
            detailed_analysis = validation_results["detailed_analysis"]
            missing_errors_analysis = detailed_analysis.get("missing_errors", [])
            
            for error_analysis in missing_errors_analysis:
                error_type = error_analysis.get("error_type", "")
                error_name = error_analysis.get("error_name", "")
                explanation = error_analysis.get("explanation", "")
                
                error_key = f"{error_type} - {error_name}"
                
                suggestion = {
                    "error_key": error_key,
                    "suggestions": [explanation]
                }
                
                # Try to find the corresponding error details for implementation guide
                for error in requested_errors:
                    if f"{error.get('type', '').upper()} - {error.get('name', '')}" == error_key:
                        implementation_guide = error.get("implementation_guide", "")
                        if implementation_guide:
                            suggestion["suggestions"].append(f"Implementation guide: {implementation_guide}")
                        break
                
                suggestions.append(suggestion)
        else:
            # Fall back to original suggestion generation
            for error_key in validation_results["missing_errors"]:
                # Find the corresponding error details
                error_details = None
                for error in requested_errors:
                    if f"{error.get('type', '').upper()} - {error.get('name', '')}" == error_key:
                        error_details = error
                        break
                
                if not error_details:
                    continue
                    
                error_type = error_details.get("type", "").lower()
                error_name = error_details.get("name", "")
                
                suggestion = {
                    "error_key": error_key,
                    "suggestions": []
                }
                
                # Get implementation guide if available
                implementation_guide = error_details.get("implementation_guide", "")
                if implementation_guide:
                    suggestion["suggestions"].append(f"Follow implementation guide: {implementation_guide}")
                
                # Add generic suggestion based on error name
                suggestion["suggestions"].append(
                    f"Look for ways to introduce a {error_name} error in the code"
                )
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def generate_improved_prompt(self, code: str, requested_errors: List[Dict[str, Any]], 
                              evaluation: Dict[str, Any]) -> str:
        """
        Generate an improved prompt for the code generator based on LLM evaluation.
        
        Args:
            code: The previously generated code
            requested_errors: List of errors that should be implemented
            evaluation: Evaluation results from evaluate_code method
            
        Returns:
            Improved prompt string for the code generator
        """
        # Start with the base prompt
        from utils.code_utils import create_code_generation_prompt
        
        # Determine domain from existing code
        domain = self._infer_domain_from_code(code)
        
        # Create base prompt
        prompt = create_code_generation_prompt(
            code_length="medium",  # Will be overridden with specifics
            difficulty_level="medium",  # Will be overridden with specifics
            selected_errors=requested_errors,
            domain=domain,
            include_error_annotations=True
        )
        
        # Add specific guidance based on evaluation
        prompt += "\n\n## FEEDBACK ON PREVIOUS CODE GENERATION ATTEMPT\n"
        
        # If there's LLM feedback, use it
        if "llm_feedback" in evaluation:
            prompt += f"\n{evaluation['llm_feedback']}\n"
        
        # If there's detailed analysis, use it for more targeted feedback
        if "detailed_analysis" in evaluation:
            detailed_analysis = evaluation["detailed_analysis"]
            
            # Add information about found errors
            found_errors = detailed_analysis.get("found_errors", [])
            if found_errors:
                prompt += "\n### What was implemented correctly:\n"
                for error in found_errors:
                    error_type = error.get("error_type", "")
                    error_name = error.get("error_name", "")
                    line_number = error.get("line_number", "Unknown")
                    code_segment = error.get("code_segment", "")
                    explanation = error.get("explanation", "")
                    
                    prompt += f"-  {error_type} - {error_name}\n"
                    prompt += f"  Found at line {line_number}: `{code_segment}`\n"
                    prompt += f"  {explanation}\n"
            
            # Add information about missing errors
            missing_errors = detailed_analysis.get("missing_errors", [])
            if missing_errors:
                prompt += "\n### Errors that need to be implemented:\n"
                
                for error in missing_errors:
                    error_type = error.get("error_type", "")
                    error_name = error.get("error_name", "")
                    explanation = error.get("explanation", "")
                    
                    prompt += f"- {error_type} - {error_name}\n"
                    prompt += f"  {explanation}\n"
                    
                    # Find the corresponding error details
                    for req_error in requested_errors:
                        if (req_error.get("type", "").upper() == error_type and 
                            req_error.get("name", "") == error_name):
                            description = req_error.get("description", "")
                            implementation_guide = req_error.get("implementation_guide", "")
                            
                            prompt += f"  Description: {description}\n"
                            prompt += f"  Implementation guide: {implementation_guide}\n\n"
        else:
            # Fall back to original approach
            if evaluation["found_errors"]:
                prompt += "\n### What was implemented correctly:\n"
                for error_key in evaluation["found_errors"]:
                    prompt += f"-  {error_key}\n"
                        
                    # Include the exact line where the error was found, if available
                    line_num = evaluation["error_locations"].get(error_key, 0)
                    if line_num > 0:
                        lines = code.splitlines()
                        line_content = lines[line_num-1] if 0 < line_num <= len(lines) else "Unknown"
                        prompt += f"  Found at line {line_num}: `{line_content.strip()}`\n"
            
            if evaluation["missing_errors"]:
                prompt += "\n### Errors that need to be implemented:\n"
                
                for error_key in evaluation["missing_errors"]:
                    prompt += f"- {error_key}\n"
                    
                    # Find the corresponding error details
                    for error in requested_errors:
                        if f"{error.get('type', '').upper()} - {error.get('name', '')}" == error_key:
                            error_type = error.get("type", "").upper()
                            name = error.get("name", "")
                            description = error.get("description", "")
                            implementation_guide = error.get("implementation_guide", "")
                            
                            prompt += f"  Description: {description}\n"
                            prompt += f"  Implementation guide: {implementation_guide}\n\n"
        
        prompt += "\n## SPECIFIC INSTRUCTIONS FOR THIS ATTEMPT\n"
        prompt += "\n1. Please revise the code to implement ALL requested errors. Be sure to follow the implementation guides."
        prompt += "\n2. Make sure to add error annotations in the standard format: `// ERROR: [TYPE] - [NAME] - [Brief explanation]`"
        prompt += "\n3. Keep your correct implementations from the previous attempt while adding the missing errors."
        prompt += "\n4. Focus especially on implementing the missing errors highlighted above."
        prompt += "\n5. Return your final code in a code block with ``` delimiters."
        
        prompt += "\n\nHere's the previous code to improve upon:\n\n```java\n"
        prompt += code
        prompt += "\n```"
        
        # Export the improved prompt if export_debug is enabled
        if self.export_debug:
            export_prompt_response(
                prompt=prompt, 
                response="", 
                operation_type="improved_generation_prompt",
                error_list=requested_errors,
                evaluation_result=evaluation
            )
        
        return prompt
    
    def _find_method_bodies(self, code: str) -> List[Tuple[int, int]]:
        """
        Find method bodies in the code for suggesting insertion points.
        
        Args:
            code: The Java code to analyze
            
        Returns:
            List of (start_line, end_line) tuples for method bodies
        """
        lines = code.splitlines()
        method_bodies = []
        current_method_start = None
        brace_count = 0
        
        for i, line in enumerate(lines):
            # Skip comments
            if is_comment(line):
                continue
                
            # Look for method declarations
            if (("public" in line or "private" in line or "protected" in line) and 
                "(" in line and ")" in line and "{" in line and ";" not in line and
                current_method_start is None):
                current_method_start = i
                brace_count = line.count("{") - line.count("}")
            
            # Count braces to find method end
            elif current_method_start is not None:
                brace_count += line.count("{") - line.count("}")
                
                if brace_count == 0:
                    method_bodies.append((current_method_start, i))
                    current_method_start = None
        
        return method_bodies
    
    def _find_class_bodies(self, code: str) -> List[Tuple[int, int]]:
        """
        Find class bodies in the code for suggesting insertion points.
        
        Args:
            code: The Java code to analyze
            
        Returns:
            List of (start_line, end_line) tuples for class bodies
        """
        lines = code.splitlines()
        class_bodies = []
        current_class_start = None
        brace_count = 0
        
        for i, line in enumerate(lines):
            # Skip comments
            if is_comment(line):
                continue
                
            # Look for class declarations
            if "class" in line and "{" in line and ";" not in line and current_class_start is None:
                current_class_start = i
                brace_count = line.count("{") - line.count("}")
            
            # Count braces to find class end
            elif current_class_start is not None:
                brace_count += line.count("{") - line.count("}")
                
                if brace_count == 0:
                    class_bodies.append((current_class_start, i))
                    current_class_start = None
        
        return class_bodies
    
    def _infer_domain_from_code(self, code: str) -> str:
        """
        Infer the domain of the code based on class and variable names.
        
        Args:
            code: The Java code
            
        Returns:
            Inferred domain string
        """
        code_lower = code.lower()
        
        # Check for common domains
        domains = {
            "student_management": ["student", "course", "enroll", "grade", "academic"],
            "file_processing": ["file", "read", "write", "path", "directory"],
            "data_validation": ["validate", "input", "check", "valid", "sanitize"],
            "calculation": ["calculate", "compute", "math", "formula", "result"],
            "inventory_system": ["inventory", "product", "stock", "item", "quantity"],
            "notification_service": ["notify", "message", "alert", "notification", "send"],
            "banking": ["account", "bank", "transaction", "balance", "deposit"],
            "e-commerce": ["cart", "product", "order", "payment", "customer"]
        }
        
        # Count domain-related terms
        domain_scores = {}
        for domain, terms in domains.items():
            score = sum(code_lower.count(term) for term in terms)
            domain_scores[domain] = score
        
        # Return the highest scoring domain, or a default
        if domain_scores:
            max_domain = max(domain_scores.items(), key=lambda x: x[1])
            if max_domain[1] > 0:
                return max_domain[0]
        
        return "general_application"  # Default domain