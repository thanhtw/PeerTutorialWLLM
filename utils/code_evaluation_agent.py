"""
Code Evaluation Agent for Java Peer Review Training System.

This module provides the CodeEvaluationAgent class which evaluates 
generated Java code to ensure it contains the required errors.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from utils.error_validation import validate_code_errors, is_comment, is_primitive_or_common

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeEvaluationAgent:
    """
    Agent for evaluating generated Java code to ensure it meets error requirements.
    
    This agent provides detailed feedback on how well the generated code
    implements the required errors, and suggests improvements for the
    code generator.
    """
    
    def __init__(self):
        """Initialize the CodeEvaluationAgent."""
        logger.info("Initializing Code Evaluation Agent")
    
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
            
            # Get basic validation results
            validation_results = validate_code_errors(code, requested_errors)
            
            # Create more detailed evaluation
            evaluation = {
                "valid": validation_results["valid"],
                "found_errors": validation_results["found_errors"],
                "missing_errors": validation_results["missing_errors"],
                "error_locations": validation_results["error_locations"],
                "feedback": self._generate_feedback(code, requested_errors, validation_results),
                "suggestions": self._generate_suggestions(code, requested_errors, validation_results)
            }
            
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
    
    def _generate_feedback(self, code: str, requested_errors: List[Dict[str, Any]], 
                          validation_results: Dict[str, Any]) -> str:
        """
        Generate detailed feedback on the implementation of errors.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            validation_results: Results from basic validation
            
        Returns:
            Detailed feedback string
        """
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
            validation_results: Results from basic validation
            
        Returns:
            List of suggestion dictionaries
        """
        lines = code.splitlines()
        suggestions = []
        
        # Generate suggestions for each missing error
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
            
            # Generate specific suggestions based on error type
            if "Cannot find symbol" in error_name:
                suggestion["suggestions"].append(
                    "Try using a variable that hasn't been declared, e.g., 'int result = undeclaredVar + 5;'"
                )
                
                # Find possible insertion points
                method_bodies = self._find_method_bodies(code)
                if method_bodies:
                    method_start, method_end = method_bodies[0]
                    suggestion["insertion_point"] = method_start + 2  # Inside the method
                    suggestion["sample_code"] = "int result = undeclaredVar + 5; // Using undeclared variable"
            
            elif "Incompatible types" in error_name:
                suggestion["suggestions"].append(
                    "Try assigning a String to an int variable, e.g., 'int value = \"hello\";'"
                )
                
                # Find possible insertion points
                method_bodies = self._find_method_bodies(code)
                if method_bodies:
                    method_start, method_end = method_bodies[0]
                    suggestion["insertion_point"] = method_start + 2  # Inside the method
                    suggestion["sample_code"] = "int value = \"hello\"; // Incompatible types"
            
            elif "Missing return statement" in error_name:
                suggestion["suggestions"].append(
                    "Create a non-void method without a return statement in some execution path"
                )
                
                method_bodies = self._find_method_bodies(code)
                if method_bodies:
                    suggestion["insertion_point"] = method_bodies[0][0]  # Start of first method
                    suggestion["sample_code"] = """public int calculateValue(int input) {
    if (input > 0) {
        return input * 2;
    }
    // Missing return statement for else case
}"""
            
            elif "MemberName" in error_name:
                suggestion["suggestions"].append(
                    "Define a member variable with improper naming (using underscore or starting with uppercase)"
                )
                
                # Find class declarations
                class_bodies = self._find_class_bodies(code)
                if class_bodies:
                    class_start, class_end = class_bodies[0]
                    suggestion["insertion_point"] = class_start + 1  # After class declaration
                    suggestion["sample_code"] = "private String User_Name; // Improper member naming"
            
            elif "MethodName" in error_name:
                suggestion["suggestions"].append(
                    "Define a method with improper naming (starting with uppercase)"
                )
                
                # Find class declarations
                class_bodies = self._find_class_bodies(code)
                if class_bodies:
                    class_start, class_end = class_bodies[0]
                    suggestion["insertion_point"] = class_start + 3  # Inside class
                    suggestion["sample_code"] = "public void PrintMessage() { } // Improper method naming"
            
            elif "TypeName" in error_name:
                suggestion["suggestions"].append(
                    "Define a class with improper naming (starting with lowercase)"
                )
                
                # Find insertion point near the top of the file
                suggestion["insertion_point"] = 5  # Near the top
                suggestion["sample_code"] = "class myClass { } // Improper class naming"
            
            elif "NullPointerException" in error_name:
                suggestion["suggestions"].append(
                    "Create an object reference set to null and then call a method on it"
                )
                
                # Find possible insertion points
                method_bodies = self._find_method_bodies(code)
                if method_bodies:
                    method_start, method_end = method_bodies[0]
                    suggestion["insertion_point"] = method_start + 2  # Inside the method
                    suggestion["sample_code"] = "String str = null;\nint length = str.length(); // Will cause NullPointerException"
            
            elif "String comparison using ==" in error_name:
                suggestion["suggestions"].append(
                    "Compare two strings using == instead of equals()"
                )
                
                method_bodies = self._find_method_bodies(code)
                if method_bodies:
                    method_start, method_end = method_bodies[0]
                    suggestion["insertion_point"] = method_start + 2  # Inside the method
                    suggestion["sample_code"] = """String s1 = "hello";
String s2 = "h" + "ello";
if (s1 == s2) { // Using == instead of equals()
    System.out.println("Strings are equal");
}"""
            
            # Add generic suggestions for any other error types
            else:
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
    
    def generate_improved_prompt(self, code: str, requested_errors: List[Dict[str, Any]], 
                           evaluation: Dict[str, Any]) -> str:
        """
        Generate an improved prompt for the code generator based on evaluation.
        Enhanced with clearer feedback and specific guidance for missing errors.
        
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
        
        if evaluation["found_errors"]:
            prompt += "\n### What was implemented correctly:\n"
            for error_key in evaluation["found_errors"]:
                prompt += f"- ✅ {error_key}\n"
                    
                # Include the exact line where the error was found, if available
                line_num = evaluation["error_locations"].get(error_key, 0)
                if line_num > 0:
                    lines = code.splitlines()
                    line_content = lines[line_num-1] if 0 < line_num <= len(lines) else "Unknown"
                    prompt += f"  Found at line {line_num}: `{line_content.strip()}`\n"
        
        if evaluation["missing_errors"]:
            prompt += "\n### Errors that need to be implemented:\n"
            
            for error_key in evaluation["missing_errors"]:
                prompt += f"- ❌ {error_key}\n"
                
                # Find the corresponding error details
                for error in requested_errors:
                    if f"{error.get('type', '').upper()} - {error.get('name', '')}" == error_key:
                        error_type = error.get("type", "").upper()
                        name = error.get("name", "")
                        description = error.get("description", "")
                        implementation_guide = error.get("implementation_guide", "")
                        
                        prompt += f"  Description: {description}\n"
                        prompt += f"  Implementation guide: {implementation_guide}\n\n"
                        
                        # Add specific suggestions from the evaluation
                        for suggestion in evaluation.get("suggestions", []):
                            if suggestion.get("error_key") == error_key:
                                for tip in suggestion.get("suggestions", []):
                                    prompt += f"  Suggestion: {tip}\n"
                                if "sample_code" in suggestion:
                                    prompt += f"  Sample code: `{suggestion['sample_code']}`\n\n"
        
        prompt += "\n## SPECIFIC INSTRUCTIONS FOR THIS ATTEMPT\n"
        prompt += "\n1. Please revise the code to implement ALL requested errors. Be sure to follow the implementation guides."
        prompt += "\n2. Make sure to add error annotations in the standard format: `// ERROR: [TYPE] - [NAME] - [Brief explanation]`"
        prompt += "\n3. Keep your correct implementations from the previous attempt while adding the missing errors."
        prompt += "\n4. Focus especially on implementing the missing errors highlighted above."
        prompt += "\n5. Return your final code in a code block with ``` delimiters."
        
        prompt += "\n\nHere's the previous code to improve upon:\n\n```java\n"
        prompt += code
        prompt += "\n```"
        
        return prompt
    
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