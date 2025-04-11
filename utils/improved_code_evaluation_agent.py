"""
Enhanced Code Evaluation Agent with structured JSON responses.

This module provides an improved code evaluation agent that requests
structured JSON with exact error locations and eliminates the need for 
post-processing error enrichment.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseLanguageModel
from utils.export_utils import export_prompt_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedCodeEvaluationAgent:
    """
    Agent for evaluating generated Java code to ensure it meets error requirements.
    
    This agent provides detailed feedback on how well the generated code
    implements the required errors, and suggests improvements for the
    code generator. Gets structured JSON with exact error locations directly from LLM.
    """
    
    def __init__(self, llm: BaseLanguageModel = None, export_debug: bool = True):
        """
        Initialize the CodeEvaluationAgent with optional LLM.
        
        Args:
            llm: Optional language model for evaluation
            export_debug: Whether to export prompts and responses to files
        """
        logger.info("Initializing Improved Code Evaluation Agent")
        self.llm = llm
        self.export_debug = export_debug
        if llm:
            logger.info("LLM provided for code evaluation")
        else:
            logger.info("No LLM provided - evaluation capabilities will be limited")
    
    def evaluate_code(self, code: str, requested_errors: List[Dict[str, Any]], session_id: str = None) -> Dict[str, Any]:
        """
        Evaluate generated code against requested errors with detailed feedback.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            session_id: Optional session ID for export consolidation
            
        Returns:
            Dictionary with evaluation results and feedback
        """
        try:
            logger.info(f"Evaluating code with {len(requested_errors)} requested errors")
            
            # Use LLM-based evaluation if available
            if self.llm:
                validation_results = self._evaluate_with_llm(code, requested_errors, session_id)
                logger.info("Using LLM for code evaluation")
            else:
                logger.warning("No LLM provided for evaluation, returning basic results")
                # Return a basic structure without evaluation
                return {
                    "valid": False,
                    "found_errors": [],
                    "missing_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                                    for error in requested_errors],
                    "error_locations": {},
                    "feedback": "No LLM available for evaluation",
                    "suggestions": []
                }
            
            # Results now come directly structured from the LLM
            return validation_results
            
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
    
    def _evaluate_with_llm(self, code: str, requested_errors: List[Dict[str, Any]], session_id: str = None) -> Dict[str, Any]:
        """
        Evaluate generated code using an LLM to identify if requested errors are present.
        Requests structured JSON with precise error locations directly from the LLM.
        
        Args:
            code: The generated Java code
            requested_errors: List of errors that should be implemented
            session_id: Optional session ID for export consolidation
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.llm:
            logger.warning("No LLM provided for evaluation")
            return {
                "valid": False,
                "found_errors": [],
                "missing_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                                for error in requested_errors],
                "error_locations": {},
                "feedback": "No LLM available for evaluation"
            }
        
        # Format the requested errors for the prompt
        error_descriptions = []
        for i, error in enumerate(requested_errors, 1):
            error_type = error.get("type", "").upper()
            name = error.get("name", "")
            description = error.get("description", "")
            implementation_guide = error.get("implementation_guide", "")
            
            error_descriptions.append(f"{i}. {error_type} ERROR - {name}\n   Description: {description}\n   Implementation guide: {implementation_guide}")
        
        error_list = "\n\n".join(error_descriptions)
        
        # Create an improved prompt for the LLM that will return well-structured JSON
        prompt = f"""You are an expert Java code reviewer tasked with evaluating whether a code snippet correctly implements specific errors.

CODE SNIPPET:
```java
{code}
```

REQUESTED ERRORS TO IMPLEMENT:
{error_list}

Your task is to analyze the code and determine exactly where and how each requested error is implemented.

IMPORTANT: You must return a valid, parseable JSON object with the following structure:

```json
{{
  "found_errors": [
    {{
      "error_type": "BUILD",
      "error_name": "NullPointerException",
      "line_number": 42,
      "code_segment": "String str = null; int length = str.length();",
      "explanation": "This will throw NullPointerException because str is null when length() is called",
      "context": "// Method context lines\nString str = null;\nint length = str.length();"
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

For each found error:
1. Provide the exact line number where the error occurs
2. Include the specific code segment that contains the error
3. Include 2-3 lines of surrounding code context
4. Provide a clear explanation of how the error is implemented

For each missing error:
1. Explain why you couldn't find the error in the code
2. Suggest how it could be implemented

Please ensure your response contains ONLY the JSON object described above, with no additional text, explanation, or markdown formatting. The JSON must be valid and parseable.
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
                    session_id=session_id,
                    error_list=requested_errors
                )
            
            # Extract JSON from response
            try:
                # First try to parse the entire response as JSON
                try:
                    analysis = json.loads(response)
                except:
                    # If that fails, try to extract JSON using regex
                    import re
                    json_match = re.search(r'({.*})', response.replace('\n', ' '), re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        analysis = json.loads(json_str)
                    else:
                        raise ValueError("Could not extract JSON from LLM response")
                
                # Process the analysis results
                found_errors = []
                missing_errors = []
                error_locations = {}
                suggestions = []
                
                # Extract found errors
                for error in analysis.get("found_errors", []):
                    error_type = error.get("error_type", "")
                    error_name = error.get("error_name", "")
                    error_key = f"{error_type} - {error_name}"
                    
                    found_errors.append(error_key)
                    error_locations[error_key] = error.get("line_number", 0)
                    
                    # Add suggestions from the context
                    if "explanation" in error:
                        suggestions.append({
                            "error_key": error_key, 
                            "suggestions": [error["explanation"]]
                        })
                
                # Extract missing errors
                for error in analysis.get("missing_errors", []):
                    error_type = error.get("error_type", "")
                    error_name = error.get("error_name", "")
                    error_key = f"{error_type} - {error_name}"
                    
                    missing_errors.append(error_key)
                    
                    # Add suggestions for implementation
                    if "explanation" in error:
                        suggestions.append({
                            "error_key": error_key, 
                            "suggestions": [error["explanation"]]
                        })
                
                # Check if we're missing any errors that weren't explicitly mentioned
                all_requested_keys = [f"{error.get('type', '').upper()} - {error.get('name', '')}" for error in requested_errors]
                for key in all_requested_keys:
                    if key not in found_errors and key not in missing_errors:
                        missing_errors.append(key)
                
                # Enhanced validation results with complete information directly from LLM
                validation_results = {
                    "valid": len(missing_errors) == 0,  # Valid only if no errors are missing
                    "found_errors": found_errors,
                    "missing_errors": missing_errors,
                    "error_locations": error_locations,
                    "feedback": analysis.get("feedback", ""),
                    "detailed_analysis": analysis,  # Keep the full LLM analysis
                    "suggestions": suggestions
                }
                
                # Export the evaluation results if export_debug is enabled
                if self.export_debug:
                    export_prompt_response(
                        prompt="", 
                        response="", 
                        operation_type="evaluation_results",
                        session_id=session_id,
                        error_list=requested_errors,
                        evaluation_result=validation_results
                    )
                
                return validation_results
            
            except Exception as e:
                logger.error(f"Error parsing JSON from LLM response: {str(e)}")
                logger.error(f"Raw response: {response[:500]}...")
                # Return basic structure
                return {
                    "valid": False,
                    "found_errors": [],
                    "missing_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                                    for error in requested_errors],
                    "error_locations": {},
                    "feedback": f"Error parsing LLM response: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error evaluating code with LLM: {str(e)}")
            # Return a basic structure
            return {
                "valid": False,
                "found_errors": [],
                "missing_errors": [f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                                for error in requested_errors],
                "error_locations": {},
                "feedback": f"Error evaluating with LLM: {str(e)}"
            }
    
    def generate_improved_prompt(self, code: str, requested_errors: List[Dict[str, Any]], 
                              evaluation: Dict[str, Any], session_id: str = None) -> str:
        """
        Generate an improved prompt for the code generator based on LLM evaluation.
        
        Args:
            code: The previously generated code
            requested_errors: List of errors that should be implemented
            evaluation: Evaluation results from evaluate_code method
            session_id: Optional session ID for export consolidation
            
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
        if "feedback" in evaluation:
            prompt += f"\n{evaluation['feedback']}\n"
        
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
            # Fall back to simple approach
            if evaluation["found_errors"]:
                prompt += "\n### What was implemented correctly:\n"
                for error_key in evaluation["found_errors"]:
                    prompt += f"-  {error_key}\n"
            
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
                session_id=session_id,
                error_list=requested_errors,
                evaluation_result=evaluation
            )
        
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