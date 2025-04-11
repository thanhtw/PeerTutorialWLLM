"""
Dedicated Code Generation Agent for Java Peer Review Training System.

This module provides a specialized CodeGenerationAgent that handles all aspects
of Java code generation with intentional errors, ensuring consistency and quality
in the generated code.
"""

import random
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain_core.language_models import BaseLanguageModel
from utils.code_utils import create_code_generation_prompt, extract_code_from_response, strip_error_annotations
from utils.improved_export_utils import export_prompt_response, generate_session_id

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DedicatedCodeGenerationAgent:
    """
    Specialized agent for generating Java code with intentional errors.
    
    This agent handles all aspects of code generation, ensuring that:
    1. All code is generated consistently with the same style
    2. Errors are implemented according to requirements
    3. Code is properly annotated with error comments
    4. Regeneration improves code based on evaluation feedback
    """
    
    def __init__(self, llm: BaseLanguageModel = None, export_debug: bool = True):
        """
        Initialize the CodeGenerationAgent with optional LLM.
        
        Args:
            llm: Optional language model for code generation
            export_debug: Whether to export prompts and responses to files
        """
        self.llm = llm
        self.export_debug = export_debug
        self.domains = [
            "student_management", "file_processing", "data_validation", 
            "calculation", "inventory_system", "notification_service",
            "logging", "banking", "e-commerce", "student_management"
        ]
        
        # Track generated code for attribution
        self.generation_metadata = {}
        
        # Create a unique agent ID for tracking
        self.agent_id = f"code_gen_agent_{generate_session_id()}"
        
        logger.info(f"Initialized CodeGenerationAgent with ID: {self.agent_id}")
    
    def generate_code_with_errors(
        self, 
        code_length: str, 
        difficulty_level: str, 
        selected_errors: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Tuple[str, str, List[str]]:
        """
        Generate Java code with the specified errors.
        
        Args:
            code_length: Desired code length (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            selected_errors: List of errors to include in the code
            session_id: Optional session ID for export consolidation
            domain: Optional domain for the code context
            
        Returns:
            Tuple of (annotated_code, clean_code, detailed_problems)
        """
        logger.info(f"Generating code with {len(selected_errors)} errors")
        logger.info(f"Parameters: length={code_length}, difficulty={difficulty_level}, domain={domain}")
        
        # Create a new session ID if not provided
        if not session_id:
            session_id = generate_session_id()
            logger.info(f"Created new session ID: {session_id}")
            
        # Track this generation
        generation_id = f"gen_{random.randint(1000, 9999)}"
        self.generation_metadata[generation_id] = {
            "session_id": session_id,
            "code_length": code_length,
            "difficulty_level": difficulty_level,
            "selected_errors": [f"{error.get('type', '')} - {error.get('name', '')}" for error in selected_errors],
            "domain": domain,
            "timestamp": self._get_timestamp()
        }
        
        # Select domain if not provided
        if not domain:
            domain = random.choice(self.domains)
            logger.info(f"Selected random domain: {domain}")
        
        # Create a detailed prompt for the LLM
        prompt = create_code_generation_prompt(
            code_length=code_length,
            difficulty_level=difficulty_level,
            selected_errors=selected_errors,
            domain=domain,
            include_error_annotations=True
        )
        
        # Add our agent signature to ensure attribution
        prompt += f"\n\nAGENT_ID: {self.agent_id}\nGENERATION_ID: {generation_id}\n"
        
        try:
            # Generate code using the LLM
            if self.llm:
                # Get response from LLM
                response = self.llm.invoke(prompt)
                
                # Export the prompt and response if export_debug is enabled
                if self.export_debug:
                    export_prompt_response(
                        prompt=prompt, 
                        response=str(response), 
                        operation_type="code_generation",
                        session_id=session_id,
                        error_list=selected_errors
                    )
                
                # Extract the code with annotations
                annotated_code = extract_code_from_response(response)
                
                # Add our generation signature as a comment
                annotated_code = f"// Generated by {self.agent_id}\n// Generation ID: {generation_id}\n\n" + annotated_code
                
                # Create clean version by stripping annotations
                clean_code = strip_error_annotations(annotated_code)
                
                # Create detailed problem descriptions
                detailed_problems = self._extract_problem_descriptions(annotated_code, selected_errors)
                
                # Update metadata with success
                self.generation_metadata[generation_id]["status"] = "success"
                self.generation_metadata[generation_id]["code_length_chars"] = len(annotated_code)
                self.generation_metadata[generation_id]["error_annotations_count"] = annotated_code.count("// ERROR:")
                
                return annotated_code, clean_code, detailed_problems
            else:
                logger.warning("No LLM provided for code generation, returning fallback code")
                # Return fallback code
                self.generation_metadata[generation_id]["status"] = "fallback_no_llm"
                
                fallback_code = f"""
// Generated by {self.agent_id} (FALLBACK)
// Generation ID: {generation_id}

/**
 * Fallback code generated when no LLM is available
 * This is a sample class with intentional errors for review practice
 */
public class FallbackExample {{
    // ERROR: CHECKSTYLE - MemberName - Variable uses non-standard naming
    private String User_Name;
    
    // ERROR: BUILD - NullPointerException - Accessing property of null object
    public void processData() {{
        String data = null;
        int length = data.length();
        System.out.println("Length: " + length);
    }}
    
    // Missing return statement
    public int calculateValue() {{
        int result = 10;
        // Missing return statement here
    }}
}}
"""
                clean_code = strip_error_annotations(fallback_code)
                detailed_problems = [
                    "CHECKSTYLE ERROR - MemberName: Variable names should use camelCase",
                    "BUILD ERROR - NullPointerException: Accessing property of null object",
                    "BUILD ERROR - Missing return statement: Method doesn't return a value"
                ]
                print("fallback_code", fallback_code)
                
                return fallback_code, clean_code, detailed_problems
                
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            self.generation_metadata[generation_id]["status"] = "error"
            self.generation_metadata[generation_id]["error"] = str(e)
            
            # Return a minimal error fallback
            error_fallback = f"""
// Generated by {self.agent_id} (ERROR FALLBACK)
// Generation ID: {generation_id}
// Error: {str(e)}

/**
 * Error fallback code - could not generate proper code
 * This is a sample class with intentional errors for review practice
 */
public class ErrorFallback {{
    // ERROR: CHECKSTYLE - MemberName - Variable uses non-standard naming
    private String Error_Message = "Code generation failed";
    
    // ERROR: BUILD - NullPointerException - Accessing property of null object
    public void showError() {{
        String data = null;
        int length = data.length();
    }}
}}
"""
            clean_code = strip_error_annotations(error_fallback)
            detailed_problems = [
                "CHECKSTYLE ERROR - MemberName: Variable names should use camelCase",
                "BUILD ERROR - NullPointerException: Accessing property of null object"
            ]
            
            return error_fallback, clean_code, detailed_problems
    
    def regenerate_code(
        self,
        previous_code: str,
        selected_errors: List[Dict[str, Any]],
        evaluation_feedback: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Tuple[str, str, List[str]]:
        """
        Regenerate code based on evaluation feedback.
        
        Args:
            previous_code: Previous code generation attempt
            selected_errors: List of errors that should be implemented
            evaluation_feedback: Feedback from code evaluation
            session_id: Optional session ID for export consolidation
            
        Returns:
            Tuple of (annotated_code, clean_code, detailed_problems)
        """
        logger.info("Regenerating code based on evaluation feedback")
        
        # Create a new session ID if not provided
        if not session_id:
            session_id = generate_session_id()
            logger.info(f"Created new session ID: {session_id}")
            
        # Track this regeneration
        regeneration_id = f"regen_{random.randint(1000, 9999)}"
        self.generation_metadata[regeneration_id] = {
            "session_id": session_id,
            "operation": "regeneration",
            "selected_errors": [f"{error.get('type', '')} - {error.get('name', '')}" for error in selected_errors],
            "timestamp": self._get_timestamp(),
            "previous_code_length": len(previous_code),
            "found_errors": len(evaluation_feedback.get("found_errors", [])),
            "missing_errors": len(evaluation_feedback.get("missing_errors", []))
        }
        
        try:
            # If we have a specific improved prompt generator, use it
            if hasattr(self, 'improved_prompt_generator') and callable(getattr(self, 'improved_prompt_generator')):
                prompt = self.improved_prompt_generator(
                    previous_code, 
                    selected_errors, 
                    evaluation_feedback
                )
            else:
                # Otherwise, create a basic regeneration prompt
                prompt = self._create_regeneration_prompt(
                    previous_code, 
                    selected_errors, 
                    evaluation_feedback
                )
            
            # Add our agent signature to ensure attribution
            prompt += f"\n\nAGENT_ID: {self.agent_id}\nREGENERATION_ID: {regeneration_id}\n"
            
            # Generate code using the LLM
            if self.llm:
                # Get response from LLM
                response = self.llm.invoke(prompt)
                
                # Export the prompt and response if export_debug is enabled
                if self.export_debug:
                    export_prompt_response(
                        prompt=prompt, 
                        response=str(response), 
                        operation_type="code_regeneration",
                        session_id=session_id,
                        error_list=selected_errors,
                        evaluation_result=evaluation_feedback
                    )
                
                # Extract the code with annotations
                annotated_code = extract_code_from_response(response)
                
                # Add our regeneration signature as a comment
                annotated_code = f"// Regenerated by {self.agent_id}\n// Regeneration ID: {regeneration_id}\n\n" + annotated_code
                
                # Create clean version by stripping annotations
                clean_code = strip_error_annotations(annotated_code)
                
                # Create detailed problem descriptions
                detailed_problems = self._extract_problem_descriptions(annotated_code, selected_errors)
                
                # Update metadata with success
                self.generation_metadata[regeneration_id]["status"] = "success"
                self.generation_metadata[regeneration_id]["code_length_chars"] = len(annotated_code)
                self.generation_metadata[regeneration_id]["error_annotations_count"] = annotated_code.count("// ERROR:")
                
                return annotated_code, clean_code, detailed_problems
                
            else:
                logger.warning("No LLM provided for code regeneration, returning original code")
                # Return original code with updated errors based on feedback
                self.generation_metadata[regeneration_id]["status"] = "fallback_no_llm"
                
                # Add missing error annotations manually
                enhanced_code = self._manually_add_missing_errors(
                    previous_code, 
                    evaluation_feedback.get("missing_errors", [])
                )
                
                # Add our regeneration signature as a comment
                enhanced_code = f"// Regenerated by {self.agent_id} (FALLBACK)\n// Regeneration ID: {regeneration_id}\n\n" + enhanced_code
                
                # Create clean version by stripping annotations
                clean_code = strip_error_annotations(enhanced_code)
                
                # Create detailed problem descriptions
                detailed_problems = self._extract_problem_descriptions(enhanced_code, selected_errors)
                
                return enhanced_code, clean_code, detailed_problems
        
        except Exception as e:
            logger.error(f"Error regenerating code: {str(e)}")
            self.generation_metadata[regeneration_id]["status"] = "error"
            self.generation_metadata[regeneration_id]["error"] = str(e)
            
            # Return original code with a warning comment
            warning_code = f"""
// Regeneration failed - using original code
// Error: {str(e)}

{previous_code}
"""
            clean_code = strip_error_annotations(warning_code)
            detailed_problems = self._extract_problem_descriptions(previous_code, selected_errors)
            
            return warning_code, clean_code, detailed_problems
    
    def _create_regeneration_prompt(
        self,
        previous_code: str,
        selected_errors: List[Dict[str, Any]],
        evaluation_feedback: Dict[str, Any]
    ) -> str:
        """
        Create a prompt for code regeneration based on evaluation feedback.
        
        Args:
            previous_code: Previous code generation attempt
            selected_errors: List of errors that should be implemented
            evaluation_feedback: Feedback from code evaluation
            
        Returns:
            Regeneration prompt
        """
        # Start with general guidance
        prompt = "You are an expert Java programmer tasked with improving code to include specific errors for training purposes.\n\n"
        
        # Add feedback about what's missing
        prompt += "## EVALUATION FEEDBACK\n\n"
        
        # Add information about found errors
        found_errors = evaluation_feedback.get("found_errors", [])
        if found_errors:
            prompt += "### CORRECTLY IMPLEMENTED ERRORS:\n"
            for error in found_errors:
                prompt += f"- {error}\n"
            prompt += "\n"
        
        # Add information about missing errors
        missing_errors = evaluation_feedback.get("missing_errors", [])
        if missing_errors:
            prompt += "### ERRORS THAT NEED TO BE IMPLEMENTED:\n"
            for error in missing_errors:
                prompt += f"- {error}\n"
                
                # Find detailed information about this error
                for req_error in selected_errors:
                    req_error_key = f"{req_error.get('type', '').upper()} - {req_error.get('name', '')}"
                    if req_error_key == error:
                        description = req_error.get("description", "")
                        implementation_guide = req_error.get("implementation_guide", "")
                        
                        prompt += f"  Description: {description}\n"
                        prompt += f"  Implementation guide: {implementation_guide}\n\n"
            prompt += "\n"
        
        # Add overall feedback
        if "feedback" in evaluation_feedback:
            prompt += f"### OVERALL FEEDBACK:\n{evaluation_feedback['feedback']}\n\n"
        
        # Add instructions for improvement
        prompt += "## INSTRUCTIONS\n\n"
        prompt += "1. Keep the correct implementations from the previous code.\n"
        prompt += "2. Add implementation for all missing errors.\n"
        prompt += "3. Make sure to add error annotations in the standard format: `// ERROR: [TYPE] - [NAME] - [Brief explanation]`\n"
        prompt += "4. Return your final code in a code block with ``` delimiters.\n\n"
        
        # Add the previous code
        prompt += "## PREVIOUS CODE TO IMPROVE:\n\n"
        prompt += "```java\n"
        prompt += previous_code
        prompt += "\n```\n\n"
        
        return prompt
    
    def _manually_add_missing_errors(self, code: str, missing_errors: List[str]) -> str:
        """
        Manually add missing error annotations to code as a fallback.
        
        Args:
            code: Original code
            missing_errors: List of missing error identifiers
            
        Returns:
            Code with additional error annotations
        """
        lines = code.splitlines()
        result_lines = []
        
        # Very basic addition of errors - not ideal but helps in fallback scenarios
        for i, line in enumerate(lines):
            result_lines.append(line)
            
            # Add errors at strategic points
            if i == 5 and missing_errors:  # After class declaration or imports
                error = missing_errors[0]
                if "membername" in error.lower() or "variablename" in error.lower():
                    result_lines.append(f"    // ERROR: {error} - Variable uses non-standard naming")
                    result_lines.append(f"    private String Some_Variable;")
                    missing_errors = missing_errors[1:]
            
            elif i == 10 and missing_errors:  # Around method declaration
                error = missing_errors[0]
                if "nullpointer" in error.lower():
                    result_lines.append(f"    // ERROR: {error} - Accessing null object")
                    result_lines.append(f"    public void nullErrorMethod() {{")
                    result_lines.append(f"        String data = null;")
                    result_lines.append(f"        int length = data.length();")
                    result_lines.append(f"    }}")
                    missing_errors = missing_errors[1:]
                elif "return" in error.lower():
                    result_lines.append(f"    // ERROR: {error} - Method missing return statement")
                    result_lines.append(f"    public int missingReturnMethod() {{")
                    result_lines.append(f"        int value = 10;")
                    result_lines.append(f"        // Missing return statement")
                    result_lines.append(f"    }}")
                    missing_errors = missing_errors[1:]
            
            elif i == 20 and missing_errors:  # Later in the code
                error = missing_errors[0]
                if "array" in error.lower() or "index" in error.lower():
                    result_lines.append(f"    // ERROR: {error} - Array index out of bounds")
                    result_lines.append(f"    public void arrayErrorMethod() {{")
                    result_lines.append(f"        int[] array = new int[3];")
                    result_lines.append(f"        int value = array[5]; // Out of bounds")
                    result_lines.append(f"    }}")
                    missing_errors = missing_errors[1:]
                else:
                    # Generic error
                    result_lines.append(f"    // ERROR: {error} - Generic implementation")
                    result_lines.append(f"    public void genericErrorMethod() {{")
                    result_lines.append(f"        // Implementation of {error}")
                    result_lines.append(f"        System.out.println(\"Error implementation\");")
                    result_lines.append(f"    }}")
                    missing_errors = missing_errors[1:]
        
        # Add any remaining errors at the end
        for error in missing_errors:
            result_lines.append(f"    // ERROR: {error} - Fallback implementation")
            result_lines.append(f"    public void fallbackErrorMethod() {{")
            result_lines.append(f"        // Implementation of {error}")
            result_lines.append(f"        System.out.println(\"Error implementation\");")
            result_lines.append(f"    }}")
        
        return "\n".join(result_lines)
    
    def _extract_problem_descriptions(self, annotated_code: str, selected_errors: List[Dict[str, Any]]) -> List[str]:
        """
        Extract problem descriptions from annotated code or create from selected errors.
        
        Args:
            annotated_code: Code with error annotations
            selected_errors: List of errors that should be implemented
            
        Returns:
            List of problem descriptions
        """
        problems = []
        
        # Try to extract problems from error annotations
        lines = annotated_code.splitlines()
        for line in lines:
            if "// ERROR:" in line:
                # Extract the error description
                description = line.split("// ERROR:")[1].strip()
                problems.append(description)
        
        # If no problems were extracted, create from selected errors
        if not problems:
            for error in selected_errors:
                error_type = error.get("type", "unknown").upper()
                name = error.get("name", "unknown")
                description = error.get("description", "")
                problems.append(f"{error_type} ERROR - {name}: {description}")
        
        return problems
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def set_improved_prompt_generator(self, generator_function):
        """
        Set a custom function for generating improved prompts during regeneration.
        
        Args:
            generator_function: Function that takes (code, errors, feedback) and returns a prompt
        """
        self.improved_prompt_generator = generator_function
        
    def get_generation_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the generation metadata.
        
        Returns:
            Dictionary with generation metadata
        """
        return self.generation_metadata
    
    def clear_metadata(self):
        """Clear the generation metadata."""
        self.generation_metadata = {}