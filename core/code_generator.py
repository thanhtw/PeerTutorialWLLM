"""
Code Generator module for Java Peer Review Training System.

This module provides the CodeGenerator class which dynamically generates
Java code snippets based on the selected difficulty level and code length,
eliminating the reliance on predefined templates.
"""

import random
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeGenerator:
    """
    Generates Java code snippets dynamically without relying on predefined templates.
    This class creates realistic Java code based on specified complexity and length.
    """
    def __init__(self, llm: BaseLanguageModel = None):
        """
        Initialize the CodeGenerator with an optional language model.
        
        Args:
            llm: Language model to use for code generation
        """
        self.llm = llm
        
        # Define complexity profiles for different code lengths
        self.complexity_profiles = {
            "short": {
                "class_count": 1,
                "method_count_range": (2, 4),
                "field_count_range": (2, 4),
                "imports_count_range": (0, 2),
                "nested_class_prob": 0.1,
                "interface_prob": 0.0
            },
            "medium": {
                "class_count": 1,
                "method_count_range": (3, 6),
                "field_count_range": (3, 6),
                "imports_count_range": (1, 4),
                "nested_class_prob": 0.3,
                "interface_prob": 0.2
            },
            "long": {
                "class_count": 2,
                "method_count_range": (5, 10),
                "field_count_range": (4, 8),
                "imports_count_range": (2, 6),
                "nested_class_prob": 0.5,
                "interface_prob": 0.4
            }
        }
        
        # Common Java domains to make code more realistic
        self.domains = [
            "user_management", "file_processing", "data_validation", 
            "calculation", "inventory_system", "notification_service",
            "logging", "banking", "e-commerce", "student_management"
        ]
    
    def generate_java_code(self, 
                           code_length: str = "medium", 
                           difficulty_level: str = "medium",
                           domain: str = None) -> str:
        """
        Generate Java code with specified length and complexity.
        
        Args:
            code_length: Desired code length (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            domain: Optional domain for the code context
            
        Returns:
            Generated Java code as a string
        """      
        return self._generate_with_llm(code_length, difficulty_level, domain)
        
    def _generate_with_llm(self, code_length: str, difficulty_level: str, domain: str = None) -> str:
        """
        Generate Java code using the language model.
        
        Args:
            code_length: Desired code length (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            domain: Optional domain for the code context
            
        Returns:
            Generated Java code as a string
        """
    
        # Select a domain if not provided
        if not domain:
            domain = random.choice(self.domains)
        
        # Create a detailed prompt for the LLM
        prompt = self._create_generation_prompt(code_length, difficulty_level, domain)
        
        try:
            # Generate the code using the LLM
            logger.info(f"Generating Java code with LLM: {code_length} length, {difficulty_level} difficulty, {domain} domain")
            response = self.llm.invoke(prompt)
            
            # Extract the Java code from the response
            code = self._extract_code_from_response(response)
            
            # Return the generated code
            return code
            
        except Exception as e:
            logger.error(f"Error generating code with LLM: {str(e)}")
            # Return a fallback code snippet instead of None
            return """
    // Fallback code - Simple Java class
    public class FallbackExample {
        public static void main(String[] args) {
            System.out.println("Hello, World!");
        }
    }
    """
           
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Java code from LLM response.
        
        Args:
            response: Full response from the LLM
            
        Returns:
            Extracted Java code
        """
        # Try to extract code blocks
        import re
        code_blocks = re.findall(r'```(?:java)?\s*(.*?)\s*```', response, re.DOTALL)
        
        if code_blocks:
            # Return the largest code block
            return max(code_blocks, key=len)
        
        return response  # If no code blocks found, return the full response
    
    def _create_generation_prompt(self, code_length: str, difficulty_level: str, domain: str) -> str:
        """
        Create a detailed prompt for the LLM to generate Java code.
        
        Args:
            code_length: Desired code length (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            domain: Domain for the code context
            
        Returns:
            Formatted prompt
        """
        complexity_profile = self.complexity_profiles.get(code_length, self.complexity_profiles["medium"])
        
        prompt = f"""
You are a Java programming expert. Create a realistic, working Java code snippet for a {domain} system.

The code should be {code_length} in length and {difficulty_level} in complexity.

Requirements:
- Create approximately {complexity_profile["class_count"]} main class(es)
- Include {complexity_profile["method_count_range"][0]}-{complexity_profile["method_count_range"][1]} methods
- Define {complexity_profile["field_count_range"][0]}-{complexity_profile["field_count_range"][1]} fields/properties
- Use {complexity_profile["imports_count_range"][0]}-{complexity_profile["imports_count_range"][1]} imports
- Include appropriate comments and documentation
- Follow standard Java naming conventions and best practices
- Make the code realistic and representative of real-world Java applications
- Do NOT include any intentional errors or problems

Return only the Java code with no additional explanations.
```java
// Your code here
```
"""
        return prompt
       
    
    
   