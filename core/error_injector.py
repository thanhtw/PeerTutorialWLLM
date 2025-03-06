"""
Error Injector module for Java Peer Review Training System.

This module uses an LLM to inject specific errors into Java code snippets
based on selected error categories from JSON files.
"""

import json
import logging
import requests
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorInjector:
    """
    Injects specific errors into Java code snippets using an LLM.
    
    This class handles loading error data from JSON files, selecting specific errors,
    and using an LLM to generate Java code with those errors.
    """
    
    def __init__(self, llm: BaseLanguageModel = None, 
                 build_errors_path: str = "build_errors.json",
                 checkstyle_errors_path: str = "checkstyle_error.json"):
        """
        Initialize the ErrorInjector with error data sources.
        
        Args:
            llm: Language model to use for error injection
            build_errors_path: Path to the build errors JSON file
            checkstyle_errors_path: Path to the checkstyle errors JSON file
        """
        self.llm = llm
        
        # Load error data
        self.build_errors = self._load_json_data(build_errors_path)
        self.checkstyle_errors = self._load_json_data(checkstyle_errors_path)
        
        # Keep track of all available error categories
        self.build_categories = list(self.build_errors.keys()) if self.build_errors else []
        self.checkstyle_categories = list(self.checkstyle_errors.keys()) if self.checkstyle_errors else []
        
        # Set Ollama base URL
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def _load_json_data(self, file_path: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Load JSON data from a file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Loaded JSON data as a dictionary or empty dict if loading fails
        """
        try:
            # Try different paths to find the file
            dir_path = os.path.dirname(os.path.realpath(__file__))
            parent_dir = os.path.dirname(dir_path)  # Get parent directory
            
            # Try different paths to find the file
            paths_to_try = [
                file_path,  # Try direct path first
                os.path.join(dir_path, file_path),  # Try in the same directory
                os.path.join(parent_dir, file_path),  # Try in parent directory
                os.path.join(parent_dir, "data", file_path)  # Try in data subdirectory
            ]
            
            for path in paths_to_try:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        return json.load(f)
            
            logger.warning(f"Could not find error data file: {file_path}")
            return {}
            
        except Exception as e:
            logger.error(f"Error loading error data from {file_path}: {str(e)}")
            return {}
    
    def get_all_error_categories(self) -> Dict[str, List[str]]:
        """
        Get all available error categories from both build errors and checkstyle errors.
        
        Returns:
            Dict[str, List[str]]: Dictionary with 'build' and 'checkstyle' categories
        """
        return {
            "build": self.build_categories,
            "checkstyle": self.checkstyle_categories
        }
    
    def select_errors(self, selected_categories: Dict[str, List[str]], 
                      difficulty_level: str = "medium",
                      error_counts: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """
        Select errors based on provided categories and difficulty.
        
        Args:
            selected_categories: Dictionary with 'build' and 'checkstyle' keys, 
                               each containing a list of selected categories
            difficulty_level: Difficulty level (easy, medium, hard)
            error_counts: Optional dictionary with 'build' and 'checkstyle' keys,
                        each containing the number of errors to select
            
        Returns:
            List of selected error dictionaries
        """
        import random
        selected_errors = []
        
        # Determine number of errors based on difficulty if not specified
        if error_counts is None:
            total_errors = {
                "easy": 2,
                "medium": 4,
                "hard": 6
            }.get(difficulty_level.lower(), 3)
            
            # Calculate number of errors for each type
            build_categories = selected_categories.get("build", [])
            checkstyle_categories = selected_categories.get("checkstyle", [])
            
            total_categories = len(build_categories) + len(checkstyle_categories)
            
            if total_categories == 0:
                # If no categories selected, default to some standard categories
                build_categories = ["CompileTimeErrors", "RuntimeErrors"] if "CompileTimeErrors" in self.build_categories else self.build_categories[:2]
                checkstyle_categories = ["NamingConventionChecks", "WhitespaceAndFormattingChecks"] if "NamingConventionChecks" in self.checkstyle_categories else self.checkstyle_categories[:2]
                selected_categories = {"build": build_categories, "checkstyle": checkstyle_categories}
                total_categories = len(build_categories) + len(checkstyle_categories)
            
            # Distribute errors between build and checkstyle
            if total_categories > 0:
                build_proportion = len(build_categories) / total_categories
                build_count = max(1, round(total_errors * build_proportion)) if build_categories else 0
                checkstyle_count = max(1, total_errors - build_count) if checkstyle_categories else 0
            else:
                build_count = total_errors // 2
                checkstyle_count = total_errors - build_count
            
            error_counts = {
                "build": build_count,
                "checkstyle": checkstyle_count
            }
        
        # Select build errors
        for category in selected_categories.get("build", []):
            if category in self.build_errors:
                errors = self.build_errors[category]
                if errors:
                    # Randomly select an error from this category
                    selected_error = random.choice(errors)
                    selected_errors.append({
                        "type": "build",
                        "category": category,
                        "name": selected_error["error_name"],
                        "description": selected_error["description"]
                    })
        
        # Select checkstyle errors
        for category in selected_categories.get("checkstyle", []):
            if category in self.checkstyle_errors:
                errors = self.checkstyle_errors[category]
                if errors:
                    # Randomly select an error from this category
                    selected_error = random.choice(errors)
                    selected_errors.append({
                        "type": "checkstyle",
                        "category": category,
                        "name": selected_error["check_name"],
                        "description": selected_error["description"]
                    })
        
        # If we have more errors than needed, randomly select the required number
        build_count = error_counts.get("build", 0)
        checkstyle_count = error_counts.get("checkstyle", 0)
        
        # Limit build errors if needed
        build_errors = [e for e in selected_errors if e["type"] == "build"]
        if len(build_errors) > build_count:
            build_errors = random.sample(build_errors, build_count)
        
        # Limit checkstyle errors if needed
        checkstyle_errors = [e for e in selected_errors if e["type"] == "checkstyle"]
        if len(checkstyle_errors) > checkstyle_count:
            checkstyle_errors = random.sample(checkstyle_errors, checkstyle_count)
        
        # Combine the selected errors
        final_errors = build_errors + checkstyle_errors
        
        # Ensure we have at least some errors if nothing was selected
        if not final_errors:
            # Default to one build error and one checkstyle error
            if self.build_errors:
                first_category = list(self.build_errors.keys())[0]
                if self.build_errors[first_category]:
                    build_error = random.choice(self.build_errors[first_category])
                    final_errors.append({
                        "type": "build",
                        "category": first_category,
                        "name": build_error["error_name"],
                        "description": build_error["description"]
                    })
                
            if self.checkstyle_errors:
                first_category = list(self.checkstyle_errors.keys())[0]
                if self.checkstyle_errors[first_category]:
                    style_error = random.choice(self.checkstyle_errors[first_category])
                    final_errors.append({
                        "type": "checkstyle",
                        "category": first_category,
                        "name": style_error["check_name"],
                        "description": style_error["description"]
                    })
        
        return final_errors
    
    def inject_errors(self, code: str, errors: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Inject selected errors into a Java code snippet using an LLM.
        
        Args:
            code: Original Java code snippet
            errors: List of error dictionaries to inject
            
        Returns:
            Tuple of (modified code, list of injected error descriptions)
        """
        if self.llm:
            return self._inject_errors_with_llm(code, errors)
        else:
            # Try Ollama API directly
            return self._inject_errors_with_ollama(code, errors)
    
    def _inject_errors_with_llm(self, code: str, errors: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Inject errors using a LangChain language model.
        
        Args:
            code: Original Java code snippet
            errors: List of error dictionaries to inject
            
        Returns:
            Tuple of (modified code, list of injected error descriptions)
        """
        if not self.llm:
            logger.warning("No LLM provided, falling back to Ollama API")
            return self._inject_errors_with_ollama(code, errors)
        
        # Create detailed instructions for the LLM
        prompt = self._create_error_injection_prompt(code, errors)
        
        try:
            # Generate the code with injected errors using the LLM
            logger.info(f"Injecting {len(errors)} errors with LLM")
            response = self.llm.invoke(prompt)
            
            # Extract the code and problems from the response
            code_with_errors, problem_descriptions = self._extract_code_and_problems(response)
            
            # Validation: Check if we got reasonable output
            if not code_with_errors or len(code_with_errors.strip()) < 50:
                logger.warning("LLM didn't return valid code with errors, falling back to Ollama API")
                return self._inject_errors_with_ollama(code, errors)
            
            # Ensure we have problem descriptions for each error
            if not problem_descriptions or len(problem_descriptions) < len(errors) // 2:
                # Generate problem descriptions if missing
                problem_descriptions = []
                for error in errors:
                    error_desc = self.format_error_description(error)
                    problem_descriptions.append(error_desc)
            
            return code_with_errors, problem_descriptions
            
        except Exception as e:
            logger.error(f"Error injecting errors with LLM: {str(e)}")
            return self._inject_errors_with_ollama(code, errors)
    
    def _inject_errors_with_ollama(self, code: str, errors: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Inject errors using direct Ollama API call.
        
        Args:
            code: Original Java code snippet
            errors: List of error dictionaries to inject
            
        Returns:
            Tuple of (modified code, list of injected error descriptions)
        """
        try:
            # Create prompt for Ollama
            prompt = self._create_error_injection_prompt(code, errors)
            
            # Get model from environment or use default
            model = os.getenv("GENERATIVE_MODEL", "llama3:7b")
            
            # Send request to Ollama
            url = f"{self.ollama_base_url}/api/generate"
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            logger.info(f"Injecting {len(errors)} errors with Ollama model: {model}")
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Extract code and problems
                code_with_errors, problem_descriptions = self._extract_code_and_problems(response_text)
                
                # Validation: Check if we got reasonable output
                if not code_with_errors or len(code_with_errors.strip()) < 50:
                    logger.warning("Ollama didn't return valid code, falling back to programmatic injection")
                    return self._inject_errors_programmatically(code, errors)
                
                # Ensure we have problem descriptions for each error
                if not problem_descriptions or len(problem_descriptions) < len(errors) // 2:
                    # Generate problem descriptions if missing
                    problem_descriptions = []
                    for error in errors:
                        error_desc = self.format_error_description(error)
                        problem_descriptions.append(error_desc)
                
                return code_with_errors, problem_descriptions
            else:
                logger.error(f"Ollama request failed with status code {response.status_code}")
                return self._inject_errors_programmatically(code, errors)
                
        except Exception as e:
            logger.error(f"Error injecting errors with Ollama: {str(e)}")
            return self._inject_errors_programmatically(code, errors)
    
    def _create_error_injection_prompt(self, code: str, errors: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for injecting errors into code.
        
        Args:
            code: Original Java code
            errors: List of error dictionaries
            
        Returns:
            Prompt for the LLM
        """
        # Create error injection instructions
        error_instructions = ""
        for i, error in enumerate(errors, 1):
            error_type = error["type"]
            category = error["category"]
            name = error["name"]
            description = error["description"]
            
            error_instructions += f"{i}. {error_type.upper()} ERROR - {name}\n"
            error_instructions += f"   Category: {category}\n"
            error_instructions += f"   Description: {description}\n"
            
            # Add specific implementation suggestion based on error type and name
            if error_type == "build":
                if "Cannot find symbol" in name:
                    error_instructions += f"   Suggestion: Use a variable, method, or class that hasn't been defined or imported\n"
                elif "NullPointer" in name:
                    error_instructions += f"   Suggestion: Create a scenario where a null object is accessed without proper null check\n"
                elif "Incompatible types" in name or "Type mismatch" in name:
                    error_instructions += f"   Suggestion: Assign a value to a variable of an incompatible type\n"
                elif "Missing return" in name:
                    error_instructions += f"   Suggestion: Remove the return statement from a non-void method\n"
                elif "Unreported exception" in name:
                    error_instructions += f"   Suggestion: Throw a checked exception without a try-catch or throws declaration\n"
                elif "Class not found" in name or "Package does not exist" in name:
                    error_instructions += f"   Suggestion: Import a non-existent class or package\n"
                elif "ArrayIndexOutOfBounds" in name or "IndexOutOfBounds" in name:
                    error_instructions += f"   Suggestion: Access an array or list with an invalid index\n"
                else:
                    error_instructions += f"   Suggestion: Implement this error in a way that matches its description\n"
            else:  # checkstyle
                if "Naming" in name or "Name" in name:
                    error_instructions += f"   Suggestion: Use inappropriate naming convention for a class, method, or variable\n"
                elif "Whitespace" in name or "Indentation" in name:
                    error_instructions += f"   Suggestion: Use inconsistent or incorrect whitespace/indentation\n"
                elif "Javadoc" in name or "Comment" in name:
                    error_instructions += f"   Suggestion: Create missing or improperly formatted Javadoc/comments\n"
                elif "Braces" in name or "Curly" in name or "LeftCurly" in name or "RightCurly" in name:
                    error_instructions += f"   Suggestion: Place curly braces inconsistently or incorrectly\n"
                elif "Import" in name:
                    error_instructions += f"   Suggestion: Create import-related issues like unused imports or star imports\n"
                elif "Empty" in name:
                    error_instructions += f"   Suggestion: Create an empty block or statement without proper comments\n"
                elif "Magic" in name:
                    error_instructions += f"   Suggestion: Use magic numbers instead of named constants\n"
                else:
                    error_instructions += f"   Suggestion: Implement this style violation naturally\n"
                
            error_instructions += "\n"
        
        # Create the full prompt
        prompt = f"""You are an expert Java programming educator who creates code review exercises with intentional errors.

Please modify the following Java code snippet to introduce specific errors:

{error_instructions}

Original Java code without errors:
```java
{code}
```

For each error you introduce:
1. Make sure the error matches the exact description provided
2. Place each error at a logical location in the code
3. Ensure the error is recognizable to a student with beginner to intermediate Java knowledge
4. Add brief comments nearby (using // Comment format) that hint at the error without directly stating it

Return your response in this specific JSON format:
```json
{{
  "code_snippet": "// Your Java code with intentional errors here",
  "problems": [
    "Description of problem 1 with its exact location in the code",
    "Description of problem 2 with its exact location in the code",
    "Description of problem 3 with its exact location in the code"
  ]
}}
```

Remember that the code should appear realistic, as if it was written by a real developer who made these specific mistakes.
"""
        
        return prompt
    
    def _extract_code_and_problems(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract code snippet and problems from LLM response.
        
        Args:
            text: Response text from LLM
            
        Returns:
            Tuple of (code snippet, list of problems)
        """
        # Try to extract JSON data from the response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text) or re.search(r'({[\s\S]*"code_snippet"[\s\S]*"problems"[\s\S]*})', text)
    
        if json_match:
            try:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                
                if "code_snippet" in data and "problems" in data:
                    return data["code_snippet"], data["problems"]
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract code block and problem list separately
        code_snippet = ""
        problems = []
        
        # Extract code block
        code_block_matches = re.findall(r'```(?:java)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_block_matches:
            # Use the largest code block
            code_snippet = max(code_block_matches, key=len)
        
        # Try to find problems section in various formats
        problem_patterns = [
            r'(?:Problems|Issues|Errors):\s*((?:\d+\.\s*.*?(?:\n|$))+)',  # Numbered list with header
            r'(?:problems|issues|errors):\s*((?:[-*•]\s*.*?(?:\n|$))+)',  # Bullet list with header
            r'(?:^|\n)(\d+\.\s*.*?)(?=\n\d+\.|\n\n|\Z)',  # Numbered list items
            r'(?:^|\n)([-*•]\s*.*?)(?=\n[-*•]|\n\n|\Z)',  # Bullet list items
        ]
        
        for pattern in problem_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    # For patterns with headers, extract individual items
                    if ":" in pattern:
                        if r'\d+' in pattern:  # Numbered list
                            items = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|\Z)', match, re.DOTALL)
                        else:  # Bullet list
                            items = re.findall(r'[-*•]\s*(.*?)(?=\n[-*•]|\n\n|\Z)', match, re.DOTALL)
                        
                        for item in items:
                            if item.strip():
                                problems.append(item.strip())
                    else:
                        # Remove the numbering/bullet
                        problem = re.sub(r'^\d+\.\s*|^[-*•]\s*', '', match)
                        if problem.strip():
                            problems.append(problem.strip())
                if problems:
                    break
        
        return code_snippet, problems
    
    def format_error_description(self, error: Dict[str, Any]) -> str:
        """
        Format an error dictionary into a readable description.
        
        Args:
            error: Error dictionary with type, category, name, and description
            
        Returns:
            Formatted error description
        """
        error_type = error["type"].title()
        category = error["category"]
        name = error["name"]
        description = error["description"]
        
        return f"{error_type} Error - {name}: {description} (Category: {category})"
    
    def _inject_errors_programmatically(self, code: str, errors: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """
        Inject errors programmatically as a fallback method.
        
        Args:
            code: Original Java code snippet
            errors: List of error dictionaries to inject
            
        Returns:
            Tuple of (modified code, list of injected error descriptions)
        """
        # This is a simplified fallback method that adds error indicators as comments
        lines = code.split("\n")
        injected_descriptions = []
        
        for i, error in enumerate(errors):
            error_type = error["type"]
            name = error["name"]
            description = error["description"]
            
            # For each error, add an indicator comment at a reasonable position in the code
            position = min(5 + i * 3, len(lines) - 1)
            error_comment = f"// ERROR: {name} - {description}"
            
            # Insert the error comment
            lines.insert(position, error_comment)
            
            # Generate a description with location
            line_number = position + 1  # 1-based line number
            injected_descriptions.append(f"{name} at line {line_number}: {description}")
        
        return "\n".join(lines), injected_descriptions