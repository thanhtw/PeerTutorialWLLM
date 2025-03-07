"""
Agent Service module for Java Peer Review Training System.

This module provides the AgentService class which coordinates between
the UI, domain objects, and LLM manager to execute the code review workflow.
"""

import logging
import random
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Import domain classes
from core.code_generator import CodeGenerator
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
        self.evaluator = StudentResponseEvaluator()
        self.feedback_manager = FeedbackManager(self.evaluator)
        
        # Load error data from JSON files
        self.build_errors = self._load_json_data("build_errors.json")
        self.checkstyle_errors = self._load_json_data("checkstyle_error.json")
        
        # Check if Ollama is available and initialize models
        self._initialize_models()
    
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
        Generate Java code with intentional errors directly selected from JSON files.
        
        Args:
            code_length: Length of code (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            selected_error_categories: Dictionary with 'build' and 'checkstyle' keys,
                                     each containing a list of selected categories
            
        Returns:
            Dictionary with generated code, known problems, and error information
        """
        try:
            # Use default categories if none specified
            if selected_error_categories is None or not any(selected_error_categories.values()):
                selected_error_categories = {
                    "build": ["CompileTimeErrors", "RuntimeErrors", "LogicalErrors"],
                    "checkstyle": ["NamingConventionChecks", "WhitespaceAndFormattingChecks", "JavadocChecks"]
                }
            
            # Directly select specific errors from the JSON files
            selected_errors = []
            problem_descriptions = []
            
            # Determine number of errors based on difficulty
            total_errors = {
                "easy": 2,
                "medium": 4,
                "hard": 6
            }.get(difficulty_level.lower(), 3)
            
            # Calculate number of errors for each type
            build_categories = selected_error_categories.get("build", [])
            checkstyle_categories = selected_error_categories.get("checkstyle", [])
            
            total_categories = len(build_categories) + len(checkstyle_categories)
            
            if total_categories > 0:
                build_proportion = len(build_categories) / total_categories
                build_count = max(1, round(total_errors * build_proportion)) if build_categories else 0
                checkstyle_count = max(1, total_errors - build_count) if checkstyle_categories else 0
            else:
                build_count = total_errors // 2
                checkstyle_count = total_errors - build_count
            
            # Select build errors
            for category in build_categories:
                if category in self.build_errors and len(selected_errors) < build_count:
                    # Randomly select an error from this category
                    error = random.choice(self.build_errors[category])
                    selected_errors.append({
                        "type": "build",
                        "category": category,
                        "name": error["error_name"],
                        "description": error["description"]
                    })
                    problem_descriptions.append(f"Build Error - {error['error_name']}: {error['description']} (Category: {category})")
            
            # If we need more build errors, select from other categories
            while len([e for e in selected_errors if e["type"] == "build"]) < build_count and build_categories:
                category = random.choice(build_categories)
                if category in self.build_errors and self.build_errors[category]:
                    error = random.choice(self.build_errors[category])
                    # Check if we already selected this error
                    if not any(e["name"] == error["error_name"] and e["category"] == category for e in selected_errors):
                        selected_errors.append({
                            "type": "build",
                            "category": category,
                            "name": error["error_name"],
                            "description": error["description"]
                        })
                        problem_descriptions.append(f"Build Error - {error['error_name']}: {error['description']} (Category: {category})")
            
            # Select checkstyle errors
            for category in checkstyle_categories:
                if category in self.checkstyle_errors and len([e for e in selected_errors if e["type"] == "checkstyle"]) < checkstyle_count:
                    # Randomly select an error from this category
                    error = random.choice(self.checkstyle_errors[category])
                    selected_errors.append({
                        "type": "checkstyle",
                        "category": category,
                        "name": error["check_name"],
                        "description": error["description"]
                    })
                    problem_descriptions.append(f"Checkstyle Error - {error['check_name']}: {error['description']} (Category: {category})")
            
            # If we need more checkstyle errors, select from other categories
            while len([e for e in selected_errors if e["type"] == "checkstyle"]) < checkstyle_count and checkstyle_categories:
                category = random.choice(checkstyle_categories)
                if category in self.checkstyle_errors and self.checkstyle_errors[category]:
                    error = random.choice(self.checkstyle_errors[category])
                    # Check if we already selected this error
                    if not any(e["name"] == error["check_name"] and e["category"] == category for e in selected_errors):
                        selected_errors.append({
                            "type": "checkstyle",
                            "category": category,
                            "name": error["check_name"],
                            "description": error["description"]
                        })
                        problem_descriptions.append(f"Checkstyle Error - {error['check_name']}: {error['description']} (Category: {category})")
            
            # Generate code with errors using LLM
            code_with_errors = self._generate_code_with_selected_errors(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=selected_errors
            )
            
            # Start a new feedback session
            self.feedback_manager.start_new_review_session(
                code_snippet=code_with_errors,
                known_problems=problem_descriptions
            )
            
            return {
                "code_snippet": code_with_errors,
                "known_problems": problem_descriptions,
                "raw_errors": selected_errors
            }
            
        except Exception as e:
            logger.error(f"Error generating code with errors: {str(e)}")
            return {
                "error": f"Error generating code: {str(e)}"
            }
    
    def _generate_code_with_selected_errors(self,
                                           code_length: str,
                                           difficulty_level: str,
                                           selected_errors: List[Dict[str, Any]]) -> str:
        """
        Generate Java code with selected errors using the LLM in one step.
        
        Args:
            code_length: Length of code (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            selected_errors: List of selected errors to include in the code
            
        Returns:
            Java code with errors
        """
        try:
            # If we have a language model, use it to directly generate code with errors
            if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
                # Create a detailed prompt for the LLM to generate code with specific errors
                prompt = self._create_direct_code_generation_prompt(
                    code_length=code_length,
                    difficulty_level=difficulty_level,
                    selected_errors=selected_errors
                )
                
                # Generate code with errors directly using the LLM
                logger.info(f"Directly generating code with {len(selected_errors)} errors")
                response = self.code_generator.llm.invoke(prompt)
                
                # Extract code from the response
                code_with_errors = self._extract_code_from_response(response)
                
                # If we got valid code, return it
                if code_with_errors and len(code_with_errors.strip()) > 50:
                    return code_with_errors
            
            # Fallback: generate clean code and add error comments
            logger.warning("Couldn't directly generate code with errors, using fallback")
            base_code = self.code_generator.generate_java_code(
                code_length=code_length,
                difficulty_level=difficulty_level,
                domain="student_management"
            )
            return self._add_error_comments(base_code, selected_errors)
            
        except Exception as e:
            logger.error(f"Error generating code with errors: {str(e)}")
            # Fallback: generate clean code and add error comments
            base_code = self.code_generator.generate_java_code(
                code_length=code_length,
                difficulty_level=difficulty_level
            )
            return self._add_error_comments(base_code, selected_errors)
    
    def _create_direct_code_generation_prompt(self, code_length: str, difficulty_level: str, selected_errors: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for directly generating Java code with specific errors.
        
        Args:
            code_length: Length of code (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            selected_errors: List of error dictionaries to include
            
        Returns:
            Prompt for the LLM
        """
        # Create domain context based on the errors
        domains = ["student_management", "file_processing", "data_validation", "calculation", "inventory_system"]
        domain = random.choice(domains)
        
        # Create complexity profile based on code length
        complexity_profile = {
            "short": "1 class with 2-4 methods and fields",
            "medium": "1 class with 4-6 methods, may include nested classes",
            "long": "2-3 classes with 5-10 methods and proper class relationships"
        }.get(code_length, "1 class with 4-6 methods")
        
        # Create error instructions
        error_instructions = ""
        for i, error in enumerate(selected_errors, 1):
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
        

         # Check if reasoning mode is enabled
        reasoning_mode = os.getenv("REASONING_MODE", "false").lower() == "true"
        # Create the full prompt
        if reasoning_mode:
            prompt = f"""You are an expert Java programming educator who creates code review exercises with intentional errors.

                        Please create a {code_length} Java code example for a {domain} system with {complexity_profile}.
                        The code should be realistic, well-structured, and include the following specific errors:

                        {error_instructions}

                        Let's think through this step by step:

                        1. First, I'll design the overall structure of the Java application for a {domain} system.
                        2. Then, I'll identify where each error should be placed to create a realistic learning scenario.
                        3. Next, I'll implement the code with these intentional errors in a way that maintains realism.
                        4. Finally, I'll review the code to ensure all required errors are present and the code is otherwise valid.

                        Requirements:
                        1. Write a complete, compilable Java code (except for the intentional errors)
                        2. Make the code realistic and representative of actual Java applications
                        3. For each error you include:
                        - Make sure it exactly matches the description provided
                        - Place it at a logical location in the code
                        - Ensure it's recognizable to a student with beginner to intermediate Java knowledge
                        - Add brief comments nearby (using // Comment format) that hint at the error without directly stating it
                        4. The difficulty level should be {difficulty_level}, appropriate for students learning Java

                        I'll now create the Java code with the required errors: """

        else:      
        # Create the full prompt
            prompt = f"""You are an expert Java programming educator who creates code review exercises with intentional errors.

                Please create a {code_length} Java code example for a {domain} system with {complexity_profile}.
                The code should be realistic, well-structured, and include the following specific errors:

                {error_instructions}

                Requirements:
                1. Write a complete, compilable Java code (except for the intentional errors)
                2. Make the code realistic and representative of actual Java applications
                3. For each error you include:
                - Make sure it exactly matches the description provided
                - Place it at a logical location in the code
                - Ensure it's recognizable to a student with beginner to intermediate Java knowledge
                - Add brief comments nearby (using // Comment format) that hint at the error without directly stating it
                4. The difficulty level should be {difficulty_level}, appropriate for students learning Java

                Return ONLY the Java code with the errors included. Do not include any explanations or JSON formatting.
                """
                
        return prompt
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Java code from LLM response.
        
        Args:
            response: Full response from the LLM
            
        Returns:
            Extracted Java code
        """
        import re
        
        # Try to extract code from code blocks
        code_blocks = re.findall(r'```(?:java)?\s*(.*?)\s*```', response, re.DOTALL)
        
        if code_blocks:
            # Return the largest code block
            return max(code_blocks, key=len)
        
        # If no code blocks are found, assume the entire response is code
        return response.strip()
    
    def _add_error_comments(self, code: str, errors: List[Dict[str, Any]]) -> str:
        """
        Add error comments to code as a fallback method.
        
        Args:
            code: Original Java code
            errors: List of error dictionaries
            
        Returns:
            Code with error comments
        """
        lines = code.split('\n')
        
        # Add comments about each error at reasonable positions in the code
        for i, error in enumerate(errors):
            error_type = error["type"]
            name = error["name"]
            description = error["description"]
            
            # Determine a reasonable position in the code for this error
            position = min(5 + i * 3, len(lines) - 1)
            
            # Create an error comment
            comment = f"// TODO: Fix {error_type} error: {name} - {description}"
            
            # Insert the comment at the determined position
            lines.insert(position, comment)
        
        return '\n'.join(lines)
    
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
        
        # Add additional sections from the original implementation...
        
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