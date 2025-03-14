"""
LangGraph Workflow for Java Peer Review Training System.

This module implements the code review workflow as a LangGraph graph.
"""

__all__ = ['JavaCodeReviewGraph']

import logging
import os
import random
import re
from typing import Dict, List, Any, Annotated, TypedDict, Tuple, cast, Optional
from langgraph.graph import StateGraph, END
from state_schema import WorkflowState, CodeSnippet, ReviewAttempt

# Import domain-specific components
from core.code_generator import CodeGenerator
from core.student_response_evaluator import StudentResponseEvaluator
from core.feedback_manager import FeedbackManager
from data.json_error_repository import JsonErrorRepository
from llm_manager import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JavaCodeReviewGraph:
    """
    LangGraph implementation of the Java Code Review workflow.
    """
    
    def __init__(self, llm_manager: LLMManager = None):
        """Initialize the graph with domain components."""
        # Initialize LLM Manager if not provided
        self.llm_manager = llm_manager or LLMManager()
        
        # Initialize repositories
        self.error_repository = JsonErrorRepository()
        
        # Initialize domain objects
        self._initialize_domain_objects()
        
        # Create the graph
        self.workflow = self._build_graph()
    
    def _initialize_domain_objects(self):
        """Initialize domain objects with LLMs if available."""
        # Check Ollama connection
        connection_status, _ = self.llm_manager.check_ollama_connection()
        
        if connection_status:
            # Initialize models for each component
            generative_model = self.llm_manager.initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
            review_model = self.llm_manager.initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
            summary_model = self.llm_manager.initialize_model_from_env("SUMMARY_MODEL", "SUMMARY_TEMPERATURE")
            compare_model = self.llm_manager.initialize_model_from_env("COMPARE_MODEL", "COMPARE_TEMPERATURE")
            
            # Initialize domain objects with models
            self.code_generator = CodeGenerator(generative_model) if generative_model else CodeGenerator()
            self.evaluator = StudentResponseEvaluator(review_model) if review_model else StudentResponseEvaluator()
            self.feedback_manager = FeedbackManager(self.evaluator)
        else:
            # Initialize domain objects without LLMs
            self.code_generator = CodeGenerator()
            self.evaluator = StudentResponseEvaluator()
            self.feedback_manager = FeedbackManager(self.evaluator)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create a new graph with our state schema
        workflow = StateGraph(WorkflowState)
        
        # Define nodes
        workflow.add_node("generate_code", self.generate_code_node)
        workflow.add_node("review_code", self.review_code_node)
        workflow.add_node("analyze_review", self.analyze_review_node)
        workflow.add_node("generate_summary", self.generate_summary_node)
        
        # Define edges
        workflow.add_edge("generate_code", "review_code")
        workflow.add_edge("review_code", "analyze_review")
        
        # Conditional edges from analyze_review
        workflow.add_conditional_edges(
            "analyze_review",
            self.should_continue_review,
            {
                "continue_review": "review_code",
                "generate_summary": "generate_summary"
            }
        )
        
        workflow.add_edge("generate_summary", END)
        
        # Set the entry point
        workflow.set_entry_point("generate_code")
        
        return workflow
    
    # Node implementations
    def generate_code_node(self, state: WorkflowState) -> WorkflowState:
        """Generate Java code with errors node."""
        try:
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            selected_error_categories = state.selected_error_categories
            
            # Generate code with errors
            # Get errors from selected categories
            selected_errors, problem_descriptions = self.error_repository.get_errors_for_llm(
                selected_categories=selected_error_categories,
                count=self._get_error_count_for_difficulty(difficulty_level),
                difficulty=difficulty_level
            )
                        
            # Generate code with selected errors
            code_with_errors = self._generate_code_with_errors(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=selected_errors
            )
            
            # Create code snippet object
            code_snippet = CodeSnippet(
                code=code_with_errors,
                known_problems=problem_descriptions,
                raw_errors={
                    "build": [e for e in selected_errors if e["type"] == "build"],
                    "checkstyle": [e for e in selected_errors if e["type"] == "checkstyle"]
                }
            )
            
            # Update state
            state.code_snippet = code_snippet
            state.current_step = "review"
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            state.error = f"Error generating code: {str(e)}"
            return state
    
    def generate_code_with_specific_errors(self, state: WorkflowState, specific_errors: List[Dict[str, Any]]) -> WorkflowState:
        """Generate Java code with specific errors selected by the user."""
        try:
            # Get parameters from state
            code_length = state.code_length
            difficulty_level = state.difficulty_level
            
            # Format problem descriptions
            problem_descriptions = []
            for error in specific_errors:
                error_type = error.get("type", "Unknown")
                name = error.get("name", "Unknown")
                description = error.get("description", "")
                category = error.get("category", "")
                
                if error_type == "build":
                    problem_descriptions.append(f"Build Error - {name}: {description} (Category: {category})")
                else:  # checkstyle
                    problem_descriptions.append(f"Checkstyle Error - {name}: {description} (Category: {category})")
            
            # Generate code with selected errors
            code_with_errors = self._generate_code_with_errors(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=specific_errors
            )
            
            # Create code snippet object
            code_snippet = CodeSnippet(
                code=code_with_errors,
                known_problems=problem_descriptions,
                raw_errors={
                    "build": [e for e in specific_errors if e["type"] == "build"],
                    "checkstyle": [e for e in specific_errors if e["type"] == "checkstyle"]
                }
            )
            
            # Update state
            state.code_snippet = code_snippet
            state.current_step = "review"
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating code with specific errors: {str(e)}")
            state.error = f"Error generating code with specific errors: {str(e)}"
            return state
    
    def review_code_node(self, state: WorkflowState) -> WorkflowState:
        """Review code node - this is a placeholder since user input happens in the UI."""
        # This node is primarily a placeholder since the actual review is submitted via the UI
        state.current_step = "review"
        return state
    
    def analyze_review_node(self, state: WorkflowState) -> WorkflowState:
        """Analyze student review node."""
        try:
            # Get the latest review
            if not state.review_history:
                state.error = "No review submitted to analyze"
                return state
                
            latest_review = state.review_history[-1]
            student_review = latest_review.student_review
            
            # Get code snippet
            if not state.code_snippet:
                state.error = "No code snippet available"
                return state
                
            code_snippet = state.code_snippet.code
            known_problems = state.code_snippet.known_problems
            
            # Analyze the student review
            analysis = self.evaluator.evaluate_review(
                code_snippet=code_snippet,
                known_problems=known_problems,
                student_review=student_review
            )
            
            # Update the review with analysis
            latest_review.analysis = analysis
            
            # Check if the review is sufficient
            review_sufficient = analysis.get("review_sufficient", False)
            state.review_sufficient = review_sufficient
            
            # Generate targeted guidance if needed
            if not review_sufficient and state.current_iteration < state.max_iterations:
                targeted_guidance = self.evaluator.generate_targeted_guidance(
                    code_snippet=code_snippet,
                    known_problems=known_problems,
                    student_review=student_review,
                    review_analysis=analysis,
                    iteration_count=state.current_iteration,
                    max_iterations=state.max_iterations
                )
                latest_review.targeted_guidance = targeted_guidance
            
            # Increment iteration count
            state.current_iteration += 1
            
            # Update state
            state.current_step = "analyze"
            
            return state
            
        except Exception as e:
            logger.error(f"Error analyzing review: {str(e)}")
            state.error = f"Error analyzing review: {str(e)}"
            return state
    
    def generate_summary_node(self, state: WorkflowState) -> WorkflowState:
        """Generate summary and comparison report node."""
        try:
            # Generate final feedback
            if not state.review_history:
                state.error = "No reviews available for summary"
                return state
                    
            # Generate feedback using the feedback manager
            all_reviews = [review.student_review for review in state.review_history]
            all_analyses = [review.analysis for review in state.review_history]
            
            # Initialize the feedback manager with the code snippet and known problems
            self.feedback_manager.start_new_review_session(
                code_snippet=state.code_snippet.code,
                known_problems=state.code_snippet.known_problems
            )
            
            # Load all reviews into the feedback manager
            for i, review_text in enumerate(all_reviews):
                if i < len(all_reviews) - 1:  # Skip the last one as we'll process it differently
                    self.feedback_manager.submit_review(review_text)
            
            # Submit the last review to get final analysis
            if all_reviews:
                self.feedback_manager.submit_review(all_reviews[-1])
            
            # Generate final feedback and comparison report
            final_feedback = self.feedback_manager.generate_final_feedback()
            
            # Safe version of comparison report generation
            comparison_report = self._generate_comparison_report(
                state.code_snippet.known_problems,
                state.review_history[-1].analysis if state.review_history else {}
            )
            
            # Update state
            state.review_summary = final_feedback
            state.comparison_report = comparison_report
            state.current_step = "complete"
            
            return state
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            state.error = f"Error generating summary: {str(e)}"
            return state
    
    # Conditional edge implementations
    def should_continue_review(self, state: WorkflowState) -> str:
        """
        Determine if we should continue with another review iteration or generate summary.
        """
        # Check if we've reached max iterations
        if state.current_iteration > state.max_iterations:
            return "generate_summary"
        
        # Check if the review is sufficient
        if state.review_sufficient:
            return "generate_summary"
        
        # Otherwise, continue reviewing
        return "continue_review"
    
    # Helper methods
    def _generate_code_with_errors(self, code_length: str, difficulty_level: str, selected_errors: List[Dict[str, Any]]) -> str:
        """Generate Java code with the selected errors."""
        # Use the code generator to create code with errors
        if hasattr(self.code_generator, 'llm') and self.code_generator.llm:
            # Create a detailed prompt for the LLM
            prompt = self._create_code_generation_prompt(
                code_length=code_length,
                difficulty_level=difficulty_level,
                selected_errors=selected_errors
            )
            
            # Generate code with errors
            response = self.code_generator.llm.invoke(prompt)
            code = self._extract_code_from_response(response)
            
            if code and len(code.strip()) > 50:
                return code
        
        # Fallback: generate clean code and manually note errors
        base_code = self.code_generator.generate_java_code(
            code_length=code_length,
            difficulty_level=difficulty_level
        )

        return self._add_error_comments(base_code, selected_errors)
    
    def _create_code_generation_prompt(self, code_length: str, difficulty_level: str, selected_errors: List[Dict[str, Any]]) -> str:
        """Create a prompt for generating code with specific errors."""
        # Create domain context based on the errors
        domains = ["student_management", "file_processing", "data_validation", "calculation", "inventory_system"]
        domain = random.choice(domains)
        
        # Ensure parameters are strings
        code_length_str = str(code_length) if not isinstance(code_length, str) else code_length
        difficulty_level_str = str(difficulty_level) if not isinstance(difficulty_level, str) else difficulty_level
        
        # Create complexity profile based on code length
        complexity_profile = {
            "short": "1 class with 2-4 methods and fields",
            "medium": "1 class with 4-6 methods, may include nested classes",
            "long": "2-3 classes with 5-10 methods and proper class relationships"
        }.get(code_length_str.lower(), "1 class with 4-6 methods")
        
        # Create error instructions using implementation guides from JSON
        error_instructions = ""
        for i, error in enumerate(selected_errors, 1):
            error_type = error["type"]
            category = error.get("category", "")
            name = error["name"]
            description = error["description"]
            implementation_guide = error.get("implementation_guide", "Create this error in a way that matches its description")
            
            error_instructions += f"{i}. {error_type.upper()} ERROR - {name}\n"
            error_instructions += f"   Category: {category}\n"
            error_instructions += f"   Description: {description}\n"
            error_instructions += f"   Implementation: {implementation_guide}\n\n"
        
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
    - Add a comment with "ERROR TYPE: ERROR NAME" (e.g. "// BUILD ERROR: Missing return statement") directly above the line containing the error
    - Add brief details in the comment about what the error is and why it's problematic
    4. The difficulty level should be {difficulty_level}, appropriate for students learning Java
    5. Return your final code in a code block with ``` delimiters

    I'll now create the Java code with the required errors:
    """
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
    - Add a comment with "ERROR TYPE: ERROR NAME" (e.g. "// BUILD ERROR: Missing return statement") directly above the line containing the error
    - Add brief details in the comment about what the error is and why it's problematic
    4. The difficulty level should be {difficulty_level}, appropriate for students learning Java
    5. Return your final code in a code block with ``` delimiters

    Return ONLY the Java code with the errors included. Do not include any explanations or JSON formatting.
    """
        
        return prompt
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Java code from LLM response."""
        # Try to extract code from code blocks
        code_blocks = re.findall(r'```(?:java)?\s*(.*?)\s*```', response, re.DOTALL)
        
        if code_blocks:
            # Return the largest code block
            return max(code_blocks, key=len)
        
        # If no code blocks are found, assume the entire response is code
        return response.strip()
    
    def _add_error_comments(self, code: str, errors: List[Dict[str, Any]]) -> str:
        """Add error comments to code as fallback."""
        lines = code.split('\n')
        
        for i, error in enumerate(errors):
            error_type = error.get("type", "unknown")
            name = error.get("name", "unknown error")
            description = error.get("description", "")
            
            position = min(5 + i * 3, len(lines) - 1)
            comment = f"// TODO: Fix {error_type} error: {name} - {description}"
            
            lines.insert(position, comment)
        
        return '\n'.join(lines)
    
    def _generate_comparison_report(self, known_problems: List[str], review_analysis: Dict[str, Any]) -> str:
        """Generate a comparison report between student review and known problems."""
        # Create report header
        report = "# Detailed Comparison: Your Review vs. Actual Issues\n\n"
        
        # New section explaining the comparison method
        report += "## How Reviews Are Compared\n\n"
        report += "Your review is compared to the known problems in the code using a semantic matching approach. "
        report += "This means we look for whether you've identified the key aspects of each issue, rather than requiring exact matching phrases.\n\n"
        
        report += "For each issue, the system checks if your comments include:\n"
        report += "1. **The correct location** of the error (line numbers, method names, etc.)\n"
        report += "2. **The appropriate error type or category** (e.g., NullPointerException, naming convention)\n"
        report += "3. **A clear explanation** of why it's problematic\n\n"
        
        report += "A problem is considered 'identified' if you correctly mentioned its key aspects. "
        report += "Partial credit may be given for partially identified issues. "
        report += "False positives are issues you reported that don't match any actual problems in the code.\n\n"
        
        # Problems section
        report += "## Code Issues Analysis\n\n"
        
        # Safely extract data from review analysis
        identified_problems = review_analysis.get("identified_problems", [])
        missed_problems = review_analysis.get("missed_problems", [])
        false_positives = review_analysis.get("false_positives", [])
        
        # Ensure all problems are properly converted to strings
        known_problems_str = [str(p) if not isinstance(p, str) else p for p in known_problems]
        identified_problems_str = [str(p) if not isinstance(p, str) else p for p in identified_problems]
        missed_problems_str = [str(p) if not isinstance(p, str) else p for p in missed_problems]
        false_positives_str = [str(p) if not isinstance(p, str) else p for p in false_positives]
        
        # Issues found correctly
        if identified_problems_str:
            report += "### Issues You Identified Correctly\n\n"
            for i, problem in enumerate(identified_problems_str, 1):
                report += f"**{i}. {problem}**\n\n"
                report += "Great job finding this issue! "
                report += "This demonstrates your understanding of this type of problem.\n\n"
        
        # Issues missed with detailed guidance
        if missed_problems_str:
            report += "### Issues You Missed\n\n"
            for i, problem in enumerate(missed_problems_str, 1):
                report += f"**{i}. {problem}**\n\n"
                
                # Enhanced guidance with example comment format
                problem_lower = problem.lower()
                report += "**How to identify this issue:**\n\n"
                
                if "null" in problem_lower or "nullpointer" in problem_lower:
                    report += "When reviewing Java code, look for variables that might be null before being accessed. "
                    report += "Check for null checks before method calls or field access. Missing null checks often lead to NullPointerExceptions at runtime.\n\n"
                    report += "**Example comment format:**\n\n"
                    report += "`Line X: [NullPointerException Risk] - The variable 'name' is accessed without a null check, which could cause a runtime exception`\n\n"
                elif "naming" in problem_lower or "convention" in problem_lower:
                    report += "Check that class names use UpperCamelCase, while methods and variables use lowerCamelCase. "
                    report += "Constants should use UPPER_SNAKE_CASE. Consistent naming improves code readability and maintainability.\n\n"
                    report += "**Example comment format:**\n\n"
                    report += "`Line X: [Naming Convention] - The variable 'user_name' should use lowerCamelCase format (userName)`\n\n"
                elif "equal" in problem_lower or "==" in problem_lower:
                    report += "String and object comparisons should use the .equals() method instead of the == operator, which only compares references. "
                    report += "Using == for content comparison is a common error that can lead to unexpected behavior.\n\n"
                    report += "**Example comment format:**\n\n"
                    report += "`Line X: [Object Comparison] - String comparison uses == operator instead of .equals() method`\n\n"
                elif "array" in problem_lower or "index" in problem_lower:
                    report += "Always verify that array indices are within valid ranges before accessing elements. "
                    report += "Check for potential ArrayIndexOutOfBoundsException risks, especially in loops.\n\n"
                    report += "**Example comment format:**\n\n"
                    report += "`Line X: [Array Bounds] - Array access without bounds checking could cause ArrayIndexOutOfBoundsException`\n\n"
                elif "whitespace" in problem_lower or "indent" in problem_lower:
                    report += "Look for consistent indentation and proper whitespace around operators and keywords. "
                    report += "Proper formatting makes code more readable and maintainable.\n\n"
                    report += "**Example comment format:**\n\n"
                    report += "`Line X: [Formatting] - Inconsistent indentation makes the code hard to read`\n\n"
                else:
                    report += "When identifying issues, be specific about the location, type of error, and why it's problematic. "
                    report += "Include line numbers and detailed explanations in your comments.\n\n"
                    report += "**Example comment format:**\n\n"
                    report += "`Line X: [Error Type] - Description of the issue and why it's problematic`\n\n"
        
        # False positives
        if false_positives_str:
            report += "### Issues You Incorrectly Identified\n\n"
            for i, problem in enumerate(false_positives_str, 1):
                report += f"**{i}. {problem}**\n\n"
                report += "This wasn't actually an issue in the code. "
                report += "Be careful not to flag correct code as problematic.\n\n"
        
        # Calculate some metrics
        total_problems = len(known_problems_str)
        identified_count = len(identified_problems_str)
        missed_count = len(missed_problems_str)
        false_positive_count = len(false_positives_str)
        
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
        
        # Add improvement tips
        report += "## Tips for Improvement\n\n"
        
        if missed_problems_str:
            # Categories of missed issues
            missed_categories = []
            
            for problem in missed_problems_str:
                problem_lower = problem.lower()
                if "null" in problem_lower:
                    missed_categories.append("null pointer handling")
                elif "naming" in problem_lower or "convention" in problem_lower:
                    missed_categories.append("naming conventions")
                elif "javadoc" in problem_lower or "comment" in problem_lower:
                    missed_categories.append("documentation")
                elif "exception" in problem_lower or "throw" in problem_lower:
                    missed_categories.append("exception handling")
                elif "loop" in problem_lower or "condition" in problem_lower:
                    missed_categories.append("logical conditions")
                elif "whitespace" in problem_lower or "indentation" in problem_lower:
                    missed_categories.append("code formatting")
                elif "array" in problem_lower or "index" in problem_lower:
                    missed_categories.append("array handling")
            
            # Remove duplicates and sort
            missed_categories = sorted(set(missed_categories))
            
            if missed_categories:
                report += "Based on your review, focus on these areas in future code reviews:\n\n"
                for category in missed_categories:
                    report += f"- **{category.title()}**\n"
                report += "\n"
        
        # Add systematic approach suggestion
        report += """### Systematic Review Approach

For more thorough code reviews, try this systematic approach:

1. **First pass**: Check for syntax errors, compilation issues, and obvious bugs
2. **Second pass**: Examine naming conventions, code style, and documentation
3. **Third pass**: Analyze logical flow, edge cases, and potential runtime errors
4. **Final pass**: Look for performance issues, security concerns, and maintainability problems

By following a structured approach, you'll catch more issues and provide more comprehensive reviews.
"""
        
        # Add effective comment format
        report += """
### Effective Comment Format

When writing code review comments, use this format for clarity and consistency:

```
Line X: [Error Type] - Description of the issue and why it's problematic
```

For example:
```
Line 42: [NullPointerException Risk] - The 'user' variable could be null here, add a null check before calling methods
```

This format helps others quickly understand the location, type, and impact of each issue.
"""
        
        return report
    
    def _get_error_count_for_difficulty(self, difficulty: str) -> int:
        """Get appropriate error count based on difficulty level."""
        difficulty_map = {
            "easy": 2,
            "medium": 4,
            "hard": 6
        }
        return difficulty_map.get(str(difficulty).lower(), 4)
    
    # Public interface methods
    def get_all_error_categories(self) -> Dict[str, List[str]]:
        """Get all available error categories."""
        return self.error_repository.get_all_categories()
    
    def submit_review(self, state: WorkflowState, student_review: str) -> WorkflowState:
        """
        Submit a student review and update the state.
        This method is called from the UI when a student submits a review.
        """
        # Create a new review attempt
        review_attempt = ReviewAttempt(
            student_review=student_review,
            iteration_number=state.current_iteration,
            analysis={},
            targeted_guidance=None
        )
        
        # Add to review history
        state.review_history.append(review_attempt)
        
        # Run the state through the analyze_review node
        updated_state = self.analyze_review_node(state)
        
        return updated_state
    
    def reset(self) -> WorkflowState:
        """Reset the workflow to initial state."""
        return WorkflowState()