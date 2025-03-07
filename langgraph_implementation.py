"""
Java Peer Code Review Training System - LangGraph Implementation

This module implements the Java code review training system using LangGraph
for workflow orchestration and PydanticAI for type validation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Annotated, Union
from enum import Enum
from datetime import datetime

# Import pydantic for type validation
from pydantic import BaseModel, Field, model_validator

# Import LangGraph components with fallbacks for different versions
from langgraph.graph import StateGraph, END
# Remove ToolNode import as it's not used
# Remove LocalStateCheckpoint import as it might not be available

# Import LangChain components
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define states with pydantic models
class CodeReviewState(str, Enum):
    """States for the code review workflow"""
    GENERATE = "generate"
    REVIEW = "review"
    ANALYZE = "analyze"
    PROVIDE_GUIDANCE = "provide_guidance"
    SUMMARIZE = "summarize"
    COMPLETE = "complete"
    ERROR = "error"

class ErrorCategory(BaseModel):
    """Model for error categories"""
    type: str
    category: str
    name: str
    description: str

class ReviewAnalysis(BaseModel):
    """Model for review analysis results"""
    identified_problems: List[str] = Field(default_factory=list)
    missed_problems: List[str] = Field(default_factory=list)
    false_positives: List[str] = Field(default_factory=list)
    accuracy_percentage: float = 0.0
    identified_percentage: float = 0.0
    identified_count: int = 0
    total_problems: int = 0
    review_sufficient: bool = False
    feedback: str = ""

class ReviewIteration(BaseModel):
    """Model for a review iteration"""
    iteration_number: int
    student_review: str
    review_analysis: ReviewAnalysis
    targeted_guidance: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class CodeReviewGraph(BaseModel):
    """Main state model for the code review workflow"""
    current_state: CodeReviewState = CodeReviewState.GENERATE
    code_snippet: str = ""
    known_problems: List[str] = Field(default_factory=list)
    raw_errors: List[Dict[str, Any]] = Field(default_factory=list)  # Use Dict instead of ErrorCategory for flexibility
    student_review: str = ""
    iteration_count: int = 1
    max_iterations: int = 3
    review_history: List[Dict[str, Any]] = Field(default_factory=list)  # Use Dict instead of ReviewIteration for flexibility
    review_analysis: Optional[Dict[str, Any]] = None  # Use Dict instead of ReviewAnalysis for flexibility
    targeted_guidance: Optional[str] = None
    review_summary: Optional[str] = None
    comparison_report: Optional[str] = None
    error_message: Optional[str] = None

# LLM Manager
class LLMManager:
    """LLM Manager class for handling model initialization and access."""
    
    def __init__(self, initialize_models=True):
        """Initialize the LLM Manager with environment variables."""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = os.getenv("DEFAULT_MODEL", "llama3:1b")
        
        # Initialize models with lazy loading option
        self.generative_model = None
        self.review_model = None
        self.summary_model = None
        self.compare_model = None
        
        if initialize_models:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models"""
        print("Initializing LLM models")
        self.generative_model = self._initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
        self.review_model = self._initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
        self.summary_model = self._initialize_model_from_env("SUMMARY_MODEL", "SUMMARY_TEMPERATURE")
        self.compare_model = self._initialize_model_from_env("COMPARE_MODEL", "COMPARE_TEMPERATURE")
    
    def _initialize_model_from_env(self, model_env_key: str, temp_env_key: str):
        """Initialize a model from environment variables."""
        try:
            model_name = os.getenv(model_env_key, self.default_model)
            temperature = float(os.getenv(temp_env_key, "0.7"))
            
            print(f"Initializing model {model_name} with temperature {temperature}")
            return Ollama(
                base_url=self.ollama_base_url,
                model=model_name,
                temperature=temperature
            )
        except Exception as e:
            print(f"Error initializing model {model_env_key}: {str(e)}")
            # Return a fallback model that logs errors but doesn't crash
            return FallbackModel(model_env_key)

# Fallback model for error handling
class FallbackModel:
    """Fallback model that logs errors instead of crashing."""
    
    def __init__(self, model_name):
        self.model_name = model_name
    
    def invoke(self, messages):
        """Log the error and return a fallback message."""
        error_msg = f"Error: Model {self.model_name} is not available. Please make sure Ollama is running."
        logger.error(error_msg)
        return AIMessage(content=f"{error_msg} This is a fallback response.")

# Repositories
class ErrorRepository:
    """Repository for accessing Java error data from JSON files."""
    
    def __init__(self, build_errors_path: str = "build_errors.json",
                 checkstyle_errors_path: str = "checkstyle_error.json"):
        """Initialize the Error Repository."""
        print("Initializing ErrorRepository")
        self.build_errors_path = build_errors_path
        self.checkstyle_errors_path = checkstyle_errors_path
        
        # Load error data
        self.build_errors = self._load_json_data(build_errors_path)
        self.checkstyle_errors = self._load_json_data(checkstyle_errors_path)
        
        # Extract categories
        self.build_categories = list(self.build_errors.keys()) if self.build_errors else []
        self.checkstyle_categories = list(self.checkstyle_errors.keys()) if self.checkstyle_errors else []
        
        print(f"Loaded {len(self.build_categories)} build categories and {len(self.checkstyle_categories)} checkstyle categories")
    
    def _load_json_data(self, file_path: str) -> Dict:
        """Load JSON data from a file."""
        try:
            print(f"Attempting to load {file_path}")
            # Try different paths to find the file
            paths_to_try = [
                file_path,
                os.path.join(os.path.dirname(__file__), file_path),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), file_path),
            ]
            
            for path in paths_to_try:
                if os.path.exists(path):
                    print(f"Found {path}, loading...")
                    with open(path, "r") as f:
                        return json.load(f)
            
            logger.warning(f"Could not find error data file: {file_path}")
            print(f"Error data file not found: {file_path}")
            # Return a default minimal structure for testing
            return {
                "RuntimeErrors": [
                    {
                        "error_name": "NullPointerException",
                        "description": "Thrown when trying to access a method or field on a null object reference."
                    }
                ],
                "LogicalErrors": [
                    {
                        "error_name": "Incorrect loop condition",
                        "description": "Occurs when a loop condition is incorrect, leading to infinite loops or premature exits."
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error loading error data from {file_path}: {str(e)}")
            print(f"Error loading data: {str(e)}")
            # Return empty dict on error
            return {}
    
    def get_all_categories(self) -> Dict[str, List[str]]:
        """Get all error categories."""
        return {
            "build": self.build_categories,
            "checkstyle": self.checkstyle_categories
        }
    
    def get_random_errors(self, selected_categories: Dict[str, List[str]], count: int = 4) -> List[Dict[str, Any]]:
        """Get random errors from selected categories."""
        import random
        
        all_errors = []
        build_categories = selected_categories.get("build", [])
        checkstyle_categories = selected_categories.get("checkstyle", [])
        
        # Build errors
        for category in build_categories:
            if category in self.build_errors:
                for error in self.build_errors[category]:
                    all_errors.append({
                        "type": "build",
                        "category": category,
                        "name": error["error_name"],
                        "description": error["description"]
                    })
        
        # Checkstyle errors
        for category in checkstyle_categories:
            if category in self.checkstyle_errors:
                for error in self.checkstyle_errors[category]:
                    all_errors.append({
                        "type": "checkstyle",
                        "category": category,
                        "name": error["check_name"],
                        "description": error["description"]
                    })
        
        # Select random errors
        if all_errors:
            # If we have fewer errors than requested, return all
            if len(all_errors) <= count:
                return all_errors
            
            # Otherwise select random errors
            return random.sample(all_errors, count)
        
        # If no errors match the selected categories, return default errors for testing
        return [
            {
                "type": "build",
                "category": "RuntimeErrors",
                "name": "NullPointerException",
                "description": "Thrown when trying to access a method or field on a null object reference."
            },
            {
                "type": "build",
                "category": "LogicalErrors",
                "name": "Incorrect loop condition",
                "description": "Occurs when a loop condition is incorrect, leading to infinite loops or premature exits."
            }
        ]

# LangGraph Node Functions

def generate_code(state: CodeReviewGraph, llm_manager: LLMManager, error_repository: ErrorRepository) -> CodeReviewGraph:
    """Generate Java code with intentional errors."""
    try:
        print("Starting code generation")
        # Get selected error categories from the input or use defaults
        selected_categories = {
            "build": ["LogicalErrors", "RuntimeErrors"],
            "checkstyle": ["NamingConventionChecks", "JavadocChecks"]
        }
        
        # Determine number of errors based on difficulty
        difficulty_level = "medium"  # Default difficulty
        total_errors = {
            "easy": 2,
            "medium": 4,
            "hard": 6
        }.get(difficulty_level, 4)
        
        # Select random errors
        selected_errors = error_repository.get_random_errors(selected_categories, total_errors)
        print(f"Selected {len(selected_errors)} errors for code generation")
        
        # Create a prompt for code generation
        prompt = ChatPromptTemplate.from_template("""
You are an expert Java programming educator who creates code review exercises with intentional errors.

Please create a medium-sized Java code example for a student management system.
The code should be realistic, well-structured, and include the following specific errors:

{errors}

Requirements:
1. Write a complete, compilable Java code (except for the intentional errors)
2. Make the code realistic and representative of actual Java applications
3. For each error you include:
   - Make sure it exactly matches the description provided
   - Place it at a logical location in the code
   - Ensure it's recognizable to a student with beginner to intermediate Java knowledge
   - Add brief comments nearby (using // Comment format) that hint at the error without directly stating it
4. The difficulty level should be {difficulty}, appropriate for students learning Java

Return ONLY the Java code with the errors included. Do not include any explanations.
        """)
        
        # Format errors for the prompt
        formatted_errors = ""
        problem_descriptions = []
        
        for i, error in enumerate(selected_errors, 1):
            formatted_errors += f"{i}. {error['type'].upper()} ERROR - {error['name']}\n"
            formatted_errors += f"   Category: {error['category']}\n"
            formatted_errors += f"   Description: {error['description']}\n\n"
            
            problem_descriptions.append(f"{error['type'].capitalize()} Error - {error['name']}: {error['description']} (Category: {error['category']})")
        
        # Generate code with the LLM
        print("Sending code generation prompt to LLM")
        messages = prompt.format_messages(
            errors=formatted_errors,
            difficulty=difficulty_level
        )
        
        response = llm_manager.generative_model.invoke(messages)
        code_with_errors = response.content
        print(f"Received code generation response: {len(code_with_errors)} characters")
        
        # Update state with generated code
        new_state = CodeReviewGraph(
            code_snippet=code_with_errors,
            known_problems=problem_descriptions,
            raw_errors=selected_errors,
            current_state=CodeReviewState.REVIEW,
            iteration_count=1,
            max_iterations=state.max_iterations
        )
        
        print("Code generation completed successfully")
        return new_state
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating code: {str(e)}")
        print(f"Error in generate_code: {str(e)}")
        traceback.print_exc()
        
        # Update state with error
        new_state = CodeReviewGraph(
            current_state=CodeReviewState.ERROR,
            error_message=f"Error generating code: {str(e)}"
        )
        
        return new_state

def analyze_review(state: CodeReviewGraph, llm_manager: LLMManager) -> CodeReviewGraph:
    """Analyze a student's code review."""
    try:
        print("Starting review analysis")
        if not state.code_snippet or not state.known_problems:
            raise ValueError("No code snippet or known problems available for review")
        
        if not state.student_review:
            raise ValueError("No student review provided for analysis")
        
        # Create a prompt for review analysis
        prompt = ChatPromptTemplate.from_template("""
You are an expert code review analyzer. Analyze how well the student's review identifies the known problems in the code.

ORIGINAL CODE:
```java
{code_snippet}
```

KNOWN PROBLEMS IN THE CODE:
{known_problems}

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

A review is considered "sufficient" if the student correctly identified at least 60% of the known problems.
Be specific in your feedback about what types of issues they missed and how they can improve their code review skills.
        """)
        
        # Format known problems
        formatted_problems = "\n".join([f"- {problem}" for problem in state.known_problems])
        
        # Generate analysis with the LLM
        print("Sending review analysis prompt to LLM")
        messages = prompt.format_messages(
            code_snippet=state.code_snippet,
            known_problems=formatted_problems,
            student_review=state.student_review
        )
        
        response = llm_manager.review_model.invoke(messages)
        
        # Extract JSON from response
        analysis_text = response.content
        print(f"Received review analysis response: {len(analysis_text)} characters")
        analysis_data = extract_json_from_text(analysis_text)
        
        # Create ReviewAnalysis object
        review_analysis = {
            "identified_problems": analysis_data.get("identified_problems", []),
            "missed_problems": analysis_data.get("missed_problems", []),
            "false_positives": analysis_data.get("false_positives", []),
            "accuracy_percentage": float(analysis_data.get("accuracy_percentage", 0.0)),
            "review_sufficient": analysis_data.get("review_sufficient", False),
            "feedback": analysis_data.get("feedback", "")
        }
        
        # Calculate additional metrics
        identified_count = len(review_analysis["identified_problems"])
        total_problems = len(state.known_problems)
        
        review_analysis["identified_count"] = identified_count
        review_analysis["total_problems"] = total_problems
        
        if total_problems > 0:
            review_analysis["identified_percentage"] = (identified_count / total_problems) * 100
        
        # Create review iteration record
        review_iteration = {
            "iteration_number": state.iteration_count,
            "student_review": state.student_review,
            "review_analysis": review_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update state with deep copy to avoid reference issues
        review_history = state.review_history.copy()
        review_history.append(review_iteration)
        
        # Determine next state
        next_state = CodeReviewState.PROVIDE_GUIDANCE
        if review_analysis["review_sufficient"] or state.iteration_count >= state.max_iterations:
            next_state = CodeReviewState.SUMMARIZE
        
        # Create a new state object to avoid mutation issues
        new_state = CodeReviewGraph(
            current_state=next_state,
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            raw_errors=state.raw_errors,
            student_review=state.student_review,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
            review_history=review_history,
            review_analysis=review_analysis
        )
        
        print(f"Review analysis completed, next state: {next_state}")
        return new_state
        
    except Exception as e:
        import traceback
        logger.error(f"Error analyzing review: {str(e)}")
        print(f"Error in analyze_review: {str(e)}")
        traceback.print_exc()
        
        # Update state with error
        new_state = CodeReviewGraph(
            current_state=CodeReviewState.ERROR,
            error_message=f"Error analyzing review: {str(e)}",
            # Preserve existing state data
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            raw_errors=state.raw_errors,
            student_review=state.student_review,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
            review_history=state.review_history
        )
        
        return new_state

def provide_guidance(state: CodeReviewGraph, llm_manager: LLMManager) -> CodeReviewGraph:
    """Generate targeted guidance for the student."""
    try:
        print("Starting guidance generation")
        if not state.review_analysis:
            raise ValueError("No review analysis available to generate guidance")
        
        # Create a prompt for guidance generation
        prompt = ChatPromptTemplate.from_template("""
You are an expert Java programming mentor who provides constructive feedback to students.
Your guidance is:
1. Encouraging and supportive
2. Specific and actionable
3. Educational - teaching students how to find issues rather than just telling them what to find
4. Focused on developing their review skills
5. Balanced - acknowledging what they did well while guiding them to improve

Please create targeted guidance for a student who has reviewed Java code but missed some important errors.

ORIGINAL JAVA CODE:
```java
{code_snippet}
```

KNOWN PROBLEMS IN THE CODE:
{known_problems}

STUDENT'S REVIEW ATTEMPT #{iteration_count} of {max_iterations}:
```
{student_review}
```

PROBLEMS CORRECTLY IDENTIFIED BY THE STUDENT:
{identified_problems}

PROBLEMS MISSED BY THE STUDENT:
{missed_problems}

The student has identified {identified_count} out of {total_problems} issues ({identified_percentage:.1f}%).

Create constructive guidance that:
1. Acknowledges what the student found correctly with specific praise
2. Provides hints about the types of errors they missed (without directly listing them all)
3. Suggests specific areas of the code to examine more carefully
4. Encourages them to look for particular Java error patterns they may have overlooked
5. If there are false positives, gently explain why those are not actually issues
6. End with specific questions that might help the student find the missed problems

The guidance should be educational and help the student improve their Java code review skills.
Format your response in markdown for clarity, using headings and bullet points where appropriate.
        """)
        
        # Format lists for the prompt
        formatted_known_problems = "\n".join([f"- {problem}" for problem in state.known_problems])
        formatted_identified = "\n".join([f"- {problem}" for problem in state.review_analysis["identified_problems"]])
        formatted_missed = "\n".join([f"- {problem}" for problem in state.review_analysis["missed_problems"]])
        
        # Generate guidance with the LLM
        print("Sending guidance generation prompt to LLM")
        messages = prompt.format_messages(
            code_snippet=state.code_snippet,
            known_problems=formatted_known_problems,
            student_review=state.student_review,
            identified_problems=formatted_identified,
            missed_problems=formatted_missed,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
            identified_count=state.review_analysis["identified_count"],
            total_problems=state.review_analysis["total_problems"],
            identified_percentage=state.review_analysis["identified_percentage"]
        )
        
        response = llm_manager.review_model.invoke(messages)
        targeted_guidance = response.content
        print(f"Received guidance: {len(targeted_guidance)} characters")
        
        # Create a new review iteration with guidance
        if state.review_history:
            updated_review_history = state.review_history.copy()
            updated_review_history[-1]["targeted_guidance"] = targeted_guidance
        else:
            updated_review_history = []
        
        # Create a new state to avoid mutation issues
        new_state = CodeReviewGraph(
            current_state=CodeReviewState.REVIEW,
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            raw_errors=state.raw_errors,
            student_review=state.student_review,
            iteration_count=state.iteration_count + 1,  # Increment for next review
            max_iterations=state.max_iterations,
            review_history=updated_review_history,
            review_analysis=state.review_analysis,
            targeted_guidance=targeted_guidance
        )
        
        print("Guidance generation completed")
        return new_state
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating guidance: {str(e)}")
        print(f"Error in provide_guidance: {str(e)}")
        traceback.print_exc()
        
        # Update state with error
        new_state = CodeReviewGraph(
            current_state=CodeReviewState.ERROR,
            error_message=f"Error generating guidance: {str(e)}",
            # Preserve existing state data
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            raw_errors=state.raw_errors,
            student_review=state.student_review,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
            review_history=state.review_history,
            review_analysis=state.review_analysis
        )
        
        return new_state

def generate_summary(state: CodeReviewGraph, llm_manager: LLMManager) -> CodeReviewGraph:
    """Generate final feedback summary and comparison report."""
    try:
        print("Starting summary generation")
        if not state.review_history:
            raise ValueError("No review history available for summary")
        
        # Create a prompt for final feedback
        feedback_prompt = ChatPromptTemplate.from_template("""
You are an expert Java programming educator. Create a comprehensive final feedback summary for a student who has completed a code review exercise.

KNOWN PROBLEMS IN THE CODE:
{known_problems}

REVIEW HISTORY:
{review_history}

FINAL REVIEW ANALYSIS:
Identified problems: {identified_problems}
Missed problems: {missed_problems}
False positives: {false_positives}
Accuracy: {accuracy_percentage:.1f}%
Identified: {identified_count} out of {total_problems} issues ({identified_percentage:.1f}%)

Create a detailed feedback summary that:
1. Analyzes the student's performance in identifying code issues
2. Highlights their strengths and areas for improvement
3. Provides specific tips for future code reviews based on what they missed
4. If they made multiple review attempts, discusses their progress across iterations

Format your response in markdown, with clear headings and sections. Be encouraging while providing constructive criticism.
        """)
        
        # Format review history
        review_history_text = ""
        for i, review in enumerate(state.review_history, 1):
            review_analysis = review["review_analysis"]
            review_history_text += f"Attempt {i}:\n"
            review_history_text += f"- Identified: {review_analysis['identified_count']} of {review_analysis['total_problems']}\n"
            review_history_text += f"- Accuracy: {review_analysis['accuracy_percentage']:.1f}%\n"
            review_history_text += f"- Student review: {review['student_review'][:200]}...\n\n"
        
        # Get the final review analysis
        final_analysis = state.review_analysis
        
        # Format lists for the prompt
        formatted_known_problems = "\n".join([f"- {problem}" for problem in state.known_problems])
        formatted_identified = "\n".join([f"- {problem}" for problem in final_analysis["identified_problems"]])
        formatted_missed = "\n".join([f"- {problem}" for problem in final_analysis["missed_problems"]])
        formatted_false_positives = "\n".join([f"- {problem}" for problem in final_analysis.get("false_positives", [])])
        
        # Generate feedback with the LLM
        print("Sending feedback summary prompt to LLM")
        feedback_messages = feedback_prompt.format_messages(
            known_problems=formatted_known_problems,
            review_history=review_history_text,
            identified_problems=formatted_identified,
            missed_problems=formatted_missed,
            false_positives=formatted_false_positives,
            accuracy_percentage=final_analysis["accuracy_percentage"],
            identified_count=final_analysis["identified_count"],
            total_problems=final_analysis["total_problems"],
            identified_percentage=final_analysis["identified_percentage"]
        )
        
        feedback_response = llm_manager.summary_model.invoke(feedback_messages)
        review_summary = feedback_response.content
        print(f"Received feedback summary: {len(review_summary)} characters")
        
        # Create a prompt for comparison report
        comparison_prompt = ChatPromptTemplate.from_template("""
You are an expert Java code review educator. Create a detailed comparison between the student's review and the actual issues in the code.

ORIGINAL CODE:
```java
{code_snippet}
```

ACTUAL PROBLEMS IN THE CODE:
{known_problems}

STUDENT'S FINAL REVIEW:
```
{student_review}
```

FINAL REVIEW ANALYSIS:
Identified problems: {identified_problems}
Missed problems: {missed_problems}
False positives: {false_positives}
Accuracy: {accuracy_percentage:.1f}%

Create a comprehensive comparison report that:
1. Provides a detailed analysis of what the student found vs. what they missed
2. Explains why the missed issues are important to catch in code reviews
3. Provides specific advice for each type of issue they missed
4. Gives constructive feedback on any false positives they reported
5. Includes a section on review patterns and strategies for improvement

Format your response in markdown with clear section headings and subheadings. Be educational and constructive.
        """)
        
        # Generate comparison report with the LLM
        print("Sending comparison report prompt to LLM")
        comparison_messages = comparison_prompt.format_messages(
            code_snippet=state.code_snippet,
            known_problems=formatted_known_problems,
            student_review=state.review_history[-1]["student_review"],
            identified_problems=formatted_identified,
            missed_problems=formatted_missed,
            false_positives=formatted_false_positives,
            accuracy_percentage=final_analysis["accuracy_percentage"]
        )
        
        comparison_response = llm_manager.compare_model.invoke(comparison_messages)
        comparison_report = comparison_response.content
        print(f"Received comparison report: {len(comparison_report)} characters")
        
        # Create a new state to avoid mutation issues
        new_state = CodeReviewGraph(
            current_state=CodeReviewState.COMPLETE,
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            raw_errors=state.raw_errors,
            student_review=state.student_review,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
            review_history=state.review_history,
            review_analysis=state.review_analysis,
            targeted_guidance=state.targeted_guidance,
            review_summary=review_summary,
            comparison_report=comparison_report
        )
        
        print("Summary generation completed")
        return new_state
        
    except Exception as e:
        import traceback
        logger.error(f"Error generating summary: {str(e)}")
        print(f"Error in generate_summary: {str(e)}")
        traceback.print_exc()
        
        # Update state with error
        new_state = CodeReviewGraph(
            current_state=CodeReviewState.ERROR,
            error_message=f"Error generating summary: {str(e)}",
            # Preserve existing state data
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            raw_errors=state.raw_errors,
            student_review=state.student_review,
            iteration_count=state.iteration_count,
            max_iterations=state.max_iterations,
            review_history=state.review_history,
            review_analysis=state.review_analysis,
            targeted_guidance=state.targeted_guidance
        )
        
        return new_state

def handle_student_review(state: CodeReviewGraph, student_review: str) -> CodeReviewGraph:
    """Handle a student review submission."""
    print(f"Handling student review: {len(student_review)} characters")
    # Create a new state object to avoid mutation issues
    new_state = CodeReviewGraph(
        current_state=CodeReviewState.ANALYZE,
        code_snippet=state.code_snippet,
        known_problems=state.known_problems,
        raw_errors=state.raw_errors,
        student_review=student_review,
        iteration_count=state.iteration_count,
        max_iterations=state.max_iterations,
        review_history=state.review_history,
        review_analysis=state.review_analysis,
        targeted_guidance=state.targeted_guidance,
        review_summary=state.review_summary,
        comparison_report=state.comparison_report
    )
    return new_state

def should_provide_guidance(state: CodeReviewGraph) -> str:
    """Conditional router to determine next step after analysis."""
    if state.current_state == CodeReviewState.ANALYZE:
        if state.review_analysis and state.review_analysis.get("review_sufficient", False):
            return CodeReviewState.SUMMARIZE
        elif state.iteration_count >= state.max_iterations:
            return CodeReviewState.SUMMARIZE
        else:
            return CodeReviewState.PROVIDE_GUIDANCE
    return END

# Helper function to extract JSON from LLM response
def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON data from LLM response text."""
    import re
    import json
    
    try:
        print("Extracting JSON from LLM response")
        # Try to find JSON block with regex
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # Try to find any JSON object
        json_match = re.search(r'({[\s\S]*?})', text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # If no JSON found, return empty dict
        logger.warning("Could not extract JSON from response")
        print("Warning: Could not extract JSON from response")
        # Return some default values to avoid errors
        return {
            "identified_problems": [],
            "missed_problems": ["Could not parse response properly"],
            "false_positives": [],
            "accuracy_percentage": 0.0,
            "review_sufficient": False,
            "feedback": "There was an issue analyzing your review. Please try again."
        }
        
    except Exception as e:
        logger.error(f"Error extracting JSON: {str(e)}")
        print(f"Error extracting JSON: {str(e)}")
        # Return some default values to avoid errors
        return {
            "identified_problems": [],
            "missed_problems": ["Error processing review"],
            "false_positives": [],
            "accuracy_percentage": 0.0,
            "review_sufficient": False,
            "feedback": f"Error processing review: {str(e)}"
        }

# Main CodeReviewAgent class
class CodeReviewAgent:
    """Main agent class that coordinates the code review workflow using LangGraph."""
    
    def __init__(self):
        """Initialize the code review agent."""
        try:
            print("Initializing CodeReviewAgent")
            # Initialize dependencies
            self.llm_manager = LLMManager(initialize_models=True)
            self.error_repository = ErrorRepository()
            
            # Build the workflow graph
            self.graph = self._build_graph()
            
            # Simplified without checkpoint
            self.workflow = self.graph.compile()
            print("CodeReviewAgent initialized successfully")
        except Exception as e:
            import traceback
            print(f"Error initializing CodeReviewAgent: {str(e)}")
            traceback.print_exc()
            raise
    
    def _build_graph(self) -> StateGraph:
        """Build the code review workflow graph."""
        try:
            print("Building workflow graph")
            # Create the graph
            graph = StateGraph(CodeReviewGraph)
            
            # Define nodes
            generate_code_node = lambda state: generate_code(state, self.llm_manager, self.error_repository)
            analyze_review_node = lambda state: analyze_review(state, self.llm_manager)
            provide_guidance_node = lambda state: provide_guidance(state, self.llm_manager)
            generate_summary_node = lambda state: generate_summary(state, self.llm_manager)
            
            # Add nodes
            graph.add_node(CodeReviewState.GENERATE, generate_code_node)
            graph.add_node(CodeReviewState.ANALYZE, analyze_review_node)
            graph.add_node(CodeReviewState.PROVIDE_GUIDANCE, provide_guidance_node)
            graph.add_node(CodeReviewState.SUMMARIZE, generate_summary_node)
            
            # Add missing nodes
            graph.add_node(CodeReviewState.COMPLETE, lambda state: state)
            graph.add_node(CodeReviewState.REVIEW, lambda state: state)
            
            # Add conditional routing
            graph.add_conditional_edges(
                CodeReviewState.ANALYZE,
                should_provide_guidance
            )
            
            # Add transitions
            graph.add_edge(CodeReviewState.GENERATE, CodeReviewState.REVIEW)
            graph.add_edge(CodeReviewState.PROVIDE_GUIDANCE, CodeReviewState.REVIEW)
            graph.add_edge(CodeReviewState.SUMMARIZE, CodeReviewState.COMPLETE)
            
            # Set entry point
            graph.set_entry_point(CodeReviewState.GENERATE)
            
            print("Workflow graph built successfully")
            return graph
        except Exception as e:
            print(f"Error building workflow graph: {str(e)}")
            raise
    
    def generate_code_problem(self) -> CodeReviewGraph:
        """Generate a Java code problem with intentional errors."""
        try:
            print("Starting code problem generation")
            # Create initial state
            initial_state = CodeReviewGraph(current_state=CodeReviewState.GENERATE)
            
            # Run the workflow to generate code - without thread parameter
            result = self.workflow.invoke(initial_state)
            print(f"Code problem generation completed, state: {result.current_state}")
            
            # Return the final state after code generation
            return result
        except Exception as e:
            import traceback
            print(f"Error in generate_code_problem: {str(e)}")
            traceback.print_exc()
            
            # Return error state
            return CodeReviewGraph(
                current_state=CodeReviewState.ERROR,
                error_message=f"Error generating code problem: {str(e)}"
            )
    
    def submit_review(self, state: CodeReviewGraph, student_review: str) -> CodeReviewGraph:
        """Submit a student review for analysis."""
        try:
            print("Starting review submission")
            # Update state with student review
            new_state = handle_student_review(state, student_review)
            
            # Run the workflow from analysis - without thread parameter
            result = self.workflow.invoke(new_state)
            print(f"Review submission completed, state: {result.current_state}")
            
            # Return the updated state
            return result
        except Exception as e:
            import traceback
            print(f"Error in submit_review: {str(e)}")
            traceback.print_exc()
            
            # Return error state
            return CodeReviewGraph(
                current_state=CodeReviewState.ERROR,
                error_message=f"Error processing review: {str(e)}",
                # Preserve existing state data
                code_snippet=state.code_snippet,
                known_problems=state.known_problems,
                raw_errors=state.raw_errors,
                student_review=student_review,
                iteration_count=state.iteration_count,
                max_iterations=state.max_iterations,
                review_history=state.review_history,
                review_analysis=state.review_analysis,
                targeted_guidance=state.targeted_guidance
            )
    
    def reset_session(self) -> None:
        """Reset the current session."""
        print("Resetting session")
        # No need to clear checkpoints, just create a new workflow
        self.workflow = self.graph.compile()

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = CodeReviewAgent()
    
    # Generate a code problem
    code_state = agent.generate_code_problem()
    print(f"Generated code with {len(code_state.known_problems)} intentional problems")
    
    # Submit a student review (example)
    student_review = "I found a null pointer exception in the getStudent method where a null reference might be dereferenced."
    result_state = agent.submit_review(code_state, student_review)
    
    # Check the result
    if result_state.current_state == CodeReviewState.COMPLETE:
        print("Review process completed!")
        print(f"Accuracy: {result_state.review_analysis['accuracy_percentage']:.1f}%")
        print(f"Found {result_state.review_analysis['identified_count']} out of {result_state.review_analysis['total_problems']} issues")
    elif result_state.current_state == CodeReviewState.REVIEW:
        print("More review iterations needed")
        print(f"Current iteration: {result_state.iteration_count-1}")
        if result_state.targeted_guidance:
            print("Targeted guidance provided for next iteration")
    elif result_state.current_state == CodeReviewState.ERROR:
        print(f"Error: {result_state.error_message}")