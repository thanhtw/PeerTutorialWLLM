"""
MCP Server module for Java Peer Review Training System.

This module implements the Model Control Protocol (MCP) server
that provides tools for code generation, evaluation, and feedback.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    name="java-review-server",
)

@mcp.tool()
def generate_java_code(code_length: str, difficulty_level: str, 
                      error_types: List[Dict[str, Any]], domain: str = "student_management") -> str:
    """Generate Java code with specific errors.
    
    Args:
        code_length: Length of code (short, medium, long)
        difficulty_level: Difficulty level (easy, medium, hard)
        error_types: List of error types to include
        domain: Domain context for the code
        
    Returns:
        Generated Java code with errors
    """
    from utils.code_utils import create_code_generation_prompt
    
    # Create error descriptions for the prompt
    error_descriptions = []
    for i, error in enumerate(error_types, 1):
        error_type = error.get("type", "").upper()
        name = error.get("name", "")
        description = error.get("description", "")
        implementation_guide = error.get("implementation_guide", "")
        
        error_descriptions.append(f"{i}. {error_type} ERROR - {name}\n"
                                f"   Description: {description}\n"
                                f"   Implementation guide: {implementation_guide}")
    
    # Call LLM manager to generate code
    from llm_manager import LLMManager
    llm_manager = LLMManager()
    
    generative_model = llm_manager.initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
    
    if not generative_model:
        return "Error: Could not initialize generative model"
    
    prompt = create_code_generation_prompt(
        code_length=code_length,
        difficulty_level=difficulty_level,
        selected_errors=error_types,
        domain=domain,
        include_error_annotations=True
    )
    
    response = generative_model.invoke(prompt)
    
    from utils.code_utils import extract_code_from_response
    code = extract_code_from_response(response)
    
    return code

@mcp.tool()
def evaluate_java_code(code: str, requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate Java code to check if it contains the requested errors.
    
    Args:
        code: Java code to evaluate
        requested_errors: List of errors that should be in the code
        
    Returns:
        Evaluation results
    """
    from utils.improved_code_evaluation_agent import ImprovedCodeEvaluationAgent
    from llm_manager import LLMManager
    
    llm_manager = LLMManager()
    evaluation_model = llm_manager.initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
    
    evaluation_agent = ImprovedCodeEvaluationAgent(evaluation_model)
    evaluation_result = evaluation_agent.evaluate_code(code, requested_errors)
    
    return evaluation_result

@mcp.tool()
def analyze_student_review(code_snippet: str, known_problems: List[str], 
                         student_review: str) -> Dict[str, Any]:
    """Analyze a student's code review.
    
    Args:
        code_snippet: Java code that was reviewed
        known_problems: List of known problems in the code
        student_review: The student's review text
        
    Returns:
        Analysis results
    """
    from core.student_response_evaluator import StudentResponseEvaluator
    from llm_manager import LLMManager
    
    llm_manager = LLMManager()
    review_model = llm_manager.initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
    
    evaluator = StudentResponseEvaluator(review_model)
    analysis = evaluator.evaluate_review(code_snippet, known_problems, student_review)
    
    return analysis

@mcp.tool()
def generate_targeted_guidance(code_snippet: str, known_problems: List[str],
                             student_review: str, review_analysis: Dict[str, Any],
                             iteration_count: int, max_iterations: int) -> str:
    """Generate targeted guidance for a student based on their review.
    
    Args:
        code_snippet: Java code that was reviewed
        known_problems: List of known problems in the code
        student_review: The student's review text
        review_analysis: Analysis of the student review
        iteration_count: Current iteration number
        max_iterations: Maximum number of iterations
        
    Returns:
        Targeted guidance text
    """
    from core.student_response_evaluator import StudentResponseEvaluator
    from llm_manager import LLMManager
    
    llm_manager = LLMManager()
    review_model = llm_manager.initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
    
    evaluator = StudentResponseEvaluator(review_model)
    guidance = evaluator.generate_targeted_guidance(
        code_snippet, known_problems, student_review, 
        review_analysis, iteration_count, max_iterations
    )
    
    return guidance

@mcp.tool()
def generate_final_feedback(code_snippet: str, known_problems: List[str], 
                          review_history: List[Dict[str, Any]]) -> str:
    """Generate final feedback after all review iterations.
    
    Args:
        code_snippet: Java code that was reviewed
        known_problems: List of known problems in the code
        review_history: History of review attempts
        
    Returns:
        Final feedback text
    """
    from core.feedback_manager import FeedbackManager
    from core.student_response_evaluator import StudentResponseEvaluator
    from llm_manager import LLMManager
    
    llm_manager = LLMManager()
    summary_model = llm_manager.initialize_model_from_env("SUMMARY_MODEL", "SUMMARY_TEMPERATURE")
    
    evaluator = StudentResponseEvaluator(summary_model)
    feedback_manager = FeedbackManager(evaluator)
    
    # Initialize the feedback manager with the code and known problems
    feedback_manager.start_new_review_session(code_snippet, known_problems)
    
    # Submit all reviews to the feedback manager
    for review in review_history:
        feedback_manager.submit_review(review.get("student_review", ""))
    
    # Generate final feedback
    feedback = feedback_manager.generate_final_feedback()
    
    return feedback

@mcp.tool()
def generate_comparison_report(known_problems: List[str], review_analysis: Dict[str, Any]) -> str:
    """Generate a comparison report between known problems and student review.
    
    Args:
        known_problems: List of known problems in the code
        review_analysis: Analysis of the student's review
        
    Returns:
        Comparison report
    """
    from utils.code_utils import generate_comparison_report
    
    report = generate_comparison_report(known_problems, review_analysis)
    
    return report

@mcp.tool()
def check_gpu_availability() -> Dict[str, Any]:
    """Check if GPU acceleration is available.
    
    Returns:
        Dictionary with GPU availability information
    """
    from llm_manager import LLMManager
    
    llm_manager = LLMManager()
    gpu_info = llm_manager.check_gpu_availability(extended=True)
    
    return gpu_info

@mcp.tool()
def get_available_models() -> List[Dict[str, Any]]:
    """Get a list of available models.
    
    Returns:
        List of available models with details
    """
    from llm_manager import LLMManager
    
    llm_manager = LLMManager()
    models = llm_manager.get_available_models()
    
    return models

@mcp.tool()
def download_model(model_name: str) -> Dict[str, Any]:
    """Download a model.
    
    Args:
        model_name: Name of the model to download
        
    Returns:
        Result of the download operation
    """
    from llm_manager import LLMManager
    
    llm_manager = LLMManager()
    success = llm_manager.download_ollama_model(model_name)
    
    return {
        "model": model_name,
        "success": success,
        "status": llm_manager.get_pull_status(model_name)
    }