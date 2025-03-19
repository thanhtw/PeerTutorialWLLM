"""
State Schema for Java Code Review Training System.

This module defines the state schema for the LangGraph-based workflow.
"""

__all__ = ['WorkflowState', 'CodeSnippet', 'ReviewAttempt']

from typing import List, Dict, Any, Optional, TypedDict, Literal
from pydantic import BaseModel, Field

# Update the CodeSnippet class in state_schema.py

class CodeSnippet(BaseModel):
    """Schema for code snippet data"""
    code: str = Field(description="The Java code snippet with annotations")
    clean_code: str = Field("", description="The Java code snippet without annotations")
    known_problems: List[str] = Field(default_factory=list, description="List of known problems in the code")
    raw_errors: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Raw error data organized by type")
    enhanced_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Enhanced error data with location information")

class ReviewAttempt(BaseModel):
    """Schema for a student review attempt"""
    student_review: str = Field(description="The student's review text")
    iteration_number: int = Field(description="Iteration number of this review")
    analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis of the review")
    targeted_guidance: Optional[str] = Field(None, description="Targeted guidance for next iteration")

class WorkflowState(BaseModel):
    """The state for the Java Code Review workflow"""
    # Current workflow step
    current_step: Literal["generate", "review", "analyze", "summarize", "complete"] = Field(
        "generate", description="Current step in the workflow"
    )
    
    # Code generation parameters
    code_length: str = Field("medium", description="Length of code (short, medium, long)")
    difficulty_level: str = Field("medium", description="Difficulty level (easy, medium, hard)")
    
    # IMPORTANT: Replace underscore field with properly named field
    # Use a single field with a clear name
    selected_error_categories: Dict[str, List[str]] = Field(
        default_factory=lambda: {"build": [], "checkstyle": []}, 
        description="Selected error categories"
    )
    
    # Code data
    code_snippet: Optional[CodeSnippet] = Field(None, description="Generated code snippet data")
    
    # Review data
    review_history: List[ReviewAttempt] = Field(default_factory=list, description="History of review attempts")
    current_iteration: int = Field(1, description="Current iteration number")
    max_iterations: int = Field(3, description="Maximum number of iterations")
    
    # Analysis results
    review_sufficient: bool = Field(False, description="Whether the review is sufficient")
    review_summary: Optional[str] = Field(None, description="Final review summary")
    comparison_report: Optional[str] = Field(None, description="Comparison report")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if any")