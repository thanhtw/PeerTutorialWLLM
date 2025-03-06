"""
Core domain logic package for Java Peer Review Training System.

This package contains the core domain classes that implement the business logic
of the code review training system.
"""

from core.code_generator import CodeGenerator
from core.error_injector import ErrorInjector
from core.student_response_evaluator import StudentResponseEvaluator
from core.feedback_manager import FeedbackManager

__all__ = [
    'CodeGenerator',
    'ErrorInjector',
    'StudentResponseEvaluator',
    'FeedbackManager'
]