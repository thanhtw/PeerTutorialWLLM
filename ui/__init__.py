"""
UI package for Java Peer Review Training System.

This package contains UI components for the Streamlit interface
that handle user interaction and display of results.
"""

from ui.error_selector import ErrorSelectorUI
from ui.code_display import CodeDisplayUI
from ui.feedback_display import FeedbackDisplayUI
from ui.model_manager import ModelManagerUI

__all__ = [
    'ErrorSelectorUI',
    'CodeDisplayUI',
    'FeedbackDisplayUI',
    'ModelManagerUI'
]