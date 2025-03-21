"""
Code Display UI module for Java Peer Review Training System.

This module provides the CodeDisplayUI class for displaying Java code snippets
and handling student review input.
"""

import streamlit as st
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from utils.code_utils import add_line_numbers,strip_error_annotations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeDisplayUI:
    """
    UI Component for displaying Java code snippets.
    
    This class handles displaying Java code snippets with syntax highlighting,
    line numbers, and optional instructor view.    """
    
   

    def render_code_display(self, code_snippet, known_problems: List[str] = None) -> None:
        """
        Render a code snippet with optional known problems for instructor view.
        
        Args:
            code_snippet: Java code snippet to display (string or CodeSnippet object)
            known_problems: Optional list of known problems for instructor view
        """
        if not code_snippet:
            st.info("No code generated yet. Use the 'Generate Code Problem' tab to create a Java code snippet.")
            return
        
        st.subheader("Java Code to Review:")
        
        # Handle different input types to get the display code
        if isinstance(code_snippet, str):
            # If string is passed directly, remove all error comments/annotations
            from utils.code_utils import strip_error_annotations
            display_code = strip_error_annotations(code_snippet)
        else:
            # If it's a CodeSnippet object
            if hasattr(code_snippet, 'clean_code') and code_snippet.clean_code:
                # Use clean version for student view
                display_code = code_snippet.clean_code
            else:
                # Only fall back if clean_code is not available, and strip comments
                from utils.code_utils import strip_error_annotations
                display_code = strip_error_annotations(code_snippet.code)
        
                
        lines = display_code.splitlines()
        cleaned_lines = []
        for line in lines:
            # Skip any line with ERROR in comments
            if "// ERROR" in line or "//ERROR" in line or "// error" in line:
                continue
            cleaned_lines.append(line)
        display_code = "\n".join(cleaned_lines)
        
        # Add line numbers to the code snippet
        numbered_code = self._add_line_numbers(display_code)
        st.code(numbered_code, language="java")
        
        # INSTRUCTOR VIEW: Show known problems if provided
        if known_problems:
            if st.checkbox("Show Known Problems (Instructor View)", value=False):
                st.subheader("Known Problems:")
                for i, problem in enumerate(known_problems, 1):
                    st.markdown(f"{i}. {problem}")
                    
                # Add option to view annotated code with error comments for instructors
                if isinstance(code_snippet, object) and hasattr(code_snippet, 'code'):
                    if st.checkbox("Show Annotated Code (with Error Comments)", value=False):
                        st.subheader("Annotated Code (with Error Comments):")
                        annotated_code = self._add_line_numbers(code_snippet.code)
                        st.code(annotated_code, language="java")
    
    def _add_line_numbers(self, code: str) -> str:
        """Add line numbers to code snippet using shared utility."""
        return add_line_numbers(code)
    
    # In ui/code_display.py, update the render_review_input method:

    def render_review_input(self, student_review: str = "", 
                    on_submit_callback: Callable[[str], None] = None,
                    iteration_count: int = 1,
                    max_iterations: int = 3,
                    targeted_guidance: str = None,
                    review_analysis: Dict[str, Any] = None) -> None:
        """
        Render a professional text area for student review input with guidance.
        
        Args:
            student_review: Initial value for the text area
            on_submit_callback: Callback function when review is submitted
            iteration_count: Current iteration number
            max_iterations: Maximum number of iterations
            targeted_guidance: Optional guidance for the student
            review_analysis: Optional analysis of previous review attempt
        """
        
        # Review container start
        st.markdown('<div class="review-container">', unsafe_allow_html=True)
        
        # Review header with iteration badge if not the first iteration
        if iteration_count > 1:
            st.markdown(
                f'<div class="review-header">'
                f'<span class="review-title">Submit Your Code Review</span>'
                f'<span class="iteration-badge">Attempt {iteration_count} of {max_iterations}</span>'
                f'</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="review-header"><span class="review-title">Submit Your Code Review</span></div>', unsafe_allow_html=True)
        
        # Create a layout for guidance and history
        if targeted_guidance or (review_analysis and iteration_count > 1):
            guidance_col, history_col = st.columns([2, 1])
            
            # Display targeted guidance if available (for iterations after the first)
            with guidance_col:
                if targeted_guidance and iteration_count > 1:
                    st.markdown(
                        f'<div class="guidance-box">'
                        f'<div class="guidance-title"><span class="guidance-icon">🎯</span> Review Guidance</div>'
                        f'{targeted_guidance}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show previous attempt results if available
                    if review_analysis:
                        st.markdown(
                            f'<div class="analysis-box">'
                            f'<div class="guidance-title"><span class="guidance-icon">📊</span> Previous Results</div>'
                            f'You identified {review_analysis.get("identified_count", 0)} of '
                            f'{review_analysis.get("total_problems", 0)} issues '
                            f'({review_analysis.get("identified_percentage", 0):.1f}%). '
                            f'Try to find more issues in this attempt.'
                            f'</div>',
                            unsafe_allow_html=True
                        )
            
            # Display previous review if available
            with history_col:
                if student_review and iteration_count > 1:
                    st.markdown('<div class="guidance-title"><span class="guidance-icon">📝</span> Previous Review</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="review-history-box">'
                        f'<pre style="margin: 0; white-space: pre-wrap; font-size: 0.85rem; color: var(--text);">{student_review}</pre>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        # Display review guidelines - UPDATED for natural language
        with st.expander("📋 Review Guidelines", expanded=False):
            st.markdown("""
            ### How to Write an Effective Code Review:
            
            1. **Be Specific**: Point out exact lines or areas where problems occur
            2. **Be Comprehensive**: Look for different types of issues
            3. **Be Constructive**: Suggest improvements, not just criticisms
            4. **Check for**:
            - Syntax and compilation errors
            - Logical errors and bugs
            - Naming conventions and coding standards
            - Code style and formatting issues
            - Documentation completeness
            - Potential security vulnerabilities
            - Efficiency and performance concerns
            5. **Format Your Review**: Use a consistent format like:
            ```
            Line X: Description of the issue and why it's problematic
            ```
            
            ### Examples of Good Review Comments:
            
            ```
            Line 15: The variable name 'cnt' is too short and unclear. It should be renamed to something more descriptive like 'counter'.
            
            Line 27: This loop will miss the last element because it uses < instead of <=
            
            Line 42: The string comparison uses == instead of .equals() which will compare references not content
            
            Line 72: Missing null check before calling method on user object
            ```
            
            You don't need to use formal error categories - writing in natural language is perfect!
            """)
        
        # Get or update the student review
        st.write("### Your Review:")
        
        # Create a unique key for the text area
        text_area_key = f"student_review_input_{iteration_count}"
        
        # Initialize with previous review text only on first load of this iteration
        initial_value = ""
        if iteration_count == 1 or not student_review:
            initial_value = ""  # Start fresh for first iteration or if no previous review
        else:
            # For subsequent iterations, preserve existing input in session state if any
            if text_area_key in st.session_state:
                initial_value = st.session_state[text_area_key]
            else:
                initial_value = ""  # Start fresh in new iterations
        
        # Get or update the student review with custom styling
        student_review_input = st.text_area(
            "Enter your review comments here",
            value=initial_value, 
            height=300,
            key=text_area_key,
            placeholder="Example:\nLine 15: The variable 'cnt' uses poor naming. Consider using 'counter' instead.\nLine 27: The loop condition should use '<=' instead of '<' to include the boundary value.",
            label_visibility="collapsed",
            help="Provide detailed feedback on the code. Be specific about line numbers and issues you've identified."
        )
        
        # Create button container for better layout
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        
        # Submit button with professional styling
        submit_text = "Submit Review" if iteration_count == 1 else f"Submit Review (Attempt {iteration_count}/{max_iterations})"
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown('<div class="submit-button">', unsafe_allow_html=True)
            submit_button = st.button(submit_text, type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="clear-button">', unsafe_allow_html=True)
            clear_button = st.button("Clear", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close button container
        
        # Handle clear button
        if clear_button:
            st.session_state[text_area_key] = ""
            st.rerun()
        
        # Handle submit button with improved validation
        if submit_button:
            if not student_review_input.strip():
                st.error("Please enter your review before submitting.")
            elif on_submit_callback:
                # Show a spinner while processing
                with st.spinner("Processing your review..."):
                    # Call the submission callback
                    on_submit_callback(student_review_input)
                    
                    # Store the submitted review in session state for this iteration
                    if f"submitted_review_{iteration_count}" not in st.session_state:
                        st.session_state[f"submitted_review_{iteration_count}"] = student_review_input
        
        # Close review container
        st.markdown('</div>', unsafe_allow_html=True)







                   