"""
Code Display UI module for Java Peer Review Training System.

This module provides the CodeDisplayUI class for displaying Java code snippets
and handling student review input.
"""

import streamlit as st
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from utils.code_utils import add_line_numbers

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
    line numbers, and optional instructor view.
    """
    
    def render_code_display(self, code_snippet: str, known_problems: List[str] = None) -> None:
        """
        Render a code snippet with optional known problems for instructor view.
        
        Args:
            code_snippet: Java code snippet to display
            known_problems: Optional list of known problems for instructor view
        """
        if not code_snippet:
            st.info("No code generated yet. Use the 'Generate Code Problem' tab to create a Java code snippet.")
            return
        
        st.subheader("Java Code to Review:")       
        
        # Add line numbers to the code snippet
        numbered_code = self._add_line_numbers(code_snippet)
        st.code(numbered_code, language="java")
        
        # Create a unique key based on content to prevent duplicate keys
        download_key = f"download_code_{hash(code_snippet)%10000}"
        
        # Download button for the code
        # if st.download_button(
        #     label="Download Code", 
        #     data=code_snippet,
        #     file_name="java_review_problem.java",
        #     mime="text/plain",
        #     key=download_key
        # ):
        #     st.success("Code downloaded successfully!")
        
        # INSTRUCTOR VIEW: Show known problems if provided
        if known_problems:
            if st.checkbox("Show Known Problems (Instructor View)", value=False):
                st.subheader("Known Problems:")
                for i, problem in enumerate(known_problems, 1):
                    st.markdown(f"{i}. {problem}")
    
    def _add_line_numbers(self, code: str) -> str:
        """Add line numbers to code snippet using shared utility."""
        return add_line_numbers(code)
    
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
        
        # Display review guidelines
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
            Line X: [Issue type] - Description of the issue
            Line Y: [Issue type] - Description of the issue
            ```
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
            placeholder="Example:\nLine 15: [Naming Convention] - The variable 'cnt' uses poor naming. Consider using 'counter' instead.\nLine 27: [Logic Error] - The loop condition should use '<=' instead of '<' to include the boundary value.",
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







                   