"""
Code Display UI module for Java Peer Review Training System.

This module provides the CodeDisplayUI class for displaying Java code snippets
and handling student review input.
"""

import streamlit as st
import logging
import random
from typing import List, Dict, Any, Optional, Tuple, Callable

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
    
    def __init__(self):
        """Initialize the CodeDisplayUI component with an instance ID for unique keys."""
        # Generate a random instance ID to ensure unique keys across instances
        self.instance_id = random.randint(1000, 9999)
    
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
        
        # Create a truly unique key using multiple factors
        # 1. Instance ID of this class
        # 2. Hash of code snippet with larger modulo
        # 3. Random suffix to ensure uniqueness
        code_hash = abs(hash(code_snippet)) % 100000
        random_suffix = random.randint(10000, 99999)
        download_key = f"download_code_{self.instance_id}_{code_hash}_{random_suffix}"
        
        # Download button for the code
        if st.download_button(
            label="Download Code", 
            data=code_snippet,
            file_name="java_review_problem.java",
            mime="text/plain",
            key=download_key
        ):
            st.success("Code downloaded successfully!")
        
        # INSTRUCTOR VIEW: Show known problems if provided
        if known_problems:
            # Use the same strategy to create a unique key for the checkbox
            checkbox_key = f"show_problems_{self.instance_id}_{random.randint(1000, 9999)}"
            if st.checkbox("Show Known Problems (Instructor View)", value=False, key=checkbox_key):
                st.subheader("Known Problems:")
                for i, problem in enumerate(known_problems, 1):
                    st.markdown(f"{i}. {problem}")
    
    def _add_line_numbers(self, code: str) -> str:
        """
        Add line numbers to code snippet.
        
        Args:
            code: The code snippet to add line numbers to
            
        Returns:
            Code with line numbers
        """
        lines = code.splitlines()
        max_line_num = len(lines)
        padding = len(str(max_line_num))
        
        # Create a list of lines with line numbers
        numbered_lines = []
        for i, line in enumerate(lines, 1):
            # Format line number with consistent padding
            line_num = str(i).rjust(padding)
            numbered_lines.append(f"{line_num} | {line}")
        
        return "\n".join(numbered_lines)
    
    def render_review_input(self, student_review: str = "", 
                          on_submit_callback: Callable[[str], None] = None,
                          iteration_count: int = 1,
                          max_iterations: int = 3,
                          targeted_guidance: str = None,
                          review_analysis: Dict[str, Any] = None) -> None:
        """
        Render a text area for student review input with guidance.
        
        Args:
            student_review: Initial value for the text area
            on_submit_callback: Callback function when review is submitted
            iteration_count: Current iteration number
            max_iterations: Maximum number of iterations
            targeted_guidance: Optional guidance for the student
            review_analysis: Optional analysis of previous review attempt
        """
        # Show iteration badge if not the first iteration
        if iteration_count > 1:
            st.markdown(
                f"<h3>Submit Your Code Review "
                f"<span class='iteration-badge'>Attempt {iteration_count} of "
                f"{max_iterations}</span></h3>", 
                unsafe_allow_html=True
            )
        else:
            st.header("Submit Your Code Review")
        
        # Create a layout for guidance and history
        if targeted_guidance or (review_analysis and iteration_count > 1):
            guidance_col, history_col = st.columns([2, 1])
            
            # Display targeted guidance if available (for iterations after the first)
            with guidance_col:
                if targeted_guidance and iteration_count > 1:
                    st.markdown(
                        f'<div class="guidance-box">'
                        f'<h4>🎯 Review Guidance</h4>'
                        f'{targeted_guidance}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show previous attempt results if available
                    if review_analysis:
                        st.markdown(
                            f'<div class="warning-box">'
                            f'<h4>⚠️ Previous Results</h4>'
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
                    st.markdown("#### Previous Review")
                    st.markdown(
                        f'<div class="review-box" style="max-height: 200px; overflow-y: auto;">'
                        f'<pre style="margin: 0; white-space: pre-wrap;">{student_review}</pre>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        # Display review guidelines
        # Create a unique key for the expander
        guidelines_key = f"guidelines_{self.instance_id}_{random.randint(1000, 9999)}"
        with st.expander("📝 Review Guidelines", expanded=False, key=guidelines_key):
            st.markdown("""
            ### How to Write an Effective Code Review:
            
            1. **Be Specific**: Point out exact lines or issues
            2. **Be Constructive**: Suggest improvements, not just problems
            3. **Check for**:
               - Syntax errors
               - Logical errors
               - Naming conventions
               - Code style and formatting
               - Documentation
               - Potential bugs
            4. **Format Your Review**: Organize your comments by issue type
            """)
        
        # Get or update the student review
        st.write("### Your Review:")
        
        # Create a unique key for the text area
        text_area_key = f"student_review_input_{self.instance_id}_{iteration_count}"
        
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
        
        # Get or update the student review
        student_review_input = st.text_area(
            "Enter your review comments here",
            value=initial_value, 
            height=300,
            key=text_area_key,
            placeholder="Example: Line 15: The variable 'cnt' uses poor naming convention. Consider using a more descriptive name like 'counter'.",
            label_visibility="collapsed"
        )
        
        # Submit controls with layout
        submit_col1, submit_col2 = st.columns([4, 1])
        
        # Submit button
        submit_text = "Submit Review" if iteration_count == 1 else f"Submit Review (Attempt {iteration_count}/{max_iterations})"
        
        # Create unique keys for buttons
        submit_key = f"submit_review_{self.instance_id}_{iteration_count}"
        clear_key = f"clear_review_{self.instance_id}_{iteration_count}"
        
        with submit_col1:
            submit_button = st.button(submit_text, type="primary", use_container_width=True, key=submit_key)
        
        with submit_col2:
            clear_button = st.button("Clear", use_container_width=True, key=clear_key)
            
        if clear_button:
            st.session_state[text_area_key] = ""
            st.rerun()
        
        if submit_button:
            if not student_review_input.strip():
                st.warning("Please enter your review before submitting.")
            elif on_submit_callback:
                # Call the submission callback
                on_submit_callback(student_review_input)
                
                # Store the submitted review in session state for this iteration
                if f"submitted_review_{iteration_count}" not in st.session_state:
                    st.session_state[f"submitted_review_{iteration_count}"] = student_review_input