"""
UI Components module for Java Peer Review Training System.

This module provides modular UI components for the Streamlit interface,
including ErrorSelectorUI, CodeDisplayUI, and FeedbackDisplayUI.
"""

import streamlit as st
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorSelectorUI:
    """
    UI Component for error category selection.
    
    This class handles displaying and selecting Java error categories
    from both build errors and checkstyle errors.
    """
    
    def __init__(self):
        """Initialize the ErrorSelectorUI component."""
        # Track selected categories
        if "selected_error_categories" not in st.session_state:
            st.session_state.selected_error_categories = {
                "build": [],
                "checkstyle": []
            }
        
        # Track error selection mode
        if "error_selection_mode" not in st.session_state:
            st.session_state.error_selection_mode = "standard"
        
        # Track expanded categories
        if "expanded_categories" not in st.session_state:
            st.session_state.expanded_categories = {}
    
    def render_category_selection(self, all_categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Render the error category selection UI.
        
        Args:
            all_categories: Dictionary with 'build' and 'checkstyle' categories
            
        Returns:
            Dictionary with selected categories
        """
        st.subheader("Select Specific Error Categories")
        st.info("Choose specific error categories to include in the generated Java code.")
        
        # Add CSS for nested subcategory display
        st.markdown("""
        <style>
            .subcategory-container {
                margin-left: 20px;
                border-left: 2px solid #e6e6e6;
                padding-left: 10px;
            }
            .error-item {
                margin: 5px 0;
                padding: 3px 0;
                font-size: 0.9em;
            }
            .category-header {
                font-weight: bold;
                margin-top: 10px;
                background-color: #f1f1f1;
                padding: 5px;
                border-radius: 5px;
            }
            .error-type-header {
                background-color: #f1f8ff;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
                font-weight: bold;
            }
            .error-category {
                border-left: 3px solid #2196F3;
                padding-left: 10px;
                margin: 5px 0;
            }
        </style>
        """, unsafe_allow_html=True)
        
        build_categories = all_categories.get("build", [])
        checkstyle_categories = all_categories.get("checkstyle", [])
        
        # Build errors section
        st.markdown("<div class='error-type-header'>Build Errors</div>", unsafe_allow_html=True)
        
        # Create a multi-column layout for build errors
        build_cols = st.columns(2)
        
        # Split the build categories into two columns
        half_length = len(build_categories) // 2
        for i, col in enumerate(build_cols):
            start_idx = i * half_length
            end_idx = start_idx + half_length if i == 0 else len(build_categories)
            
            with col:
                for category in build_categories[start_idx:end_idx]:
                    # Create a unique key for this category
                    category_key = f"build_{category}"
                    
                    # Check if category is selected
                    is_selected = st.checkbox(
                        category,
                        key=category_key,
                        value=category in st.session_state.selected_error_categories["build"]
                    )
                    
                    # Update selection state
                    if is_selected:
                        if category not in st.session_state.selected_error_categories["build"]:
                            st.session_state.selected_error_categories["build"].append(category)
                    else:
                        if category in st.session_state.selected_error_categories["build"]:
                            st.session_state.selected_error_categories["build"].remove(category)
        
        # Checkstyle errors section
        st.markdown("<div class='error-type-header'>Checkstyle Errors</div>", unsafe_allow_html=True)
        
        # Create a multi-column layout for checkstyle errors
        checkstyle_cols = st.columns(2)
        
        # Split the checkstyle categories into two columns
        half_length = len(checkstyle_categories) // 2
        for i, col in enumerate(checkstyle_cols):
            start_idx = i * half_length
            end_idx = start_idx + half_length if i == 0 else len(checkstyle_categories)
            
            with col:
                for category in checkstyle_categories[start_idx:end_idx]:
                    # Create a unique key for this category
                    category_key = f"checkstyle_{category}"
                    
                    # Check if category is selected
                    is_selected = st.checkbox(
                        category,
                        key=category_key,
                        value=category in st.session_state.selected_error_categories["checkstyle"]
                    )
                    
                    # Update selection state
                    if is_selected:
                        if category not in st.session_state.selected_error_categories["checkstyle"]:
                            st.session_state.selected_error_categories["checkstyle"].append(category)
                    else:
                        if category in st.session_state.selected_error_categories["checkstyle"]:
                            st.session_state.selected_error_categories["checkstyle"].remove(category)
        
        # Selection summary
        build_selected = st.session_state.selected_error_categories["build"]
        checkstyle_selected = st.session_state.selected_error_categories["checkstyle"]
        
        st.write("### Selected Categories")
        if not build_selected and not checkstyle_selected:
            st.warning("No categories selected. Default categories will be used.")
        else:
            if build_selected:
                st.write("Build Error Categories:")
                for category in build_selected:
                    st.markdown(f"<div class='error-category'>{category}</div>", unsafe_allow_html=True)
            
            if checkstyle_selected:
                st.write("Checkstyle Error Categories:")
                for category in checkstyle_selected:
                    st.markdown(f"<div class='error-category'>{category}</div>", unsafe_allow_html=True)
        
        return st.session_state.selected_error_categories
    
    def render_simple_mode(self) -> List[str]:
        """
        Render a simplified problem area selection UI.
        
        Returns:
            List of selected problem areas
        """
        # Initialize selected problem areas if not in session state
        if "problem_areas" not in st.session_state:
            st.session_state.problem_areas = ["Style", "Logical", "Performance"]
        
        # Let user select problem areas
        problem_areas = st.multiselect(
            "Problem Areas",
            ["Style", "Logical", "Performance", "Security", "Design"],
            default=st.session_state.problem_areas,
            key="problem_areas_select"
        )
        
        # Update session state
        st.session_state.problem_areas = problem_areas
        
        return problem_areas
    
    def render_mode_selector(self) -> str:
        """
        Render the mode selector UI.
        
        Returns:
            Selected mode ("standard" or "advanced")
        """
        # Error selection mode toggle
        error_mode = st.radio(
            "Error Selection Mode",
            options=["Standard (by problem areas)", "Advanced (by specific error categories)"],
            index=0 if st.session_state.error_selection_mode == "standard" else 1,
            key="error_mode_select"
        )
        
        # Update error selection mode
        if "Standard" in error_mode and st.session_state.error_selection_mode != "standard":
            st.session_state.error_selection_mode = "standard"
            # Reset selected categories
            st.session_state.selected_error_categories = {"build": [], "checkstyle": []}
        elif "Advanced" in error_mode and st.session_state.error_selection_mode != "advanced":
            st.session_state.error_selection_mode = "advanced"
        
        return st.session_state.error_selection_mode
    
    def render_code_params(self) -> Dict[str, str]:
        """
        Render code generation parameters UI.
        
        Returns:
            Dictionary with code generation parameters
        """
        # Initialize parameters if not in session state
        if "difficulty_level" not in st.session_state:
            st.session_state.difficulty_level = "Medium"
        if "code_length" not in st.session_state:
            st.session_state.code_length = "Medium"
        
        # Let user select difficulty level and code length
        difficulty_level = st.select_slider(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard"],
            value=st.session_state.difficulty_level,
            key="difficulty_level_select"
        )
        
        code_length = st.select_slider(
            "Code Length",
            options=["Short", "Medium", "Long"],
            value=st.session_state.code_length,
            key="code_length_select"
        )
        
        # Update session state
        st.session_state.difficulty_level = difficulty_level
        st.session_state.code_length = code_length
        
        return {
            "difficulty_level": difficulty_level.lower(),
            "code_length": code_length.lower()
        }

class CodeDisplayUI:
    """
    UI Component for displaying Java code snippets.
    
    This class handles displaying Java code snippets with syntax highlighting,
    line numbers, and optional instructor view.
    """
    
    def __init__(self):
        """Initialize the CodeDisplayUI component."""
        pass
    
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
        
        # INSTRUCTOR VIEW: Show known problems if provided
        if known_problems:
            if st.checkbox("Show Known Problems (Instructor View)", value=False):
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
            st.header(
                f"Submit Your Code Review "
                f"<span class='iteration-badge'>Attempt {iteration_count} of "
                f"{max_iterations}</span>", 
                unsafe_allow_html=True
            )
        else:
            st.header("Submit Your Code Review")
        
        # Display targeted guidance if available (for iterations after the first)
        if targeted_guidance and iteration_count > 1:
            st.markdown(
                f'<div class="guidance-box">'
                f'<h4>Review Guidance</h4>'
                f'{targeted_guidance}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Show previous attempt results if available
            if review_analysis:
                st.markdown(
                    f'<div class="warning-box">'
                    f'<h4>Previous Attempt Results</h4>'
                    f'You identified {review_analysis.get("identified_count", 0)} of '
                    f'{review_analysis.get("total_problems", 0)} issues '
                    f'({review_analysis.get("identified_percentage", 0):.1f}%). '
                    f'Can you find more issues in this attempt?'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        st.subheader("Your Review:")
        st.write("Please review the code above and identify any issues or problems:")
        
        # Create a unique key for the text area
        text_area_key = f"student_review_input_{iteration_count}"
        
        # Get or update the student review
        student_review_input = st.text_area(
            "Enter your review comments here",
            value=student_review,
            height=200,
            key=text_area_key
        )
        
        # Submit button
        submit_text = "Submit Review" if iteration_count == 1 else f"Submit Review (Attempt {iteration_count} of {max_iterations})"
        
        if st.button(submit_text, type="primary"):
            if not student_review_input.strip():
                st.warning("Please enter your review before submitting.")
            elif on_submit_callback:
                on_submit_callback(student_review_input)

class FeedbackDisplayUI:
    """
    UI Component for displaying feedback on student reviews.
    
    This class handles displaying analysis results, review history,
    and feedback on student reviews.
    """
    
    def __init__(self):
        """Initialize the FeedbackDisplayUI component."""
        pass
    
    def render_results(self, 
                      comparison_report: str = None,
                      review_summary: str = None,
                      review_analysis: Dict[str, Any] = None,
                      review_history: List[Dict[str, Any]] = None,
                      on_reset_callback: Callable[[], None] = None) -> None:
        """
        Render the analysis results and feedback.
        
        Args:
            comparison_report: Comparison report text
            review_summary: Review summary text
            review_analysis: Analysis of student review
            review_history: History of review iterations
            on_reset_callback: Callback function when reset button is clicked
        """
        if not comparison_report and not review_summary and not review_analysis:
            st.info("No analysis results available. Please submit your review in the 'Submit Review' tab first.")
            return
        
        # Display the comparison report
        if comparison_report:
            st.subheader("Educational Feedback:")
            st.markdown(
                f'<div class="comparison-report">{comparison_report}</div>',
                unsafe_allow_html=True
            )
        
        # Show review history in an expander if there are multiple iterations
        if review_history and len(review_history) > 1:
            with st.expander("Review History", expanded=False):
                st.write("Your review attempts:")
                
                for review in review_history:
                    review_analysis = review.get("review_analysis", {})
                    iteration = review.get("iteration_number", 0)
                    
                    st.markdown(
                        f'<div class="review-history-item">'
                        f'<h4>Attempt {iteration}</h4>'
                        f'<p>Found {review_analysis.get("identified_count", 0)} of '
                        f'{review_analysis.get("total_problems", 0)} issues '
                        f'({review_analysis.get("accuracy_percentage", 0):.1f}% accuracy)</p>'
                        f'<details>'
                        f'<summary>View this review</summary>'
                        f'<pre>{review.get("student_review", "")}</pre>'
                        f'</details>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        # Display analysis details in an expander
        if review_summary or review_analysis:
            with st.expander("Detailed Analysis", expanded=False):
                # Display review summary
                if review_summary:
                    st.subheader("Review Summary:")
                    st.markdown(review_summary)
                
                # Display review analysis
                if review_analysis:
                    st.subheader("Review Analysis:")
                    accuracy = review_analysis.get("accuracy_percentage", 0)
                    identified_percentage = review_analysis.get("identified_percentage", 0)
                    
                    st.write(f"**Accuracy:** {accuracy:.1f}%")
                    st.write(f"**Problems Identified:** {identified_percentage:.1f}% of all issues")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Correctly Identified Issues:**")
                        for issue in review_analysis.get("identified_problems", []):
                            st.write(f"✓ {issue}")
                    
                    with col2:
                        st.write("**Missed Issues:**")
                        for issue in review_analysis.get("missed_problems", []):
                            st.write(f"✗ {issue}")
                    
                    # Display false positives if any
                    if review_analysis.get("false_positives"):
                        st.write("**False Positives:**")
                        for issue in review_analysis.get("false_positives"):
                            st.write(f"⚠ {issue}")
        
        # Start over button
        if st.button("Start New Review", type="primary"):
            # Call the reset callback if provided
            if on_reset_callback:
                on_reset_callback()
            else:
                # Reset session state
                for key in list(st.session_state.keys()):
                    # Keep error selection mode and categories
                    if key not in ["error_selection_mode", "selected_error_categories"]:
                        del st.session_state[key]
                
                # Rerun the app
                st.rerun()