"""
Streamlit UI for Java Peer Review Training System.

This module provides a Streamlit web interface for the Java code review training system
integrated with the LangGraph workflow implementation.
"""

import streamlit as st
import os
import logging
import time
import sys
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging to see any issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add debug prints to help identify loading issues
print("Starting app imports...")

try:
    # Import the LangGraph implementation
    from langgraph_implementation import (
        CodeReviewAgent, 
        CodeReviewGraph, 
        CodeReviewState, 
        ErrorRepository
    )
    print("Successfully imported LangGraph implementation")
except Exception as e:
    print(f"Error importing LangGraph implementation: {str(e)}")
    traceback.print_exc()
    st.error(f"Failed to load LangGraph implementation: {str(e)}")
    st.stop()

print("Imports completed")

# Set page config
st.set_page_config(
    page_title="Java Code Review Training System",
    page_icon="☕",  # Java coffee cup icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            margin-bottom: 0.5rem;
        }
        .stTextArea textarea {
            font-family: monospace;
        }
        .code-block {
            background-color: #f0f0f0;
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .problem-list {
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
            padding: 10px;
            margin: 10px 0;
        }
        .student-review {
            background-color: #f8f9fa;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 10px 0;
        }
        .review-analysis {
            background-color: #f8f9fa;
            border-left: 4px solid #FF9800;
            padding: 10px;
            margin: 10px 0;
        }
        .comparison-report {
            background-color: #f8f9fa;
            border-left: 4px solid #9C27B0;
            padding: 10px;
            margin: 10px 0;
        }
        .model-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f8f9fa;
        }
        .status-ok {
            color: #4CAF50;
            font-weight: bold;
        }
        .status-warning {
            color: #FF9800;
            font-weight: bold;
        }
        .status-error {
            color: #F44336;
            font-weight: bold;
        }
        .guidance-box {
            background-color: #e8f4f8;
            border-left: 4px solid #03A9F4;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .iteration-badge {
            background-color: #E1F5FE;
            color: #0288D1;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            display: inline-block;
            margin-left: 10px;
        }
        .feedback-box {
            background-color: #E8F5E9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .warning-box {
            background-color: #FFF8E1;
            border-left: 4px solid #FFC107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .review-history-item {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            background-color: #fafafa;
        }
        .tab-content {
            padding: 1rem;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
    </style>
""", unsafe_allow_html=True)

def initialize_session():
    """Initialize session state and components."""
    try:
        print("Initializing session state")
        
        # Initialize general state
        if "code_state" not in st.session_state:
            st.session_state.code_state = None
        if "error" not in st.session_state:
            st.session_state.error = None
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = 0
        if "max_iterations" not in st.session_state:
            st.session_state.max_iterations = 3
        
        # Initialize error repository
        if "error_repository" not in st.session_state:
            print("Initializing error repository")
            st.session_state.error_repository = ErrorRepository()
        
        # Initialize error selection mode and categories
        if "error_selection_mode" not in st.session_state:
            st.session_state.error_selection_mode = "standard"
        if "selected_error_categories" not in st.session_state:
            st.session_state.selected_error_categories = {
                "build": [],
                "checkstyle": []
            }
        
        # Initialize agent if not already done
        if "agent" not in st.session_state:
            print("Initializing agent")
            with st.spinner("Initializing system..."):
                st.session_state.agent = CodeReviewAgent()
                print("Agent initialized successfully")
                
        print("Session initialization complete")
    except Exception as e:
        print(f"Error in initialize_session: {str(e)}")
        traceback.print_exc()
        st.error(f"Failed to initialize session: {str(e)}")

def check_ollama_status():
    """Check the status of Ollama and required models."""
    try:
        import requests
        print("Checking Ollama status")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Check Ollama connection
        connection_status = False
        try:
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
            connection_status = response.status_code == 200
            print(f"Ollama connection status: {connection_status}")
        except Exception as e:
            print(f"Error connecting to Ollama: {str(e)}")
            connection_status = False
        
        # Check if default model is available
        default_model = os.getenv("DEFAULT_MODEL", "llama3:1b")
        default_model_available = False
        
        if connection_status:
            try:
                response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    default_model_available = any(model["name"] == default_model for model in models)
                    print(f"Default model available: {default_model_available}")
            except Exception as e:
                print(f"Error checking model availability: {str(e)}")
                default_model_available = False
        
        # Check if all role-specific models are configured in environment
        required_models = ["GENERATIVE_MODEL", "REVIEW_MODEL", "SUMMARY_MODEL", "COMPARE_MODEL"]
        all_models_configured = all(os.getenv(model) for model in required_models)
        print(f"All models configured: {all_models_configured}")
        
        return {
            "ollama_running": connection_status,
            "default_model_available": default_model_available,
            "all_models_configured": all_models_configured
        }
    except Exception as e:
        print(f"Error checking Ollama status: {str(e)}")
        traceback.print_exc()
        return {
            "ollama_running": False,
            "default_model_available": False,
            "all_models_configured": False,
            "error": str(e)
        }

def render_sidebar():
    """Render the sidebar with status and settings."""
    with st.sidebar:
        st.header("Model Settings")
        
        # Show status
        status = check_ollama_status()
        st.subheader("System Status")
        
        if status["ollama_running"]:
            st.markdown(f"- Ollama: <span class='status-ok'>Running</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"- Ollama: <span class='status-error'>Not Running</span>", unsafe_allow_html=True)
            st.error("Ollama is not running. Please start it first.")
            
            # Troubleshooting information
            with st.expander("Troubleshooting"):
                st.markdown("""
                1. **Check if Ollama is running:**
                   ```bash
                   curl http://localhost:11434/api/tags
                   ```
                   
                2. **Make sure the model is downloaded:**
                   ```bash
                   ollama pull llama3:1b
                   ```
                   
                3. **Start Ollama:**
                   - On Linux/Mac: `ollama serve`
                   - On Windows: Start the Ollama application
                """)
        
        if status["default_model_available"]:
            st.markdown(f"- Default model: <span class='status-ok'>Available</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"- Default model: <span class='status-warning'>Not Found</span>", unsafe_allow_html=True)
            if status["ollama_running"]:
                default_model = os.getenv("DEFAULT_MODEL", "llama3:1b")
                st.warning(f"Default model '{default_model}' not found. You need to pull it.")
                if st.button("Pull Default Model"):
                    with st.spinner(f"Pulling {default_model}..."):
                        try:
                            import requests
                            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                            response = requests.post(
                                f"{ollama_base_url}/api/pull",
                                json={"name": default_model, "stream": False},
                                timeout=60
                            )
                            if response.status_code == 200:
                                st.success("Default model pulled successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to pull default model.")
                        except Exception as e:
                            st.error(f"Error pulling model: {str(e)}")
        
        if status["all_models_configured"]:
            st.markdown(f"- Model configuration: <span class='status-ok'>Complete</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"- Model configuration: <span class='status-warning'>Incomplete</span>", unsafe_allow_html=True)
        
        # Sidebar section for iterative review settings
        st.markdown("---")
        st.header("Review Settings")
        
        max_iterations = st.slider(
            "Maximum Review Attempts:",
            min_value=1,
            max_value=5,
            value=st.session_state.max_iterations,
            help="Maximum number of review attempts allowed before final evaluation"
        )
        
        # Update max iterations in session state
        if max_iterations != st.session_state.max_iterations:
            st.session_state.max_iterations = max_iterations
            
            # Update max iterations in code state if already generated
            if st.session_state.code_state:
                st.session_state.code_state.max_iterations = max_iterations

def render_error_selector():
    """Render the error category selector UI."""
    st.subheader("Select Error Types")
    st.info("Choose the types of errors to include in the generated Java code.")
    
    # Error selection mode toggle
    error_mode = st.radio(
        "Error Selection Mode",
        options=["Standard (by problem areas)", "Advanced (by specific error categories)"],
        index=0 if st.session_state.error_selection_mode == "standard" else 1,
        key="error_mode_selector"
    )
    
    # Update error selection mode
    if "Standard" in error_mode and st.session_state.error_selection_mode != "standard":
        st.session_state.error_selection_mode = "standard"
        # Reset selected categories
        st.session_state.selected_error_categories = {"build": [], "checkstyle": []}
    elif "Advanced" in error_mode and st.session_state.error_selection_mode != "advanced":
        st.session_state.error_selection_mode = "advanced"
    
    # Get code generation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        difficulty_level = st.select_slider(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard"],
            value="Medium",
            key="difficulty_slider"
        )
    
    with col2:
        code_length = st.select_slider(
            "Code Length",
            options=["Short", "Medium", "Long"],
            value="Medium",
            key="length_slider"
        )
    
    # Error selection UI
    if st.session_state.error_selection_mode == "standard":
        # In standard mode, use simple problem area selection
        problem_areas = st.multiselect(
            "Problem Areas",
            ["Style", "Logical", "Performance", "Security", "Design"],
            default=["Style", "Logical", "Performance"],
            key="problem_areas_selector"
        )
        
        # Map problem areas to categories for the backend
        selected_categories = {
            "build": [],
            "checkstyle": []
        }
        
        # Map problem areas to error categories
        area_mapping = {
            "Style": {
                "build": [],
                "checkstyle": ["NamingConventionChecks", "WhitespaceAndFormattingChecks", "JavadocChecks"]
            },
            "Logical": {
                "build": ["LogicalErrors"],
                "checkstyle": []
            },
            "Performance": {
                "build": ["RuntimeErrors"],
                "checkstyle": ["MetricsChecks"]
            },
            "Security": {
                "build": ["RuntimeErrors", "LogicalErrors"],
                "checkstyle": ["CodeQualityChecks"]
            },
            "Design": {
                "build": ["LogicalErrors"],
                "checkstyle": ["MiscellaneousChecks", "FileStructureChecks", "BlockChecks"]
            }
        }
        
        # Build selected categories from problem areas
        for area in problem_areas:
            if area in area_mapping:
                mapping = area_mapping[area]
                for category in mapping["build"]:
                    if category not in selected_categories["build"]:
                        selected_categories["build"].append(category)
                for category in mapping["checkstyle"]:
                    if category not in selected_categories["checkstyle"]:
                        selected_categories["checkstyle"].append(category)
        
        st.session_state.selected_error_categories = selected_categories
    else:
        # In advanced mode, let user select specific error categories
        all_categories = st.session_state.error_repository.get_all_categories()
        
        # Build errors section
        st.markdown("#### Build Errors")
        build_categories = all_categories.get("build", [])
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
        st.markdown("#### Checkstyle Errors")
        checkstyle_categories = all_categories.get("checkstyle", [])
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
    
    return {
        "difficulty_level": difficulty_level.lower(),
        "code_length": code_length.lower(),
        "selected_error_categories": st.session_state.selected_error_categories
    }

def generate_code_problem(params):
    """Generate a code problem with progress indicator."""
    # Show progress during generation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing Java code generation...")
        progress_bar.progress(20)
        
        # Create initial state with parameters
        print("Creating initial state for code generation")
        code_state = CodeReviewGraph(
            current_state=CodeReviewState.GENERATE,
            max_iterations=st.session_state.max_iterations
        )
        
        status_text.text("Generating Java code problem...")
        progress_bar.progress(40)
        
        # Generate code problem
        print("Calling agent.generate_code_problem()")
        result = st.session_state.agent.generate_code_problem()
        print(f"Code generation complete, state: {result.current_state}")
        
        progress_bar.progress(90)
        status_text.text("Finalizing results...")
        time.sleep(0.5)
        
        # Check for errors
        if result.current_state == CodeReviewState.ERROR:
            progress_bar.empty()
            status_text.empty()
            st.session_state.error = result.error_message
            print(f"Code generation error: {result.error_message}")
            return False
        
        # Update session state
        st.session_state.code_state = result
        st.session_state.active_tab = 1  # Move to the review tab
        st.session_state.error = None
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        print("Code generation successful")
        return True
        
    except Exception as e:
        print(f"Error generating code problem: {str(e)}")
        traceback.print_exc()
        progress_bar.empty()
        status_text.empty()
        st.session_state.error = f"Error generating code problem: {str(e)}"
        return False

def process_student_review(student_review):
    """Process a student review with progress indicator."""
    # Show progress during analysis
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Processing student review...")
        progress_bar.progress(20)
        
        status_text.text("Analyzing your review...")
        progress_bar.progress(40)
        
        # Submit the review
        print("Submitting student review for analysis")
        result = st.session_state.agent.submit_review(st.session_state.code_state, student_review)
        print(f"Review analysis complete, state: {result.current_state}")
        
        status_text.text("Generating feedback...")
        progress_bar.progress(70)
        
        # Check for errors
        if result.current_state == CodeReviewState.ERROR:
            progress_bar.empty()
            status_text.empty()
            st.session_state.error = result.error_message
            print(f"Review analysis error: {result.error_message}")
            return False
        
        # Update session state
        st.session_state.code_state = result
        
        # If complete, move to the analysis tab
        if result.current_state == CodeReviewState.COMPLETE:
            st.session_state.active_tab = 2  # Move to the analysis tab
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        print("Review processing successful")
        return True
        
    except Exception as e:
        print(f"Error processing student review: {str(e)}")
        traceback.print_exc()
        progress_bar.empty()
        status_text.empty()
        st.session_state.error = f"Error processing student review: {str(e)}"
        return False

def render_code_display(code_snippet, known_problems=None):
    """Render a code snippet with optional known problems for instructor view."""
    if not code_snippet:
        st.info("No code generated yet. Use the 'Generate Code Problem' tab to create a Java code snippet.")
        return
    
    st.subheader("Java Code to Review:")
    
    # Add line numbers to the code snippet
    lines = code_snippet.splitlines()
    max_line_num = len(lines)
    padding = len(str(max_line_num))
    
    # Create a list of lines with line numbers
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        # Format line number with consistent padding
        line_num = str(i).rjust(padding)
        numbered_lines.append(f"{line_num} | {line}")
    
    numbered_code = "\n".join(numbered_lines)
    st.code(numbered_code, language="java")
    
    # INSTRUCTOR VIEW: Show known problems if provided
    if known_problems:
        if st.checkbox("Show Known Problems (Instructor View)", value=False, key="show_problems_checkbox"):
            st.subheader("Known Problems:")
            for i, problem in enumerate(known_problems, 1):
                st.markdown(f"{i}. {problem}")

def render_review_input(student_review=""):
    """Render a text area for student review input with guidance."""
    code_state = st.session_state.code_state
    
    # Show iteration badge if not the first iteration
    if code_state.iteration_count > 1:
        st.header(
            f"Submit Your Code Review "
            f"<span class='iteration-badge'>Attempt {code_state.iteration_count} of "
            f"{code_state.max_iterations}</span>", 
            unsafe_allow_html=True
        )
    else:
        st.header("Submit Your Code Review")
    
    # Display targeted guidance if available (for iterations after the first)
    if code_state.targeted_guidance and code_state.iteration_count > 1:
        st.markdown(
            f'<div class="guidance-box">'
            f'<h4>Review Guidance</h4>'
            f'{code_state.targeted_guidance}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Show previous attempt results if available
        if code_state.review_analysis:
            st.markdown(
                f'<div class="warning-box">'
                f'<h4>Previous Attempt Results</h4>'
                f'You identified {code_state.review_analysis.identified_count} of '
                f'{code_state.review_analysis.total_problems} issues '
                f'({code_state.review_analysis.identified_percentage:.1f}%). '
                f'Can you find more issues in this attempt?'
                f'</div>',
                unsafe_allow_html=True
            )
    
    st.subheader("Your Review:")
    st.write("Please review the code above and identify any issues or problems:")
    
    # Create a unique key for the text area
    text_area_key = f"student_review_input_{code_state.iteration_count}"
    
    # Get or update the student review
    student_review_input = st.text_area(
        "Enter your review comments here",
        value=student_review,
        height=200,
        key=text_area_key
    )
    
    # Submit button
    submit_text = "Submit Review" if code_state.iteration_count == 1 else f"Submit Review (Attempt {code_state.iteration_count} of {code_state.max_iterations})"
    
    if st.button(submit_text, type="primary", key="submit_review_button"):
        if not student_review_input.strip():
            st.warning("Please enter your review before submitting.")
        else:
            with st.spinner("Analyzing your review..."):
                success = process_student_review(student_review_input)
                if success:
                    st.rerun()

def render_results():
    """Render the analysis results and feedback."""
    code_state = st.session_state.code_state
    
    if not code_state or not (code_state.comparison_report or code_state.review_summary):
        st.info("No analysis results available. Please submit your review in the 'Submit Review' tab first.")
        return
    
    # Display the comparison report
    if code_state.comparison_report:
        st.subheader("Educational Feedback:")
        st.markdown(
            f'<div class="comparison-report">{code_state.comparison_report}</div>',
            unsafe_allow_html=True
        )
    
    # Show review history in an expander if there are multiple iterations
    if code_state.review_history and len(code_state.review_history) > 1:
        with st.expander("Review History", expanded=False):
            st.write("Your review attempts:")
            
            for review in code_state.review_history:
                review_analysis = review.review_analysis
                iteration = review.iteration_number
                
                st.markdown(
                    f'<div class="review-history-item">'
                    f'<h4>Attempt {iteration}</h4>'
                    f'<p>Found {review_analysis.identified_count} of '
                    f'{review_analysis.total_problems} issues '
                    f'({review_analysis.accuracy_percentage:.1f}% accuracy)</p>'
                    f'<details>'
                    f'<summary>View this review</summary>'
                    f'<pre>{review.student_review}</pre>'
                    f'</details>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    # Display analysis details in an expander
    if code_state.review_summary or code_state.review_analysis:
        with st.expander("Detailed Analysis", expanded=False):
            # Display review summary
            if code_state.review_summary:
                st.subheader("Review Summary:")
                st.markdown(code_state.review_summary)
            
            # Display review analysis
            if code_state.review_analysis:
                st.subheader("Review Analysis:")
                accuracy = code_state.review_analysis.accuracy_percentage
                identified_percentage = code_state.review_analysis.identified_percentage
                
                st.write(f"**Accuracy:** {accuracy:.1f}%")
                st.write(f"**Problems Identified:** {identified_percentage:.1f}% of all issues")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Correctly Identified Issues:**")
                    for issue in code_state.review_analysis.identified_problems:
                        st.write(f"✓ {issue}")
                
                with col2:
                    st.write("**Missed Issues:**")
                    for issue in code_state.review_analysis.missed_problems:
                        st.write(f"✗ {issue}")
                
                # Display false positives if any
                if code_state.review_analysis.false_positives:
                    st.write("**False Positives:**")
                    for issue in code_state.review_analysis.false_positives:
                        st.write(f"⚠ {issue}")
    
    # Start over button
    if st.button("Start New Review", type="primary", key="start_new_review_button"):
        try:
            # Reset the agent
            print("Resetting session")
            st.session_state.agent.reset_session()
            
            # Reset session state
            st.session_state.code_state = None
            st.session_state.error = None
            st.session_state.active_tab = 0
            
            # Rerun the app
            st.rerun()
        except Exception as e:
            print(f"Error resetting session: {str(e)}")
            traceback.print_exc()
            st.error(f"Error resetting session: {str(e)}")

def main():
    """Main application function."""
    try:
        print("Starting main function")
        # Initialize session state
        initialize_session()
        
        # Header
        st.title("Java Code Review Training System")
        st.markdown("### Train your Java code review skills with AI-generated exercises")
        
        # Render sidebar
        render_sidebar()
        
        # Display error message if there's an error
        if st.session_state.error:
            st.error(f"Error: {st.session_state.error}")
            if st.button("Clear Error", key="clear_error_button"):
                st.session_state.error = None
                st.rerun()
        
        # Create tabs for different steps of the workflow
        tabs = st.tabs(["1. Generate Code Problem", "2. Submit Review", "3. Analysis & Feedback"])
        
        # Set the active tab based on session state
        active_tab = st.session_state.active_tab
        
        with tabs[0]:
            st.header("Generate Java Code Problem")
            
            # Render error selector
            params = render_error_selector()
            
            # Generate button
            generate_button = st.button("Generate Java Code Problem", type="primary", key="generate_button")
            
            if generate_button:
                with st.spinner("Generating Java code with intentional issues..."):
                    success = generate_code_problem(params)
                    
                    if success:
                        st.rerun()
            
            # Display existing code if available
            if st.session_state.code_state and st.session_state.code_state.code_snippet:
                render_code_display(
                    st.session_state.code_state.code_snippet,
                    st.session_state.code_state.known_problems
                )
        
        with tabs[1]:
            # Student review input and submission
            if not st.session_state.code_state or not st.session_state.code_state.code_snippet:
                st.info("Please generate a code problem first in the 'Generate Code Problem' tab.")
            else:
                # Review display and submission
                render_code_display(st.session_state.code_state.code_snippet)
                
                # Render review input
                render_review_input(student_review=st.session_state.code_state.student_review)
        
        with tabs[2]:
            st.header("Analysis & Feedback")
            
            if not st.session_state.code_state or not st.session_state.code_state.comparison_report:
                st.info("Please submit your review in the 'Submit Review' tab first.")
            else:
                # Display feedback results
                render_results()
                
        print("Main function completed")
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()
        st.error(f"Application error: {str(e)}")
        st.write("Please check the console logs for more details.")

# Simple test function that can be used to verify Streamlit is working
def test_function():
    st.title("Simple Test App")
    st.write("If you can see this, Streamlit is working properly.")
    
    if st.button("Click Me"):
        st.success("Button clicked!")

if __name__ == "__main__":
    # Uncomment this line to run a simple test instead of the main application
    # test_function()
    
    # Comment this line if running the test function
    main()