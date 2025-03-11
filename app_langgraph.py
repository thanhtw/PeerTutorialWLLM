"""
Java Peer Code Review Training System - LangGraph Version

This module provides a Streamlit web interface for the Java code review training system
using LangGraph for workflow management.
"""

import streamlit as st
import sys
import os
import logging
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import LangGraph components
import sys
import os

# Add the current directory to the path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from langgraph_workflow import JavaCodeReviewGraph
from state_schema import WorkflowState

# Import UI components
from ui.ui_components import ErrorSelectorUI, CodeDisplayUI, FeedbackDisplayUI
from llm_manager import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Java Code Review Training System",
    page_icon="☕",  # Java coffee cup icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling (same as original)
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

def check_ollama_status() -> Dict[str, bool]:
    """
    Check the status of Ollama and required models.
    
    Returns:
        Dictionary with status information
    """
    llm_manager = LLMManager()
    
    # Check Ollama connection
    connection_status, _ = llm_manager.check_ollama_connection()
    
    # Check if default model is available
    default_model_available = False
    if connection_status:
        default_model = llm_manager.default_model
        default_model_available = llm_manager.check_model_availability(default_model)
    
    # Check if all role-specific models are configured in environment
    required_models = ["GENERATIVE_MODEL", "REVIEW_MODEL", "SUMMARY_MODEL", "COMPARE_MODEL"]
    all_models_configured = all(os.getenv(model) for model in required_models)
    
    return {
        "ollama_running": connection_status,
        "default_model_available": default_model_available,
        "all_models_configured": all_models_configured
    }

def init_session_state():
    """Initialize session state variables."""
    # Initialize workflow state if not present
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = WorkflowState()
    
    # Initialize active tab if not present
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Initialize error state if not present
    if 'error' not in st.session_state:
        st.session_state.error = None

def generate_code_problem(workflow: JavaCodeReviewGraph, 
                         params: Dict[str, str], 
                         selected_error_categories: Dict[str, List[str]]):
    """
    Generate a code problem with progress indicator.
    
    Args:
        workflow: JavaCodeReviewGraph instance
        params: Code generation parameters
        selected_error_categories: Selected error categories
    """
    # Show progress during generation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Generating Java code problem...")
        progress_bar.progress(30)
        
        # Update workflow state with generation parameters
        state = st.session_state.workflow_state
        state.code_length = params["code_length"]
        state.difficulty_level = params["difficulty_level"]
        state.selected_error_categories = selected_error_categories
        
        # Run the generation node
        progress_bar.progress(60)
        updated_state = workflow.generate_code_node(state)
        
        progress_bar.progress(90)
        status_text.text("Finalizing results...")
        time.sleep(0.5)
        
        # Check for errors
        if updated_state.error:
            progress_bar.empty()
            status_text.empty()
            st.session_state.error = updated_state.error
            return False
        
        # Update session state
        st.session_state.workflow_state = updated_state
        st.session_state.active_tab = 1  # Move to the review tab
        st.session_state.error = None
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating code problem: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        st.session_state.error = f"Error generating code problem: {str(e)}"
        return False

def process_student_review(workflow: JavaCodeReviewGraph, student_review: str):
    """
    Process a student review with progress indicator.
    
    Args:
        workflow: JavaCodeReviewGraph instance
        student_review: Student review text
    """
    # Show progress during analysis
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Processing student review...")
        progress_bar.progress(20)
        
        # Get current state
        state = st.session_state.workflow_state
        
        # Store the current review in session state for display consistency
        current_iteration = state.current_iteration
        st.session_state[f"submitted_review_{current_iteration}"] = student_review
        
        # Submit the review and update the state
        status_text.text("Analyzing your review...")
        progress_bar.progress(50)
        updated_state = workflow.submit_review(state, student_review)
        
        # Check for errors
        if updated_state.error:
            progress_bar.empty()
            status_text.empty()
            st.session_state.error = updated_state.error
            return False
        
        # Update session state
        st.session_state.workflow_state = updated_state
        
        # Check if we should generate summary
        if workflow.should_continue_review(updated_state) == "generate_summary":
            status_text.text("Generating final feedback...")
            progress_bar.progress(75)
            
            # Run the summary generation node
            final_state = workflow.generate_summary_node(updated_state)
            st.session_state.workflow_state = final_state
            
            # Move to the analysis tab
            st.session_state.active_tab = 2
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Force a rerun to update the UI
        st.rerun()
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing student review: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        st.session_state.error = f"Error processing student review: {str(e)}"
        return False

def render_sidebar(llm_manager: LLMManager):
    """
    Render the sidebar with status and settings.
    
    Args:
        llm_manager: LLMManager instance
    """
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
                st.warning(f"Default model '{llm_manager.default_model}' not found. You need to pull it.")
                if st.button("Pull Default Model"):
                    with st.spinner(f"Pulling {llm_manager.default_model}..."):
                        if llm_manager.download_ollama_model(llm_manager.default_model):
                            st.success("Default model pulled successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to pull default model.")
        
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
            value=st.session_state.workflow_state.max_iterations,
            help="Maximum number of review attempts allowed before final evaluation"
        )
        
        # Update max iterations in workflow state
        if max_iterations != st.session_state.workflow_state.max_iterations:
            st.session_state.workflow_state.max_iterations = max_iterations

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Initialize LLM manager and workflow
    llm_manager = LLMManager()
    workflow = JavaCodeReviewGraph(llm_manager)
    
    # Initialize UI components
    error_selector_ui = ErrorSelectorUI()
    code_display_ui = CodeDisplayUI()
    feedback_display_ui = FeedbackDisplayUI()
    
    # Header
    st.title("Java Code Review Training System")
    st.markdown("### Train your Java code review skills with AI-generated exercises")
    
    # Render sidebar
    render_sidebar(llm_manager)
    
    # Display error message if there's an error
    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()
    
    # Create tabs for different steps of the workflow
    tabs = st.tabs(["1. Generate Code Problem", "2. Submit Review", "3. Analysis & Feedback"])
    
    # Set the active tab based on session state
    active_tab = st.session_state.active_tab
    
    with tabs[0]:
        st.header("Generate Java Code Problem")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fixed to Java
            st.info("This system is specialized for Java code review training.")
            
            # Select error selection mode
            mode = error_selector_ui.render_mode_selector()
            
            # Get code generation parameters
            params = error_selector_ui.render_code_params()
            
        with col2:
            # Show standard or advanced error selection based on mode
            if mode == "standard":
                # In standard mode, use simple problem area selection
                problem_areas = error_selector_ui.render_simple_mode()
                
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
            else:
                # In advanced mode, let user select specific error categories
                selected_categories = error_selector_ui.render_category_selection(
                    workflow.get_all_error_categories()
                )
        
        # Generate button
        generate_button = st.button("Generate Java Code Problem", type="primary")
        
        if generate_button:
            with st.spinner("Generating Java code with intentional issues..."):
                success = generate_code_problem(
                    workflow,
                    params,
                    selected_categories
                )
                
                if success:
                    st.rerun()
        
        # Display existing code if available
        state = st.session_state.workflow_state        
        if state.code_snippet:
            code_display_ui.render_code_display(
                state.code_snippet.code,
                state.code_snippet.known_problems
            )
    
    with tabs[1]:
        state = st.session_state.workflow_state
        
        # Student review input and submission
        if not state.code_snippet:
            st.info("Please generate a code problem first in the 'Generate Code Problem' tab.")
        else:
            # Review display and submission
            code_display_ui.render_code_display(state.code_snippet.code)
            
            # Submission callback
            def handle_review_submission(student_review):
                with st.spinner("Analyzing your review..."):
                    success = process_student_review(workflow, student_review)
                    if success:
                        st.rerun()
            
            # Get the latest review and guidance if available
            latest_review = state.review_history[-1] if state.review_history else None
            targeted_guidance = latest_review.targeted_guidance if latest_review else None
            latest_analysis = latest_review.analysis if latest_review else None
            
            # Render review input with feedback
            code_display_ui.render_review_input(
                student_review=latest_review.student_review if latest_review else "",
                on_submit_callback=handle_review_submission,
                iteration_count=state.current_iteration,
                max_iterations=state.max_iterations,
                targeted_guidance=targeted_guidance,
                review_analysis=latest_analysis
            )
    
    with tabs[2]:
        st.header("Analysis & Feedback")
        
        state = st.session_state.workflow_state
        
        if not state.comparison_report and not state.review_summary:
            st.info("Please submit your review in the 'Submit Review' tab first.")
        else:
            # Reset callback
            def handle_reset():
                # Create a new workflow state
                st.session_state.workflow_state = WorkflowState()
                
                # Reset active tab
                st.session_state.active_tab = 0
                
                # Rerun the app
                st.rerun()
            
            # Get the latest review analysis
            latest_review = state.review_history[-1] if state.review_history else None
            latest_analysis = latest_review.analysis if latest_review else None
            
            # Convert review history to the format expected by FeedbackDisplayUI
            review_history = []
            for review in state.review_history:
                review_history.append({
                    "iteration_number": review.iteration_number,
                    "student_review": review.student_review,
                    "review_analysis": review.analysis
                })
            
            # Display feedback results
            feedback_display_ui.render_results(
                comparison_report=state.comparison_report,
                review_summary=state.review_summary,
                review_analysis=latest_analysis,
                review_history=review_history,
                on_reset_callback=handle_reset
            )

if __name__ == "__main__":
    main()