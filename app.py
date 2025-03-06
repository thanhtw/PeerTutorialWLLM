"""
Refactored Java Peer Code Review Training System - Main Application

This module provides a Streamlit web interface for the Java code review training system
with an improved modular architecture, object-oriented design, and direct JSON error handling.
"""

import streamlit as st
import sys
import os
import logging
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import components from the refactored architecture
from service.agent_service import AgentService
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
    # General state
    if 'code_snippet' not in st.session_state:
        st.session_state.code_snippet = ""
    if 'known_problems' not in st.session_state:
        st.session_state.known_problems = []
    if 'student_review' not in st.session_state:
        st.session_state.student_review = ""
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "generate"
    if 'error' not in st.session_state:
        st.session_state.error = None
    
    # Iteration tracking
    if 'iteration_count' not in st.session_state:
        st.session_state.iteration_count = 1
    if 'max_iterations' not in st.session_state:
        st.session_state.max_iterations = 3
    
    # Analysis results
    if 'review_analysis' not in st.session_state:
        st.session_state.review_analysis = None
    if 'review_history' not in st.session_state:
        st.session_state.review_history = []
    if 'targeted_guidance' not in st.session_state:
        st.session_state.targeted_guidance = None
    if 'review_summary' not in st.session_state:
        st.session_state.review_summary = None
    if 'comparison_report' not in st.session_state:
        st.session_state.comparison_report = None

def generate_code_problem(agent_service: AgentService, 
                         params: Dict[str, str], 
                         selected_error_categories: Dict[str, List[str]]):
    """
    Generate a code problem with progress indicator.
    
    Args:
        agent_service: AgentService instance
        params: Code generation parameters
        selected_error_categories: Selected error categories
    """
    # Show progress during generation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Generating Java code problem...")
        progress_bar.progress(30)
        
        # Generate code problem
        result = agent_service.generate_code_with_errors(
            code_length=params["code_length"],
            difficulty_level=params["difficulty_level"],
            selected_error_categories=selected_error_categories
        )
        
        progress_bar.progress(90)
        status_text.text("Finalizing results...")
        time.sleep(0.5)
        
        # Check for errors
        if "error" in result:
            progress_bar.empty()
            status_text.empty()
            st.session_state.error = result["error"]
            return False
        
        # Update session state
        st.session_state.code_snippet = result.get("code_snippet", "")
        st.session_state.known_problems = result.get("known_problems", [])
        st.session_state.current_step = "review"
        st.session_state.active_tab = 1  # Move to the review tab
        st.session_state.iteration_count = 1
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

def process_student_review(agent_service: AgentService, student_review: str):
    """
    Process a student review with progress indicator.
    
    Args:
        agent_service: AgentService instance
        student_review: Student review text
    """
    # Show progress during analysis
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Processing student review...")
        progress_bar.progress(20)
        
        # Update the student review in session state
        st.session_state.student_review = student_review
        
        # Process the review
        status_text.text("Analyzing your review...")
        progress_bar.progress(40)
        
        result = agent_service.process_student_review(student_review)
        
        status_text.text("Generating feedback...")
        progress_bar.progress(70)
        
        # Check for errors
        if "error" in result:
            progress_bar.empty()
            status_text.empty()
            st.session_state.error = result["error"]
            return False
        
        # Update session state
        st.session_state.review_analysis = result.get("review_analysis", {})
        st.session_state.review_history = agent_service.get_review_history()
        st.session_state.targeted_guidance = result.get("targeted_guidance", "")
        st.session_state.current_step = result.get("current_step", "wait_for_review")
        
        # If complete, get summary and comparison
        if result.get("current_step") == "complete":
            st.session_state.review_summary = result.get("review_summary", "")
            st.session_state.comparison_report = result.get("comparison_report", "")
            st.session_state.active_tab = 2  # Move to the analysis tab
        else:
            # Update iteration count for next review
            st.session_state.iteration_count = result.get("iteration_count", 1)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
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
            value=st.session_state.max_iterations,
            help="Maximum number of review attempts allowed before final evaluation"
        )
        
        # Update max iterations in session state
        if max_iterations != st.session_state.max_iterations:
            st.session_state.max_iterations = max_iterations

def main():
    """Main application function."""
    # Initialize session state
    init_session_state()
    
    # Initialize services
    llm_manager = LLMManager()
    agent_service = AgentService(llm_manager)
    
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
                    agent_service.get_all_error_categories()
                )
        
        # Generate button
        generate_button = st.button("Generate Java Code Problem", type="primary")
        
        if generate_button:
            with st.spinner("Generating Java code with intentional issues..."):
                success = generate_code_problem(
                    agent_service,
                    params,
                    selected_categories
                )
                
                if success:
                    st.rerun()
        
        # Display existing code if available
        if st.session_state.code_snippet:
            code_display_ui.render_code_display(
                st.session_state.code_snippet,
                st.session_state.known_problems
            )
    
    with tabs[1]:
        # Student review input and submission
        if not st.session_state.code_snippet:
            st.info("Please generate a code problem first in the 'Generate Code Problem' tab.")
        else:
            # Review display and submission
            code_display_ui.render_code_display(st.session_state.code_snippet)
            
            # Submission callback
            def handle_review_submission(student_review):
                with st.spinner("Analyzing your review..."):
                    success = process_student_review(agent_service, student_review)
                    if success:
                        st.rerun()
            
            # Render review input with feedback
            code_display_ui.render_review_input(
                student_review=st.session_state.student_review,
                on_submit_callback=handle_review_submission,
                iteration_count=st.session_state.iteration_count,
                max_iterations=st.session_state.max_iterations,
                targeted_guidance=st.session_state.targeted_guidance,
                review_analysis=st.session_state.review_analysis
            )
    
    with tabs[2]:
        st.header("Analysis & Feedback")
        
        if not st.session_state.comparison_report and not st.session_state.review_summary:
            st.info("Please submit your review in the 'Submit Review' tab first.")
        else:
            # Reset callback
            def handle_reset():
                agent_service.reset_session()
                
                # Reset session state
                for key in list(st.session_state.keys()):
                    # Keep error selection mode and categories
                    if key not in ["error_selection_mode", "selected_error_categories"]:
                        del st.session_state[key]
                
                # Re-initialize session state
                init_session_state()
                
                # Go back to the first tab
                st.session_state.active_tab = 0
                
                # Rerun the app
                st.rerun()
            
            # Display feedback results
            feedback_display_ui.render_results(
                comparison_report=st.session_state.comparison_report,
                review_summary=st.session_state.review_summary,
                review_analysis=st.session_state.review_analysis,
                review_history=st.session_state.review_history,
                on_reset_callback=handle_reset
            )

if __name__ == "__main__":
    main()