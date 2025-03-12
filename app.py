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

# Configure logging
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to the path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import LangGraph components
from langgraph_workflow import JavaCodeReviewGraph
from state_schema import WorkflowState

# Import UI components
from ui.error_selector import ErrorSelectorUI
from ui.code_display import CodeDisplayUI
from ui.feedback_display import FeedbackDisplayUI
from ui.model_manager import ModelManagerUI

# Import LLM Manager
from llm_manager import LLMManager

# Import data access
from data.json_error_repository import JsonErrorRepository

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Java Code Review Trainer",
    page_icon="",  # Java coffee cup icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
    <style>
        /* Global styles */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1200px;
        }
        
        h1, h2, h3 {
            margin-bottom: 0.5rem;
        }
        
        /* Code display */
        .stTextArea textarea {
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        
        .code-block {
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
            padding: 15px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            margin: 10px 0;
            font-size: 14px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        /* Content containers */
        .content-section {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #e9ecef;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Status indicators */
        .status-ok {
            color: #28a745;
            font-weight: bold;
        }
        
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        
        /* Feedback boxes */
        .guidance-box {
            background-color: #e8f4f8;
            border-left: 4px solid #0275d8;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .feedback-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .review-box {
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
            padding: 15px;
            margin: 10px 0;
        }
        
        /* Badge styling */
        .iteration-badge {
            display: inline-block;
            background-color: #0275d8;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }
        
        /* Button spacing and sizing */
        .stButton button {
            font-weight: 500;
            padding: 0.375rem 0.75rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 4px 4px 0px 0px;
            padding: 8px 16px;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0275d8;
            color: white;
        }
        
        /* Card styling for model display */
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
        }
        
        /* Make the form fields more compact */
        div[data-testid="stForm"] {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
        
        /* Make the checkboxes more compact */
        div[data-testid="stVerticalBlock"] > div[data-testid="stCheckbox"] {
            margin-bottom: 0;
            padding-bottom: 0;
        }
        
        /* Adjust the text area height */
        .review-textarea textarea {
            min-height: 300px;
        }
        
        /* Streamlit sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
    </style>
""", unsafe_allow_html=True)

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
    
    # Initialize sidebar state if not present
    if 'sidebar_tab' not in st.session_state:
        st.session_state.sidebar_tab = "Status"

def generate_code_problem(workflow: JavaCodeReviewGraph, 
                         params: Dict[str, str], 
                         error_selection_mode: str,
                         selected_error_categories: Dict[str, List[str]],
                         selected_specific_errors: List[Dict[str, Any]] = None):
    """
    Generate a code problem with progress indicator.
    """
    # Show progress during generation
    with st.status("Generating Java code problem...", expanded=True) as status:
        try:
            status.update(label="Preparing parameters...", state="running", expanded=True)
            
            # Update workflow state with generation parameters
            state = st.session_state.workflow_state
            
            # Ensure code_length and difficulty_level are strings
            code_length = str(params.get("code_length", "medium"))
            difficulty_level = str(params.get("difficulty_level", "medium"))
            
            state.code_length = code_length
            state.difficulty_level = difficulty_level
            state.selected_error_categories = selected_error_categories
            
            # Run the generation node with appropriate error selection
            status.update(label="Generating code with errors...", state="running")
            
            if error_selection_mode == "specific" and selected_specific_errors:
                # Use specific errors for generation
                updated_state = workflow.generate_code_with_specific_errors(
                    state, 
                    selected_specific_errors
                )
            else:
                # Use category-based error selection
                updated_state = workflow.generate_code_node(state)
            
            # Check for errors
            if updated_state.error:
                status.update(label=f"Error: {updated_state.error}", state="error")
                st.session_state.error = updated_state.error
                return False
            
            # Update session state
            st.session_state.workflow_state = updated_state
            st.session_state.active_tab = 1  # Move to the review tab
            st.session_state.error = None
            
            status.update(label="Code generated successfully!", state="complete")
            return True
            
        except Exception as e:
            logger.error(f"Error generating code problem: {str(e)}")
            status.update(label=f"Error: {str(e)}", state="error")
            st.session_state.error = f"Error generating code problem: {str(e)}"
            return False

def process_student_review(workflow: JavaCodeReviewGraph, student_review: str):
    """
    Process a student review with progress indicator.
    """
    # Show progress during analysis
    with st.status("Processing your review...", expanded=True) as status:
        try:
            # Get current state
            state = st.session_state.workflow_state
            
            # Store the current review in session state for display consistency
            current_iteration = state.current_iteration
            st.session_state[f"submitted_review_{current_iteration}"] = student_review
            
            # Submit the review and update the state
            status.update(label="Analyzing your review...", state="running")
            updated_state = workflow.submit_review(state, student_review)
            
            # Check for errors
            if updated_state.error:
                status.update(label=f"Error: {updated_state.error}", state="error")
                st.session_state.error = updated_state.error
                return False
            
            # Update session state
            st.session_state.workflow_state = updated_state
            
            # Check if we should generate summary
            if workflow.should_continue_review(updated_state) == "generate_summary":
                status.update(label="Generating final feedback...", state="running")
                
                # Run the summary generation node
                final_state = workflow.generate_summary_node(updated_state)
                st.session_state.workflow_state = final_state
                
                # Move to the analysis tab
                st.session_state.active_tab = 2
            
            status.update(label="Analysis complete!", state="complete")
            
            # Force a rerun to update the UI
            st.rerun()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing student review: {str(e)}")
            status.update(label=f"Error: {str(e)}", state="error")
            st.session_state.error = f"Error processing student review: {str(e)}"
            return False

def render_sidebar(llm_manager: LLMManager, workflow: JavaCodeReviewGraph):
    """
    Render the sidebar with status and settings.
    """
    with st.sidebar:
        st.title("Java Code Review Trainer")
        
        # Create sidebar tabs
        sidebar_tabs = st.radio(
            "Menu",
            ["Status", "Settings", "Models"],
            horizontal=True,
            index=["Status", "Settings", "Models"].index(st.session_state.sidebar_tab) 
            if st.session_state.sidebar_tab in ["Status", "Settings", "Models"] else 0
        )
        
        # Update sidebar tab in session state
        st.session_state.sidebar_tab = sidebar_tabs
        
        st.markdown("---")
        
        if sidebar_tabs == "Status":
            render_status_sidebar(llm_manager)
            
        elif sidebar_tabs == "Settings":
            render_settings_sidebar(workflow)
            
        elif sidebar_tabs == "Models":
            # Initialize ModelManagerUI
            model_manager_ui = ModelManagerUI(llm_manager)
            
            # Render the model manager UI
            model_selections = model_manager_ui.render_model_manager()
            
            # If models have changed, reinitialize the workflow
            if any(model_selections[role] != os.getenv(f"{role.upper()}_MODEL", llm_manager.default_model) 
                  for role in ["generative", "review", "summary", "compare"]):
                
                # Update environment variables
                for role, model in model_selections.items():
                    os.environ[f"{role.upper()}_MODEL"] = model
                
                # Show reinitializing message
                st.info("Model selections changed. Reinitializing models...")

def render_status_sidebar(llm_manager: LLMManager):
    """Render the status sidebar tab"""
    st.header("System Status")
    
    # Create a status card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Check Ollama status
    status = check_ollama_status(llm_manager)
    
    if status["ollama_running"]:
        st.markdown(f"- Ollama: <span class='status-ok'> Running</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"- Ollama: <span class='status-error'> Not Running</span>", unsafe_allow_html=True)
        
        # Troubleshooting information
        with st.expander("Troubleshooting"):
            st.markdown("""
            1. **Check if Ollama is running:**
               ```bash
               curl http://localhost:11434/api/tags
               ```
               
            2. **Make sure Ollama is started:**
               - On Linux/Mac: `ollama serve`
               - On Windows: Start the Ollama application
            """)
    
    if status["default_model_available"]:
        st.markdown(f"- Default model: <span class='status-ok'> Available</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"- Default model: <span class='status-warning'>� Not Found</span>", unsafe_allow_html=True)
        if status["ollama_running"]:
            if st.button("Pull Default Model", key="pull_default_btn"):
                with st.spinner(f"Pulling {llm_manager.default_model}..."):
                    if llm_manager.download_ollama_model(llm_manager.default_model):
                        st.success("Default model pulled successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to pull default model.")
    
    if status["all_models_configured"]:
        st.markdown(f"- Model configuration: <span class='status-ok'> Complete</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"- Model configuration: <span class='status-warning'>� Incomplete</span>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Application info
    st.subheader("About")
    st.markdown("""
    This application helps you practice code review skills by:
    
    1. Generating Java code with intentional errors
    2. Letting you identify those errors
    3. Providing feedback on your review accuracy
    
    The system uses Ollama to run LLMs locally.
    """)

def render_settings_sidebar(workflow: JavaCodeReviewGraph):
    """Render the settings sidebar tab"""
    st.header("Review Settings")
    
    # Create a settings card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Review iteration settings
    max_iterations = st.slider(
        "Maximum Review Attempts",
        min_value=1,
        max_value=5,
        value=st.session_state.workflow_state.max_iterations,
        help="Maximum attempts before final evaluation"
    )
    
    # Update max iterations in workflow state
    if max_iterations != st.session_state.workflow_state.max_iterations:
        st.session_state.workflow_state.max_iterations = max_iterations
    
    # Minimum identified percentage for sufficient review
    min_identified_percentage = st.slider(
        "Required Accuracy",
        min_value=30,
        max_value=90,
        value=60,
        step=5,
        help="Minimum % of issues to identify for a sufficient review"
    )
    
    # Update student response evaluator
    if hasattr(workflow, 'evaluator') and hasattr(workflow.evaluator, 'min_identified_percentage'):
        workflow.evaluator.min_identified_percentage = min_identified_percentage
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display settings
    st.subheader("Display Settings")
    
    # Create display settings card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Toggle for instructor view
    instructor_view = st.checkbox(
        "Enable Instructor View",
        value=False,
        help="Show known errors (for instructors)"
    )
    
    if "instructor_view" not in st.session_state or instructor_view != st.session_state.instructor_view:
        st.session_state.instructor_view = instructor_view
    
    st.markdown('</div>', unsafe_allow_html=True)

def check_ollama_status(llm_manager: LLMManager) -> Dict[str, bool]:
    """
    Check the status of Ollama and required models.
    """
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

def render_generate_tab(workflow, error_selector_ui, code_display_ui):
    """Render the code generation tab."""
    st.markdown("### Generate a Java Code Problem")
    
    # Create a content card for the form
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        # Code parameters
        st.subheader("Code Options")
        params = error_selector_ui.render_code_params()
        
        # Mode selection (simplified)
        st.subheader("Error Selection")
        mode = error_selector_ui.render_mode_selector()
    
    with right_col:
        if mode == "standard":
            # Standard mode - problem areas
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
            
            # No specific errors in standard mode
            selected_specific_errors = []
            
        elif mode == "advanced":
            # Advanced mode - error categories
            selected_categories = error_selector_ui.render_category_selection(
                workflow.get_all_error_categories()
            )
            
            # No specific errors in advanced mode
            selected_specific_errors = []
            
        else:  # specific mode
            # Specific mode - exact errors
            error_repository = JsonErrorRepository()
            selected_specific_errors = error_selector_ui.render_specific_error_selection(error_repository)
            
            # We still need categories for fallback
            selected_categories = {
                "build": ["CompileTimeErrors", "RuntimeErrors", "LogicalErrors"],
                "checkstyle": ["NamingConventionChecks", "WhitespaceAndFormattingChecks"]
            }
    
    # Generate button - centered at the bottom
    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    generate_button = st.button("Generate Code Problem", type="primary", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if generate_button:
        # Ensure params has string values for code_length and difficulty_level
        safe_params = {
            "code_length": str(params.get("code_length", "medium")),
            "difficulty_level": str(params.get("difficulty_level", "medium"))
        }
        
        success = generate_code_problem(
            workflow,
            safe_params,
            mode,
            selected_categories,
            selected_specific_errors
        )
        
        if success:
            st.rerun()
    
    # Display existing code if available
    state = st.session_state.workflow_state
    if state.code_snippet:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        code_display_ui.render_code_display(
            state.code_snippet.code,
            state.code_snippet.known_problems if st.session_state.get("instructor_view", False) else None
        )
        st.markdown('</div>', unsafe_allow_html=True)

def render_review_tab(workflow, code_display_ui):
    """Render the review submission tab."""
    state = st.session_state.workflow_state
    
    if not state.code_snippet:
        st.info("Please generate a code problem first in the 'Generate Code Problem' tab.")
        return
    
    # Code display section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    code_display_ui.render_code_display(state.code_snippet.code)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Review submission section
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    
    # Submission callback
    def handle_review_submission(student_review):
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
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_feedback_tab(workflow, feedback_display_ui):
    """Render the feedback and analysis tab."""
    state = st.session_state.workflow_state
    
    if not state.comparison_report and not state.review_summary:
        st.info("Please submit your review in the 'Submit Review' tab first.")
        return
    
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
    st.markdown('<div class="content-section">', unsafe_allow_html=True)
    feedback_display_ui.render_results(
        comparison_report=state.comparison_report,
        review_summary=state.review_summary,
        review_analysis=latest_analysis,
        review_history=review_history,
        on_reset_callback=handle_reset
    )
    st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Render sidebar
    render_sidebar(llm_manager, workflow)
    
    # Header
    st.markdown("# Java Code Review Training System")
    st.caption("Learn and practice Java code review skills with generated exercises")
    
    # Display error message if there's an error
    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()
    
    # Create tabs for different steps of the workflow
    tabs = st.tabs([
        "1� Generate Problem", 
        "2� Submit Review", 
        "3� View Feedback"
    ])
    
    # Set the active tab based on session state
    active_tab = st.session_state.active_tab
    
    # Tab content
    with tabs[0]:
        render_generate_tab(workflow, error_selector_ui, code_display_ui)
    
    with tabs[1]:
        render_review_tab(workflow, code_display_ui)
    
    with tabs[2]:
        render_feedback_tab(workflow, feedback_display_ui)

if __name__ == "__main__":
    main()