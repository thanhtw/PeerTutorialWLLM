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

# Import CSS utilities
from static.css_utils import load_css, inject_custom_css


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
    page_icon="",  # Java coffee cup icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS from external files
css_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "css")
loaded_files = load_css(css_directory=css_dir)
inject_custom_css()



if not load_css(css_directory=css_dir):
    # Fallback to inline CSS if loading fails
    logger.warning("Failed to load CSS files, falling back to inline CSS")  

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
    Generate a code problem with progress indicator and enhanced error debugging.
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
            
            # Debug print current state before updating
            print("\n========== GENERATE_CODE_PROBLEM FUNCTION ==========")
            print(f"UI Selected Categories: {selected_error_categories}")
            print(f"UI Selected Specific Errors: {len(selected_specific_errors) if selected_specific_errors else 0}")
            print(f"Mode: {error_selection_mode}")
            
            # Explicitly ensure we have some error selections based on the mode
            has_selections = False
            
            if error_selection_mode == "specific" and selected_specific_errors:
                # Specific errors mode
                has_selections = len(selected_specific_errors) > 0
                print(f"Specific mode has selections: {has_selections}")
                
            elif error_selection_mode == "standard" or error_selection_mode == "advanced":
                # Category-based selection modes
                build_selected = selected_error_categories.get("build", [])
                checkstyle_selected = selected_error_categories.get("checkstyle", [])
                has_selections = len(build_selected) > 0 or len(checkstyle_selected) > 0
                print(f"Category mode has selections: {has_selections} (Build: {len(build_selected)}, Checkstyle: {len(checkstyle_selected)})")
            
            # If no selections were made, show error and return
            if not has_selections:
                error_msg = "No error categories or specific errors selected. Please select at least one error type."
                status.update(label=f"Error: {error_msg}", state="error")
                st.session_state.error = error_msg
                return False
            
            # Update the state with selected error categories
            # Use only the properly named field without underscore
            state.selected_error_categories = selected_error_categories
            
            print(f"Updated state with categories: {state.selected_error_categories}")
            
            # ENHANCED DEBUGGING: Pre-calculate what errors will be selected to show before generation
            if error_selection_mode != "specific":
                # This will show you what actual errors are selected from categories before generation
                from data.json_error_repository import JsonErrorRepository
                error_repo = JsonErrorRepository()
                error_count = {
                    "easy": 2,
                    "medium": 4,
                    "hard": 6
                }.get(difficulty_level.lower(), 4)
                
                # Get the actual errors that would be selected for these categories
                preview_errors, _ = error_repo.get_errors_for_llm(
                    selected_categories=selected_error_categories,
                    count=error_count,
                    difficulty=difficulty_level
                )
                
                print("\n========== PREVIEW OF ERRORS TO BE SENT TO LLM ==========")
                print(f"The following {len(preview_errors)} errors will be used for code generation:")
                for i, error in enumerate(preview_errors, 1):
                    error_type = error.get("type", "Unknown")
                    name = error.get("name", "Unknown")
                    category = error.get("category", "Unknown")
                    print(f"  {i}. {error_type} - {name} (from {category})")
                    
                    # FIXED: Proper handling of implementation guide
                    if "implementation_guide" in error:
                        implementation_guide = error.get('implementation_guide', '')
                        if len(implementation_guide) > 100:
                            print(f"     Implementation: {implementation_guide[:100]}...")
                        else:
                            print(f"     Implementation: {implementation_guide}")
                
                print("=================================================")
            
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
    Process a student review with progress indicator and improved error handling.
    
    Args:
        workflow: The JavaCodeReviewGraph workflow instance
        student_review: The student's review text
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Show progress during analysis
    with st.status("Processing your review...", expanded=True) as status:
        try:
            # Get current state
            if not hasattr(st.session_state, 'workflow_state'):
                status.update(label="Error: Workflow state not initialized", state="error")
                st.session_state.error = "Please generate a code problem first"
                return False
                
            state = st.session_state.workflow_state
            
            # Check if code snippet exists
            if not state.code_snippet:
                status.update(label="Error: No code snippet available", state="error")
                st.session_state.error = "Please generate a code problem first"
                return False
            
            # Check if student review is empty
            if not student_review.strip():
                status.update(label="Error: Review cannot be empty", state="error")
                st.session_state.error = "Please enter your review before submitting"
                return False
            
            # Store the current review in session state for display consistency
            current_iteration = state.current_iteration
            st.session_state[f"submitted_review_{current_iteration}"] = student_review
            
            # Update status
            status.update(label="Analyzing your review...", state="running")
            
            # Log submission attempt
            logger.info(f"Submitting review (iteration {current_iteration}): {student_review[:100]}...")
            
            # Submit the review and update the state
            updated_state = workflow.submit_review(state, student_review)
            
            # Check for errors
            if updated_state.error:
                status.update(label=f"Error: {updated_state.error}", state="error")
                st.session_state.error = updated_state.error
                logger.error(f"Error during review analysis: {updated_state.error}")
                return False
            
            # Update session state
            st.session_state.workflow_state = updated_state
            
            # Log successful analysis
            logger.info(f"Review analysis complete for iteration {current_iteration}")
            
            # Check if we should generate summary
            if workflow.should_continue_review(updated_state) == "generate_summary":
                status.update(label="Generating final feedback...", state="running")
                
                # Run the summary generation node
                try:
                    final_state = workflow.generate_summary_node(updated_state)
                    st.session_state.workflow_state = final_state
                    
                    # Move to the analysis tab
                    st.session_state.active_tab = 2
                    
                    status.update(label="Analysis complete! Moving to Feedback tab...", state="complete")
                except Exception as e:
                    error_msg = f"Error generating final feedback: {str(e)}"
                    logger.error(error_msg)
                    status.update(label=error_msg, state="error")
                    st.session_state.error = error_msg
                    return False
            else:
                status.update(label="Analysis complete!", state="complete")
            
            # Force a rerun to update the UI
            time.sleep(0.5)  # Short delay to ensure the status message is visible
            st.rerun()
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing student review: {str(e)}"
            logger.error(error_msg)
            status.update(label=error_msg, state="error")
            st.session_state.error = error_msg
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
    """Render the status sidebar tab with GPU information and model details."""
    st.header("System Status")
    
    # Check Ollama status
    status = check_ollama_status(llm_manager)
    
    # Get GPU information
    gpu_info = llm_manager.refresh_gpu_info()
    has_gpu = gpu_info.get("has_gpu", False)
    
    # Get active models
    active_models = llm_manager.get_active_models()
    
    # Status indicators with better styling
    st.markdown("""
        <div class="status-container">
    """, unsafe_allow_html=True)
    
    # Ollama status with icon
    if status["ollama_running"]:
        st.markdown(
            '<div class="status-item">'
            '<span>🟢</span>'
            '<div class="status-text">Ollama</div>'
            '<span class="status-badge badge-ok">Running</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-item">'
            '<span>🔴</span>'
            '<div class="status-text">Ollama</div>'
            '<span class="status-badge badge-error">Not Running</span>'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Simple troubleshooting button
        if st.button("👨‍💻 Troubleshooting Tips", key="troubleshoot_btn"):
            st.info("""
            **Quick Fix:**
            - Make sure Ollama is installed
            - Start Ollama with `ollama serve` in terminal
            - Windows users: Launch the Ollama application
            """)
    
    # GPU Status - New and more prominent
    if has_gpu:
        gpu_name = gpu_info.get("gpu_name", "GPU")
        memory_total = gpu_info.get("formatted_total", "Unknown")
        memory_used = gpu_info.get("formatted_used", "Unknown")
        utilization = gpu_info.get("utilization", 0)
        memory_percent = gpu_info.get("memory_used_percent", 0)
        
        # GPU Status icon and badge
        st.markdown(
            '<div class="status-item gpu-status-item">'
            '<span class="gpu-status-icon">🚀</span>'
            '<div class="gpu-status-text">GPU</div>'
            '<span class="status-badge badge-gpu">Active</span>'
            '</div>',
            unsafe_allow_html=True
        )
        
        # GPU details section
        st.markdown(
            f'<div class="gpu-info-section">'
            f'<div class="gpu-info-header">'
            f'<span>📊</span>'
            f'<div class="gpu-info-title">{gpu_name}</div>'
            f'</div>'
            f'<div class="gpu-info-detail">'
            f'<span class="gpu-info-label">Memory</span>'
            f'<span>{memory_used} / {memory_total}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Only show utilization if available
        if utilization is not None:
            st.markdown(
                f'<div class="gpu-info-detail">'
                f'<span class="gpu-info-label">Utilization</span>'
                f'<span>{utilization:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # Close GPU info section
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status-item">'
            '<span>⚠️</span>'
            '<div class="status-text">GPU</div>'
            '<span class="status-badge badge-warning">Not Available</span>'
            '</div>',
            unsafe_allow_html=True
        )
    
    # Default model status
    if status["default_model_available"]:
        st.markdown(
            '<div class="status-item">'
            '<span>🟢</span>'
            '<div class="status-text">Default Model</div>'
            '<span class="status-badge badge-ok">Available</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-item">'
            '<span>🟠</span>'
            '<div class="status-text">Default Model</div>'
            '<span class="status-badge badge-warning">Not Found</span>'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Download button only if Ollama is running
        if status["ollama_running"]:
            if st.button("⬇️ Download Model", key="pull_default_btn"):
                with st.spinner(f"Downloading {llm_manager.default_model}..."):
                    if llm_manager.download_ollama_model(llm_manager.default_model):
                        st.success("Model downloaded successfully!")
                        st.rerun()
                    else:
                        st.error("Download failed.")
    
    # Model configuration status
    if status["all_models_configured"]:
        st.markdown(
            '<div class="status-item">'
            '<span>🟢</span>'
            '<div class="status-text">Configuration</div>'
            '<span class="status-badge badge-ok">Complete</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-item">'
            '<span>🟠</span>'
            '<div class="status-text">Configuration</div>'
            '<span class="status-badge badge-warning">Incomplete</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="font-size: 12px; margin-left: 25px; color: #666;">'
            'Go to the "Models" tab to configure'
            '</div>',
            unsafe_allow_html=True
        )
    
    # Close the status container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Active Models Section - New
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Active Models")
    
    # Display current models in use with GPU status
    for role, model_info in active_models.items():
        model_name = model_info.get("name", "Unknown")
        uses_gpu = model_info.get("uses_gpu", False) or model_info.get("gpu_optimized", False)
        
        # Role display name formatting
        role_display = {
            "generative": "Code Generation",
            "review": "Review Analysis",
            "summary": "Feedback",
            "compare": "Comparison"
        }.get(role, role.capitalize())
        
        # Model display with GPU badge if applicable
        gpu_badge = '<span class="gpu-model-highlight">GPU</span>' if uses_gpu and has_gpu else ''
        
        st.markdown(
            f'<div class="gpu-model-container">'
            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<div><strong>{role_display}</strong></div>'
            f'<div>{model_name} {gpu_badge}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About section with better styling
    st.markdown(
        '<div class="about-box">'
        '<h3>📚 How This App Works</h3>'
        '<p>Practice Java code review in 3 simple steps:</p>'
        '<ol>'
        '<li>💻 Generate Java code with intentional errors</li>'
        '<li>🔍 Find and identify the issues in the code</li>'
        '<li>📊 Get feedback on your review accuracy</li>'
        '</ol>'
        '<p>Click "Generate Problem" to start practicing!</p>'
        '</div>',
        unsafe_allow_html=True
    )


def render_settings_sidebar(workflow: JavaCodeReviewGraph):
    """Render the settings sidebar tab"""
    st.header("Review Settings")
    
    # Create a settings card
    #st.markdown('<div class="card">', unsafe_allow_html=True)
    
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
    #st.markdown('<div class="card">', unsafe_allow_html=True)
    
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
    
    # Check GPU status
    gpu_info = llm_manager.check_gpu_availability()
    has_gpu = gpu_info.get("has_gpu", False)
    
    return {
        "ollama_running": connection_status,
        "default_model_available": default_model_available,
        "all_models_configured": all_models_configured,
        "gpu_available": has_gpu
    }

def render_generate_tab(workflow, error_selector_ui, code_display_ui):
    """Render the code generation tab with improved validation for error selection."""
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        # Code parameters        
        params = error_selector_ui.render_code_params()   
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
            
            # Update session state directly
            st.session_state.selected_error_categories = selected_categories
            
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
            
            # Empty categories for specific mode since we're using exact errors
            selected_categories = {
                "build": [],
                "checkstyle": []
            }
    
    # Generate button - centered at the bottom
    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    
    # Check if any errors are selected before enabling generation
    no_errors_selected = (
        (mode == "standard" and len(problem_areas) == 0) or
        (mode == "advanced" and 
         len(selected_categories.get("build", [])) == 0 and 
         len(selected_categories.get("checkstyle", [])) == 0) or
        (mode == "specific" and 
         (selected_specific_errors is None or len(selected_specific_errors) == 0))
    )
    
    # Debug selections before generation
    print("\n========== GENERATE BUTTON STATE ==========")
    print(f"Mode: {mode}")
    if mode == "standard":
        print(f"Problem areas: {problem_areas}")
    elif mode == "advanced":
        print(f"Selected categories: {selected_categories}")
    else:
        print(f"Selected specific errors: {len(selected_specific_errors) if selected_specific_errors else 0}")
    print(f"No errors selected: {no_errors_selected}")
    
    if no_errors_selected:
        st.warning("Please select at least one error type before generating code.")
        generate_button_disabled = True
    else:
        generate_button_disabled = False
    
    generate_button = st.button(
        "Generate Code Problem", 
        type="primary", 
        use_container_width=False,
        disabled=generate_button_disabled
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if generate_button and not generate_button_disabled:
        # Ensure params has string values for code_length and difficulty_level
        safe_params = {
            "code_length": str(params.get("code_length", "medium")),
            "difficulty_level": str(params.get("difficulty_level", "medium"))
        }
        
        # Additional debug before generation
        print("\n========== GENERATING CODE ==========")
        print(f"Selected categories: {selected_categories}")
        print(f"Selected specific errors: {len(selected_specific_errors) if selected_specific_errors else 0}")
        
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
        code_display_ui.render_code_display(
            state.code_snippet.code,
            state.code_snippet.known_problems if st.session_state.get("instructor_view", False) else None
        )

def render_review_tab(workflow, code_display_ui):
    """Render the review submission tab with enhanced UI."""
    state = st.session_state.workflow_state
    
    if not state.code_snippet:
        st.info("Please generate a code problem first in the 'Generate Code Problem' tab.")
        return
    
    # Professional code display section
    code_display_ui.render_code_display(state.code_snippet.code)
    
    # Enhanced submission callback
    def handle_review_submission(student_review):
        return process_student_review(workflow, student_review)
    
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

def create_enhanced_tabs(tab_labels):
    """
    Create enhanced tabs with professional styling.
    
    Args:
        tab_labels: List of tab labels to display
        
    Returns:
        List of streamlit tab objects
    """
    # Clean the labels - NO HTML
    import re
    clean_labels = []
    for label in tab_labels:
        # Remove any existing numbering like "1. " at the beginning
        clean_label = re.sub(r'^\d+[\.\-\:]\s*', '', label)
        clean_labels.append(clean_label)  
    
    # Create the tabs with clean labels (no HTML)
    tabs = st.tabs(clean_labels)
    
    # Apply custom HTML/CSS to style the tabs
    for i, tab in enumerate(tabs):
        # This is just to set the attributes - won't actually render
        # because we're using the clean labels above
        st.markdown(
            f"""
            <script>
                // Set the step number as a data attribute
                document.querySelectorAll('[data-baseweb="tab"]')[{i}].setAttribute('data-step-number', '{i+1}');
            </script>
            """,
            unsafe_allow_html=True
        )
    
    return tabs

def render_feedback_tab(workflow, feedback_display_ui):
    """Render the feedback and analysis tab with enhanced visuals."""
    state = st.session_state.workflow_state
    
    if not state.comparison_report and not state.review_summary:
        st.info("Please submit your review in the 'Submit Review' tab first.")
        return
    
    # Reset callback with confirmation
    def handle_reset():
        # Create a confirmation dialog
        if st.session_state.get("confirm_reset", False) or st.button("Confirm Reset", key="confirm_reset_btn"):
            # Create a new workflow state
            st.session_state.workflow_state = WorkflowState()
            
            # Reset active tab
            st.session_state.active_tab = 0
            
            # Reset confirmation flag
            if "confirm_reset" in st.session_state:
                del st.session_state.confirm_reset
            
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

def render_model_manager_sidebar(llm_manager, model_manager_ui):
    """Render the improved model manager in the sidebar."""
    # Get available models
    available_models = llm_manager.get_available_models()
    
    # Filter out only pulled models for selection
    model_options = [model["id"] for model in available_models if model["pulled"]]
    
    if not model_options:
        st.warning("No models available. Please pull at least one model in the Models tab.")
        return {}
    
    # Render the enhanced model selection table
    model_selections = model_manager_ui.render_model_selection_table(model_options)
    
    # Check for model pulls in progress
    if st.session_state.model_operations["pulling"]:
        model_name = st.session_state.model_operations["current_pull"]
        
        # Create a progress indicator
        st.subheader(f"Pulling model: {model_name}")
        progress_bar = st.progress(0)
        
        # Start the download in a separate thread
        if st.session_state.model_operations["pull_progress"] == 0:
            success = llm_manager.download_ollama_model(model_name)
            
            if success:
                st.session_state.model_operations["pulling"] = False
                st.session_state.model_operations["last_pulled"] = model_name
                st.session_state.model_operations["error"] = None
                st.rerun()
            else:
                st.session_state.model_operations["pulling"] = False
                st.session_state.model_operations["error"] = f"Failed to pull model: {model_name}"
                st.rerun()
        
        # Update progress bar
        progress_status = llm_manager.get_pull_status(model_name)
        progress_bar.progress(progress_status.get("progress", 0) / 100)
    
    return model_selections

def main():
    """Enhanced main application function."""
    # Initialize session state
    init_session_state()
    
    # Initialize LLM manager and workflow
    llm_manager = LLMManager()
    workflow = JavaCodeReviewGraph(llm_manager)
    
    # Initialize UI components
    error_selector_ui = ErrorSelectorUI()
    code_display_ui = CodeDisplayUI()
    feedback_display_ui = FeedbackDisplayUI()
    model_manager_ui = ModelManagerUI(llm_manager)
    
    # Render sidebar
    render_sidebar(llm_manager, workflow)
    
    # Header with improved styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: rgb(178 185 213); margin-bottom: 5px;">Java Code Review Training System</h1>
        <p style="font-size: 1.1rem; color: #666;">Learn and practice Java code review skills with AI-generated exercises</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display error message if there's an error
    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()
    
    # Create enhanced tabs for different steps of the workflow
    tab_labels = [
        "1. Generate Problem", 
        "2. Submit Review", 
        "3. View Feedback"
    ]
    
    # Use the enhanced tabs function
    tabs = create_enhanced_tabs(tab_labels)
    
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