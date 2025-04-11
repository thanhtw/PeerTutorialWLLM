"""
Java Peer Code Review Training System with MCP - Main Application

This module provides a Streamlit web interface for the Java code review training system
using LangGraph for workflow management and MCP for model interactions.
"""

import streamlit as st
import sys
import os
import logging
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import CSS utilities
from static.css_utils import load_css

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

# Import state schema
from state_schema import WorkflowState

# Import UI components
from ui.error_selector import ErrorSelectorUI
from ui.code_display import CodeDisplayUI
from ui.feedback_display import FeedbackDisplayUI
from ui.model_manager import ModelManagerUI

# Import MCP components
from mcp_server import mcp
from mcp_client import MCPClient
from improved_langgraph_workflow_mcp import MCPJavaCodeReviewGraph

# Import LLM Manager (for backward compatibility)
from llm_manager import LLMManager

# Import data access
from data.json_error_repository import JsonErrorRepository

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Java Code Review Trainer (MCP)",
    page_icon="",  # Java coffee cup icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS from external files
css_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "css")
loaded_files = load_css(css_directory=css_dir)

if not load_css(css_directory=css_dir):
    # Fallback to inline CSS if loading fails
    logger.warning("Failed to load CSS files, falling back to inline CSS")

def run_mcp_server():
    """Start the MCP server in a separate thread."""
    try:
        import threading
        
        def server_thread():
            import uvicorn
            port = int(os.getenv("MCP_PORT", "8000"))
            host = os.getenv("MCP_HOST", "localhost")
            # Get the absolute path to the mcp_server module
            module_path = "mcp_server:mcp"
            logger.info(f"Starting MCP server at {host}:{port} with module {module_path}")
            uvicorn.run(module_path, host=host, port=port, log_level="info", reload=False)
        
        # Start server in a thread
        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
        logger.info("MCP server thread started")
        
        # Wait a moment for server to initialize
        time.sleep(2)
        
        return True
    except Exception as e:
        logger.error(f"Error starting MCP server: {str(e)}")
        return False

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
    
    # Initialize MCP client if not present
    if 'mcp_client' not in st.session_state:
        st.session_state.mcp_client = MCPClient()
    
    # Initialize MCP server status
    if 'mcp_server_started' not in st.session_state:
        st.session_state.mcp_server_started = False

def start_mcp_if_needed():
    """Start the MCP server if it's not already running."""
    if not st.session_state.mcp_server_started:
        success = run_mcp_server()
        st.session_state.mcp_server_started = success
        if success:
            st.success("MCP server started successfully")
        else:
            st.error("Failed to start MCP server")

def generate_code_problem(workflow: MCPJavaCodeReviewGraph, 
                         params: Dict[str, str], 
                         error_selection_mode: str,
                         selected_error_categories: Dict[str, List[str]],
                         selected_specific_errors: List[Dict[str, Any]] = None):
    """Generate a code problem with progress indicator and evaluation visualization."""
    try:
        # Initialize state and parameters
        state = st.session_state.workflow_state
        code_length = str(params.get("code_length", "medium"))
        difficulty_level = str(params.get("difficulty_level", "medium"))
        state.code_length = code_length
        state.difficulty_level = difficulty_level
        
        # Verify we have error selections
        has_selections = False
        if error_selection_mode == "specific" and selected_specific_errors:
            has_selections = len(selected_specific_errors) > 0
        elif error_selection_mode == "standard" or error_selection_mode == "advanced":
            build_selected = selected_error_categories.get("build", [])
            checkstyle_selected = selected_error_categories.get("checkstyle", [])
            has_selections = len(build_selected) > 0 or len(checkstyle_selected) > 0
        
        if not has_selections:
            st.error("No error categories or specific errors selected. Please select at least one error type.")
            return False
        
        # Update the state with selected error categories
        state.selected_error_categories = selected_error_categories
        
        # First stage: Generate initial code
        with st.status("Generating initial Java code...", expanded=True) as status:
            if error_selection_mode == "specific" and selected_specific_errors:
                updated_state = workflow.generate_code_with_specific_errors(state, selected_specific_errors)
            else:
                updated_state = workflow.generate_code_node(state)
            
            if updated_state.error:
                st.error(f"Error: {updated_state.error}")
                return False
        
        # Second stage: Display the evaluation process
        st.info("Evaluating and improving the code...")
        
        # Create a process visualization
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Code Generation & Evaluation Process")
            
            # Create a progress container
            progress_container = st.container()
            with progress_container:
                # Create a progress bar
                progress_bar = st.progress(0.25)
                st.write("**Step 1:** Initial code generation completed")
                
                # Evaluate the code
                with st.status("Evaluating code quality...", expanded=False):
                    updated_state = workflow.evaluate_code_node(updated_state)
                
                progress_bar.progress(0.5)
                st.write("**Step 2:** Code evaluation completed")
                
                # Show evaluation results
                if hasattr(updated_state, 'evaluation_result') and updated_state.evaluation_result:
                    found = len(updated_state.evaluation_result.get("found_errors", []))
                    missing = len(updated_state.evaluation_result.get("missing_errors", []))
                    total = found + missing
                    if total == 0:
                        total = 1  # Avoid division by zero
                    
                    quality_percentage = (found / total * 100)
                    st.write(f"**Initial quality:** Found {found}/{total} required errors ({quality_percentage:.1f}%)")
                    
                    # Regeneration cycle if needed
                    if missing > 0 and updated_state.current_step == "regenerate":
                        st.write("**Step 3:** Improving code quality")
                        
                        attempt = 1
                        max_attempts = getattr(updated_state, 'max_evaluation_attempts', 3)
                        previous_found = found
                        
                        # Loop through regeneration attempts
                        while (getattr(updated_state, 'current_step', None) == "regenerate" and 
                              attempt < max_attempts):
                            progress_value = 0.5 + (0.5 * (attempt / max_attempts))
                            progress_bar.progress(progress_value)
                            
                            # Regenerate code
                            with st.status(f"Regenerating code (Attempt {attempt+1})...", expanded=False):
                                updated_state = workflow.regenerate_code_node(updated_state)
                            
                            # Re-evaluate code
                            with st.status(f"Re-evaluating code...", expanded=False):
                                updated_state = workflow.evaluate_code_node(updated_state)
                            
                            # Show updated results
                            if hasattr(updated_state, 'evaluation_result'):
                                new_found = len(updated_state.evaluation_result.get("found_errors", []))
                                new_missing = len(updated_state.evaluation_result.get("missing_errors", []))
                                
                                st.write(f"**Quality after attempt {attempt+1}:** Found {new_found}/{total} required errors " +
                                      f"({new_found/total*100:.1f}%)")
                                
                                if new_found > previous_found:
                                    st.success(f" Added {new_found - previous_found} new errors in this attempt!")
                                    
                                previous_found = new_found
                            
                            attempt += 1
                    
                    # Complete the progress
                    progress_bar.progress(1.0)
                    
                    # Show final outcome
                    if quality_percentage == 100:
                        st.success(" All requested errors successfully implemented!")
                    elif quality_percentage >= 80:
                        st.success(f" Good quality code generated with {quality_percentage:.1f}% of requested errors!")
                    else:
                        st.warning(f"� Code generated with {quality_percentage:.1f}% of requested errors. " +
                                "Some errors could not be implemented but the code is still suitable for review practice.")
                
        with col2:
            # Show statistics in the sidebar
            st.subheader("Generation Stats")
            
            if hasattr(updated_state, 'evaluation_result') and updated_state.evaluation_result:
                found = len(updated_state.evaluation_result.get("found_errors", []))
                missing = len(updated_state.evaluation_result.get("missing_errors", []))
                total = found + missing
                if total > 0:
                    quality_percentage = (found / total * 100)
                    st.metric("Quality", f"{quality_percentage:.1f}%")
                
                st.metric("Errors Found", f"{found}/{total}")
                
                if hasattr(updated_state, 'evaluation_attempts'):
                    st.metric("Generation Attempts", updated_state.evaluation_attempts)
        
        # Update session state
        st.session_state.workflow_state = updated_state
        st.session_state.active_tab = 1  # Move to the review tab
        st.session_state.error = None
        
        # Also show the generated code in this tab for immediate feedback
        st.subheader("Generated Java Code")
        
        code_to_display = None
        if hasattr(updated_state.code_snippet, 'clean_code') and updated_state.code_snippet.clean_code:
            code_to_display = updated_state.code_snippet.clean_code
        elif hasattr(updated_state.code_snippet, 'code') and updated_state.code_snippet.code:
            code_to_display = updated_state.code_snippet.code
            
        if code_to_display:
            st.code(code_to_display, language="java")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating code problem: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"Error generating code problem: {str(e)}")
        return False

def process_student_review(workflow: MCPJavaCodeReviewGraph, student_review: str):
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

def check_mcp_server_status():
    """Check the status of the MCP server."""
    try:
        mcp_client = st.session_state.mcp_client
        status = mcp_client.check_server_status()
        
        if status.get("status") == "online":
            return True, "MCP server is running"
        else:
            return False, f"MCP server is not running: {status.get('details', status.get('error', 'Unknown error'))}"
    except Exception as e:
        return False, f"Error checking MCP server: {str(e)}"

def render_sidebar(llm_manager: LLMManager, workflow: MCPJavaCodeReviewGraph):
    """
    Render the sidebar with status and settings.
    """
    with st.sidebar:
        st.title("Java Code Review Trainer")
        st.markdown("### Using MCP")
        
        # Create sidebar tabs
        sidebar_tabs = st.radio(
            "Menu",
            ["Status", "Settings", "MCP Server"],
            horizontal=True,
            index=["Status", "Settings", "MCP Server"].index(st.session_state.sidebar_tab) 
            if st.session_state.sidebar_tab in ["Status", "Settings", "MCP Server"] else 0
        )
        
        # Update sidebar tab in session state
        st.session_state.sidebar_tab = sidebar_tabs
        
        st.markdown("---")
        
        if sidebar_tabs == "Status":
            render_status_sidebar(llm_manager)
            
        elif sidebar_tabs == "Settings":
            render_settings_sidebar(workflow)
            
        elif sidebar_tabs == "MCP Server":
            render_mcp_sidebar()

def render_mcp_sidebar():
    """Render the MCP server status and controls in the sidebar."""
    st.header("MCP Server Status")
    
    # Check MCP server status
    mcp_status, message = check_mcp_server_status()
    
    # Status indicator
    if mcp_status:
        st.markdown(
            '<div class="status-item">'
            '<span>=�</span>'
            '<div class="status-text">MCP Server</div>'
            '<span class="status-badge badge-ok">Running</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-item">'
            '<span>=4</span>'
            '<div class="status-text">MCP Server</div>'
            '<span class="status-badge badge-error">Not Running</span>'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Start server button
        if st.button("Start MCP Server"):
            with st.spinner("Starting MCP server..."):
                success = run_mcp_server()
                if success:
                    st.session_state.mcp_server_started = True
                    st.success("MCP server started successfully!")
                    st.rerun()
                else:
                    st.error("Failed to start MCP server")
    
    # MCP tools information
    st.subheader("Available MCP Tools")
    
    # List of tools provided by MCP
    tools = [
        {"name": "generate_java_code", "description": "Generate Java code with errors"},
        {"name": "evaluate_java_code", "description": "Evaluate Java code for errors"},
        {"name": "analyze_student_review", "description": "Analyze a student's code review"},
        {"name": "generate_targeted_guidance", "description": "Generate guidance for students"},
        {"name": "generate_final_feedback", "description": "Generate final feedback summary"},
        {"name": "generate_comparison_report", "description": "Compare student review to known problems"},
        {"name": "check_gpu_availability", "description": "Check if GPU is available"},
        {"name": "get_available_models", "description": "List available models"},
        {"name": "download_model", "description": "Download a model"}
    ]
    
    for tool in tools:
        st.markdown(
            f'<div style="margin-bottom: 10px;">'
            f'<strong>{tool["name"]}</strong><br/>'
            f'<span style="font-size: 0.9em; color: #666;">{tool["description"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # MCP configuration
    st.subheader("MCP Configuration")
    
    mcp_host = os.getenv("MCP_HOST", "localhost")
    mcp_port = os.getenv("MCP_PORT", "8000")
    
    st.code(f"Host: {mcp_host}\nPort: {mcp_port}", language="bash")
    
    # Add restart button
    if mcp_status and st.button("Restart MCP Server"):
        with st.spinner("Restarting MCP server..."):
            # First set the flag to not started
            st.session_state.mcp_server_started = False
            # Then start it again
            success = run_mcp_server()
            if success:
                st.session_state.mcp_server_started = True
                st.success("MCP server restarted successfully!")
                st.rerun()
            else:
                st.error("Failed to restart MCP server")

def render_status_sidebar(llm_manager: LLMManager):
    """Render the status sidebar tab with GPU information and model details."""
    st.header("System Status")
    
    # Get status information
    ollama_status, _ = llm_manager.check_ollama_connection()
    
    # Check MCP server status
    mcp_status, _ = check_mcp_server_status()
    
    # Get GPU information
    gpu_info = llm_manager.refresh_gpu_info()
    has_gpu = gpu_info.get("has_gpu", False)
    
    # Status indicators with better styling
    st.markdown("""
        <div class="status-container">
    """, unsafe_allow_html=True)
    
    # MCP status with icon
    if mcp_status:
        st.markdown(
            '<div class="status-item">'
            '<span>=�</span>'
            '<div class="status-text">MCP Server</div>'
            '<span class="status-badge badge-ok">Running</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-item">'
            '<span>=4</span>'
            '<div class="status-text">MCP Server</div>'
            '<span class="status-badge badge-error">Not Running</span>'
            '</div>',
            unsafe_allow_html=True
        )
    
    # Ollama status with icon
    if ollama_status:
        st.markdown(
            '<div class="status-item">'
            '<span>=�</span>'
            '<div class="status-text">Ollama</div>'
            '<span class="status-badge badge-ok">Running</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-item">'
            '<span>=4</span>'
            '<div class="status-text">Ollama</div>'
            '<span class="status-badge badge-error">Not Running</span>'
            '</div>',
            unsafe_allow_html=True
        )
    
    # GPU Status
    if has_gpu:
        gpu_name = gpu_info.get("gpu_name", "GPU")
        memory_total = gpu_info.get("formatted_total", "Unknown")
        memory_used = gpu_info.get("formatted_used", "Unknown")
        utilization = gpu_info.get("utilization", 0)
        memory_percent = gpu_info.get("memory_used_percent", 0)
        
        # GPU Status icon and badge
        st.markdown(
            '<div class="status-item gpu-status-item">'
            '<span class="gpu-status-icon">=�</span>'
            '<div class="gpu-status-text">GPU</div>'
            '<span class="status-badge badge-gpu">Active</span>'
            '</div>',
            unsafe_allow_html=True
        )
        
        # GPU details section
        st.markdown(
            f'<div class="gpu-info-section">'
            f'<div class="gpu-info-header">'
            f'<span>=�</span>'
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
            '<span>�</span>'
            '<div class="status-text">GPU</div>'
            '<span class="status-badge badge-warning">Not Available</span>'
            '</div>',
            unsafe_allow_html=True
        )
    
    # Close the status container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About section with better styling
    st.markdown(
        '<div class="about-box">'
        '<h3>=� How This App Works</h3>'
        '<p>Practice Java code review in 3 simple steps:</p>'
        '<ol>'
        '<li>=� Generate Java code with intentional errors</li>'
        '<li>='
 'Find and identify the issues in the code</li>'
        '<li>=� Get feedback on your review accuracy</li>'
        '</ol>'
        '<p>Click "Generate Problem" to start practicing!</p>'
        '</div>',
        unsafe_allow_html=True
    )

def render_settings_sidebar(workflow: MCPJavaCodeReviewGraph):
    """Render the settings sidebar tab"""
    st.header("Review Settings")
    
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
    
    # Display settings
    st.subheader("Display Settings")
    
    # Toggle for instructor view
    instructor_view = st.checkbox(
        "Enable Instructor View",
        value=False,
        help="Show known errors (for instructors)"
    )
    
    if "instructor_view" not in st.session_state or instructor_view != st.session_state.instructor_view:
        st.session_state.instructor_view = instructor_view

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
        
        success = generate_code_problem(
            workflow,
            safe_params,
            mode,
            selected_categories,
            selected_specific_errors
        )
        
        if success:
            st.rerun()
    
    # Display code if available
    if 'workflow_state' in st.session_state:
        state = st.session_state.workflow_state
        
        if hasattr(state, 'code_snippet') and state.code_snippet:
            st.subheader("Java Code for Review")
            
            code_to_display = None
            if hasattr(state.code_snippet, 'clean_code') and state.code_snippet.clean_code:
                code_to_display = state.code_snippet.clean_code
            elif hasattr(state.code_snippet, 'code') and state.code_snippet.code:
                code_to_display = state.code_snippet.code
                
            if code_to_display:
                code_display_ui.render_code_display(
                    code_to_display,
                    state.code_snippet.known_problems if st.session_state.get("instructor_view", False) else None
                )
    else:
        st.info("No code generated yet. Select error types and click 'Generate Code Problem' to create a Java code snippet.")

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

def main():
    """Enhanced main application function with MCP integration."""
    # Initialize session state
    init_session_state()
    
    # Start MCP server if needed
    start_mcp_if_needed()
    
    # Initialize LLM manager and workflow
    llm_manager = LLMManager()
    mcp_client = st.session_state.mcp_client
    workflow = MCPJavaCodeReviewGraph(mcp_client=mcp_client, llm_manager=llm_manager)
    
    # Initialize UI components
    error_selector_ui = ErrorSelectorUI()
    code_display_ui = CodeDisplayUI()
    feedback_display_ui = FeedbackDisplayUI()
    
    # Render sidebar
    render_sidebar(llm_manager, workflow)
    
    # Header with improved styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: rgb(178 185 213); margin-bottom: 5px;">Java Code Review Training System</h1>
        <p style="font-size: 1.1rem; color: #666;">Using Model Control Protocol (MCP) for enhanced model interactions</p>
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