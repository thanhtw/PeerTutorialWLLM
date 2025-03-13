"""
Model Manager UI module for Java Peer Review Training System.

This module provides the ModelManagerUI class for managing Ollama models
through the web interface.
"""

import streamlit as st
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManagerUI:
    """
    UI Component for managing Ollama models.
    
    This class handles displaying available models, pulling new models,
    and selecting models for different roles in the application.
    """
    
    def __init__(self, llm_manager):
        """
        Initialize the ModelManagerUI component.
        
        Args:
            llm_manager: LLMManager instance for interacting with Ollama
        """
        self.llm_manager = llm_manager
        
        # Initialize session state for model selections
        if "model_selections" not in st.session_state:
            # Initialize with environment variables if available
            st.session_state.model_selections = {
                "generative": os.getenv("GENERATIVE_MODEL", llm_manager.default_model),
                "review": os.getenv("REVIEW_MODEL", llm_manager.default_model),
                "summary": os.getenv("SUMMARY_MODEL", llm_manager.default_model),
                "compare": os.getenv("COMPARE_MODEL", llm_manager.default_model)
            }
            
        # Initialize session state for model operations
        if "model_operations" not in st.session_state:
            st.session_state.model_operations = {
                "pulling": False,
                "current_pull": None,
                "pull_progress": 0,
                "last_pulled": None,
                "error": None
            }
    
    def render_model_card(self, model, gpu_available=False):
        """
        Render a professional model card with GPU status.
        
        Args:
            model: Model information dictionary
            gpu_available: Whether GPU acceleration is available
        """
        # Set card styling based on availability
        card_class = "model-available" if model["pulled"] else "model-not-available"
        badge_class = "badge-available" if model["pulled"] else "badge-not-available"
        status_text = "Available" if model["pulled"] else "Not pulled"
        
        # Add GPU badge if available and model is pulled
        gpu_badge = ""
        if gpu_available and model["pulled"]:
            gpu_badge = '<span class="model-badge badge-gpu">GPU Ready</span>'
        
        st.markdown(f"""
            <div class="model-card {card_class}">
                <div class="model-header">
                    <div>
                        <span class="model-name">{model["name"]}</span>
                        <span class="model-id">({model["id"]})</span>
                    </div>
                    <div>
                        <span class="model-badge {badge_class}">{status_text}</span>
                        {gpu_badge}
                    </div>
                </div>
                <div class="model-description">
                    {model["description"]}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Add pull button for models that aren't pulled
        if not model["pulled"]:
            if st.button(f"Pull {model['id']}", key=f"pull_{model['id']}"):
                st.session_state.model_operations["pulling"] = True
                st.session_state.model_operations["current_pull"] = model["id"]
                st.session_state.model_operations["pull_progress"] = 0
                st.rerun()

    def render_model_manager(self) -> Dict[str, str]:
        """
        Render the Ollama model management UI with improved single column layout
        and fixed text colors for better visibility.
        
        Returns:
            Dictionary with selected models for each role
        """
        st.header("Ollama Model Management")
        
                
        # Check connection to Ollama
        connection_status, message = self.llm_manager.check_ollama_connection()
        
        if not connection_status:
            st.error(f"Cannot connect to Ollama: {message}")
            
            with st.expander("Troubleshooting"):
                st.markdown("""
                1. **Check if Ollama is running:**
                ```bash
                curl http://localhost:11434/api/tags
                ```
                
                2. **Make sure the Ollama service is started:**
                - On Linux/Mac: `ollama serve`
                - On Windows: Start the Ollama application
                
                3. **Check the Ollama URL in .env file:**
                - Default is http://localhost:11434
                """)
            return st.session_state.model_selections
        
        # Get available models
        available_models = self.llm_manager.get_available_models()

        # Check GPU availability
        gpu_info = self.llm_manager.check_gpu_availability()
        gpu_available = gpu_info.get("has_gpu", False)

        # Display GPU status at the top
        if gpu_available:
            st.success(f"🚀 GPU Acceleration Enabled: {gpu_info.get('gpu_name', 'GPU')}")
        else:
            st.warning("⚠️ GPU Acceleration Not Available - Model inference will use CPU only")
            with st.expander("GPU Setup Help"):
                st.markdown("""
                ### Setting up GPU Acceleration for Ollama
                
                1. **Requirements**:
                - NVIDIA GPU with CUDA support (GTX series, RTX series, etc.)
                - Or AMD GPU with ROCm support
                
                2. **Install Drivers**:
                - For NVIDIA: Install CUDA drivers from NVIDIA website
                - For AMD: Install ROCm drivers
                
                3. **Configure Ollama**:
                - Ensure Ollama is set up to use your GPU
                - See [Ollama GPU Documentation](https://github.com/ollama/ollama/blob/main/docs/gpu.md)
                
                4. **Restart Ollama**:
                - After configuration, restart the Ollama service
                """)
        
        # Section 1: Available Models
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Available Models")
        
        if not available_models:
            st.info("No models found. Pull a model to get started.")
        else:
            # Create a filtered list of models to display
            display_models = []
            for model in available_models:
                display_models.append({
                    "name": model["name"],
                    "id": model["id"],
                    "pulled": model["pulled"],
                    "description": model["description"]
                })
            
            # Generate model cards using CSS classes
            for i, model in enumerate(display_models):
                # Set card styling based on availability
                card_class = "model-available" if model["pulled"] else "model-not-available"
                badge_class = "badge-available" if model["pulled"] else "badge-not-available"
                status_text = "Available" if model["pulled"] else "Not pulled"
                
                st.markdown(f"""
                    <div class="model-card {card_class}">
                        <div class="model-header">
                            <div>
                                <span class="model-name">{model["name"]}</span>
                                <span class="model-id">({model["id"]})</span>
                            </div>
                            <span class="model-badge {badge_class}">{status_text}</span>
                        </div>
                        <div class="model-description">
                            {model["description"]}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add pull button for models that aren't pulled
                if not model["pulled"]:
                    if st.button(f"Pull {model['id']}", key=f"pull_{model['id']}"):
                        st.session_state.model_operations["pulling"] = True
                        st.session_state.model_operations["current_pull"] = model["id"]
                        st.session_state.model_operations["pull_progress"] = 0
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section 2: Pull New Model
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Pull New Model")
        
        # When a model is being pulled, show progress
        if st.session_state.model_operations["pulling"]:
            model_name = st.session_state.model_operations["current_pull"]
            st.info(f"Pulling model: {model_name}")
            progress_bar = st.progress(st.session_state.model_operations["pull_progress"])
            
            if st.button("Cancel"):
                st.session_state.model_operations["pulling"] = False
                st.session_state.model_operations["current_pull"] = None
                st.session_state.model_operations["pull_progress"] = 0
                st.rerun()
        else:
            # Model pull form
            with st.form("pull_model_form"):
                model_id = st.text_input(
                    "Model ID", 
                    placeholder="e.g., llama3:8b, phi3:mini, gemma:2b",
                    help="Enter the ID of the model you want to pull from Ollama"
                )
                
                submitted = st.form_submit_button("Pull Model")
                
                if submitted and model_id:
                    st.session_state.model_operations["pulling"] = True
                    st.session_state.model_operations["current_pull"] = model_id
                    st.session_state.model_operations["pull_progress"] = 0
                    st.rerun()
        
        # Show last pulled model
        if st.session_state.model_operations["last_pulled"]:
            st.success(f"Successfully pulled model: {st.session_state.model_operations['last_pulled']}")
        
        # Show error if any
        if st.session_state.model_operations["error"]:
            st.error(st.session_state.model_operations["error"])
            if st.button("Clear Error"):
                st.session_state.model_operations["error"] = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section 3: Model Selection
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model Selection")
        
        # Get model options (only pulled models)
        model_options = [model["id"] for model in available_models if model["pulled"]]
        
        if not model_options:
            st.warning("No models are available. Please pull at least one model.")
        else:
            # Use the improved model selection table with fixed text colors
            model_selections = self.render_model_selection_table(model_options)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Return the current model selections
        return st.session_state.model_selections
    
    def _render_model_table(self, models: List[Dict[str, Any]]):
        """
        Render a table of available models.
        
        Args:
            models: List of model dictionaries
        """
        # Create a more dynamic and interactive model display
        for model in models:
            # Create a colored background based on status
            if model["pulled"]:
                bg_color = "#e8f5e9"  # Light green for pulled models
                border_color = "#4CAF50"
                status_icon = ""
                status_text = "Available"
            else:
                bg_color = "#f5f5f5"  # Light gray for not pulled models
                border_color = "#9e9e9e"
                status_icon = "�"
                status_text = "Not pulled"
            
            # Create a card for each model
            st.markdown(
                f"""
                <div>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <strong>{model["name"]}</strong> 
                            <span style="opacity: 0.7; font-size: 0.9em;">({model["id"]})</span>
                        </div>
                        <div>
                            <span style="background-color: {border_color}; color: white; 
                                  padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">
                                {status_icon} {status_text}
                            </span>
                        </div>
                    </div>
                    <div style="font-size: 0.9em; margin-top: 5px;">
                        {model["description"]}
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Add pull button for models that aren't pulled
            if not model["pulled"]:
                if st.button(f"Pull {model['id']}", key=f"pull_{model['id']}"):
                    st.session_state.model_operations["pulling"] = True
                    st.session_state.model_operations["current_pull"] = model["id"]
                    st.session_state.model_operations["pull_progress"] = 0
                    st.rerun()

    def render_model_selection_table(self, model_options):
        """
        Render a professionally styled model selection table with fixed text colors
        
        Args:
            model_options: List of available model options
        
        Returns:
            Dictionary with selected models for each role
        """        
        st.markdown("<h3 style='color: var(--text);'>Model Configuration</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: var(--text-secondary);'>Select which model to use for each stage of the code review process:</p>", unsafe_allow_html=True)
        
        # Start container for model roles
        st.markdown('<div class="model-selection-container">', unsafe_allow_html=True)
        
        # Define role configurations with icons and descriptions
        role_configs = {
            "generative": {
                "title": "Code Generation",
                "icon": "💻",
                "description": "Creates Java code with intentional errors for review practice",
                "selector_id": "generative-model-cell"
            },
            "review": {
                "title": "Review Analysis",
                "icon": "🔍",
                "description": "Analyzes student reviews to identify discovered and missed issues",
                "selector_id": "review-model-cell"
            },
            "summary": {
                "title": "Feedback Generation",
                "icon": "📊",
                "description": "Creates detailed feedback summaries after review completion",
                "selector_id": "summary-model-cell"
            },
            "compare": {
                "title": "Comparative Analysis",
                "icon": "⚖️",
                "description": "Compares student reviews with known issues for evaluation",
                "selector_id": "compare-model-cell"
            }
        }
        
        # Create a card for each role
        for role, config in role_configs.items():
            st.markdown(f'<div class="model-role">', unsafe_allow_html=True)
            
            # Role header
            st.markdown(f"""
            <div class="role-header">
                <div>
                    <span class="role-title"><span class="role-icon">{config["icon"]}</span>{config["title"]}</span>
                </div>
            </div>
            <div class="role-description">{config["description"]}</div>
            """, unsafe_allow_html=True)
            
            # Add the model selector
            st.markdown(f'<div id="{config["selector_id"]}">', unsafe_allow_html=True)
            selected_model = st.selectbox(
                f"Select model for {config['title']}",
                options=model_options,
                index=model_options.index(st.session_state.model_selections[role]) 
                if st.session_state.model_selections[role] in model_options else 0,
                key=f"{role}_model_select",
                help=f"Choose which model to use for {config['title'].lower()}"
            )
            
            # Update the selection in session state
            st.session_state.model_selections[role] = selected_model
            
            # Show selected model with badge
            model_size_badge = "Large" if "opus" in selected_model.lower() or "70b" in selected_model or "13b" in selected_model or "8b" in selected_model else "Medium" if "sonnet" in selected_model.lower() or "7b" in selected_model else "Small"
            
            st.markdown(f"""
            <div class="selected-model">
                <div>Selected: <strong style="color: var(--text);">{selected_model}</strong></div>
                <span class="model-badge">{model_size_badge}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # End container
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a "Save Configuration" button
        if st.button("💾 Save Configuration", type="primary"):
            # Update environment variables
            for role, model in st.session_state.model_selections.items():
                os.environ[f"{role.upper()}_MODEL"] = model
            
            st.success("Model configuration saved successfully!")
        
        return st.session_state.model_selections