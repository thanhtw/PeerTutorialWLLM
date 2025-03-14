"""
Model Manager UI module for Java Peer Review Training System.

This module provides the ModelManagerUI class for managing Ollama models
through the web interface.
"""

import streamlit as st
import re
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
        
        # Ensure GPU info is refreshed
        self.gpu_info = llm_manager.refresh_gpu_info()
    
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
        gpu_class = ""
        if gpu_available and model["pulled"] and model.get("gpu_optimized", False):
            gpu_badge = '<span class="model-badge badge-gpu">GPU Ready</span>'
            gpu_class = "gpu-ready"
        
        # Extract and clean model information
        model_name = model["name"]
        model_id = model["id"]
        
        # Clean the description - strip HTML tags entirely
        import re
        model_description = re.sub(r'<[^>]*>', '', model["description"])
        
        # Render the card
        st.markdown(f"""
            <div class="model-card {card_class} {gpu_class}">
                <div class="model-header">
                    <div>
                        <span class="model-name">{model_name}</span>
                        <span class="model-id">({model_id})</span>
                    </div>
                    <div><div>
                        <span class="model-badge {badge_class}">{status_text}</span>
                        {gpu_badge}
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
        Render the Ollama model management UI with improved GPU information
        and model details.
        
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

        # Check GPU availability with detailed information
        gpu_info = self.llm_manager.refresh_gpu_info()
        gpu_available = gpu_info.get("has_gpu", False)

        # Display GPU status at the top - Enhanced with more details
        if gpu_available:
            gpu_name = gpu_info.get("gpu_name", "GPU")
            memory_total = gpu_info.get("formatted_total", "Unknown")
            memory_used = gpu_info.get("formatted_used", "Unknown")
            memory_percent = gpu_info.get("memory_used_percent", 0)
            utilization = gpu_info.get("utilization", None)
            
            # Create a GPU info card with detailed metrics
            st.markdown(
                f"""
                <div class="gpu-info-section">
                    <div class="gpu-info-header">
                        <span>🚀</span>
                        <div class="gpu-info-title">GPU Acceleration Enabled: {gpu_name}</div>
                    </div>
                    <div class="gpu-info-detail">
                        <span class="gpu-info-label">Memory</span>
                        <span>{memory_used} / {memory_total}</span>
                    </div>
                """
                + (f"""
                    <div class="gpu-info-detail">
                        <span class="gpu-info-label">Memory Usage</span>
                        <span>{memory_percent:.1f}%</span>
                    </div>
                """ if memory_percent else "")
                + (f"""
                    <div class="gpu-info-detail">
                        <span class="gpu-info-label">Utilization</span>
                        <span>{utilization:.1f}%</span>
                    </div>
                """ if utilization is not None else "")
                + """
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("""
            ⚠️ **GPU Acceleration Not Available** - Model inference will use CPU only
            
            For better performance, consider setting up a compatible GPU.
            """)
            
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
                
                4. **Check Config**:
                   - Make sure `ENABLE_GPU=true` is set in your .env file
                   - Check for any GPU-related errors in the console output
                """)
        
        # Section 1: Available Models - Enhanced with GPU info
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Available Models")
        
        if not available_models:
            st.info("No models found. Pull a model to get started.")
        else:
            # Create a filtered list of models to display
            display_models = []
            for model in available_models:
                # Check if model has GPU-specific parameters
                gpu_optimized = model.get("gpu_optimized", False)
                if model["pulled"]:
                    try:
                        model_info = self.llm_manager.get_model_details(model["id"])
                        gpu_optimized = model_info.get("gpu_optimized", gpu_optimized)
                    except:
                        pass
                
                display_models.append({
                    "name": model["name"],
                    "id": model["id"],
                    "pulled": model["pulled"],
                    "description": model["description"],
                    "gpu_optimized": gpu_optimized
                })
            
            # Generate model cards with GPU optimization status
            for i, model in enumerate(display_models):
                self.render_model_card(model, gpu_available)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section 2: Pull New Model
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Pull New Model")
        
        # When a model is being pulled, show progress
        if st.session_state.model_operations["pulling"]:
            model_name = st.session_state.model_operations["current_pull"]
            st.info(f"Pulling model: {model_name}")
            progress_bar = st.progress(st.session_state.model_operations["pull_progress"] / 100)
            
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
        
        # Section 3: Model Selection with GPU information
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Model Selection")
        
        # Get model options (only pulled models)
        model_options = [model["id"] for model in available_models if model["pulled"]]
        
        if not model_options:
            st.warning("No models are available. Please pull at least one model.")
        else:
            # Use the improved model selection table with GPU information
            model_selections = self.render_model_selection_table(model_options, gpu_available)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Section 4: GPU Settings - New section
        if gpu_available:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("GPU Acceleration Settings")
            
            # GPU Layers setting
            current_layers = self.llm_manager.gpu_layers
            gpu_layers = st.slider(
                "GPU Layers", 
                min_value=-1, 
                max_value=100, 
                value=current_layers,
                help="-1 means use all available layers, lower values use less GPU memory"
            )
            
            # GPU Force setting
            force_gpu = st.checkbox(
                "Force GPU Usage", 
                value=self.llm_manager.force_gpu,
                help="When enabled, all models will attempt to use GPU acceleration"
            )
            
            # Save button for GPU settings
            if st.button("Save GPU Settings"):
                # Update LLM Manager settings
                self.llm_manager.gpu_layers = gpu_layers
                self.llm_manager.force_gpu = force_gpu
                
                # Update environment variables
                os.environ["GPU_LAYERS"] = str(gpu_layers)
                os.environ["ENABLE_GPU"] = "true" if force_gpu else "false"
                
                # Show success message
                st.success("GPU settings saved successfully!")
                
                # Clear initialized models to apply new settings
                self.llm_manager.initialized_models = {}
                
                # Rerun the app
                st.rerun()
                
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

    def render_model_selection_table(self, model_options, gpu_available=False):
        """
        Render a professionally styled model selection table with GPU status information
        
        Args:
            model_options: List of available model options
            gpu_available: Whether GPU acceleration is available
        
        Returns:
            Dictionary with selected models for each role
        """        
        st.markdown("<h3 style='color: var(--text);'>Model Configuration</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: var(--text-secondary);'>Select which model to use for each stage of the code review process:</p>", unsafe_allow_html=True)
        
        # Start container for model roles
        st.markdown('<div class="model-selection-container">', unsafe_allow_html=True)
        
        # Get active models with GPU info
        active_models = self.llm_manager.get_active_models()
        
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
            # Check if this role's model uses GPU
            role_model_info = active_models.get(role, {})
            uses_gpu = role_model_info.get("uses_gpu", False) or role_model_info.get("gpu_optimized", False)
            
            # Add GPU class if applicable
            gpu_class = "gpu-enabled" if uses_gpu and gpu_available else ""
            
            st.markdown(f'<div class="model-role {gpu_class}">', unsafe_allow_html=True)
            
            # Role header
            st.markdown(f"""
            <div class="role-header">
                <div>
                    <span class="role-title"><span class="role-icon">{config["icon"]}</span>{config["title"]}</span>
                    {('<span class="gpu-model-highlight">GPU</span>' if uses_gpu and gpu_available else '')}
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
            
            # Show selected model with badge and GPU status
            model_size_badge = "Large" if "opus" in selected_model.lower() or "70b" in selected_model or "13b" in selected_model or "8b" in selected_model else "Medium" if "sonnet" in selected_model.lower() or "7b" in selected_model else "Small"
            
            # Check if the selected model is GPU-optimized
            gpu_optimized = False
            for model in self.llm_manager.get_available_models():
                if model["id"] == selected_model and model["pulled"]:
                    gpu_optimized = model.get("gpu_optimized", False)
                    break
            
            # Add GPU badge if applicable
            gpu_badge = '<span class="model-badge badge-gpu">GPU</span>' if gpu_optimized and gpu_available else ''
            
            st.markdown(f"""
            <div class="selected-model">
                <div>Selected: <strong style="color: var(--text);">{selected_model}</strong></div>
                <div>
                    <span class="model-badge">{model_size_badge}</span>
                    {gpu_badge}
                </div>
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
            
            # Clear initialized models to apply new settings
            self.llm_manager.initialized_models = {}
            
            # Rerun to apply changes
            st.rerun()
        
        return st.session_state.model_selections