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
    
    def render_model_manager(self) -> Dict[str, str]:
        """
        Render the Ollama model management UI with improved single column layout.
        Uses external CSS files instead of inline styles.
        
        Returns:
            Dictionary with selected models for each role
        """
        st.header("Ollama Model Management")
        
        # No inline CSS needed here - the styles are loaded from the external CSS file
        # using the css_utils.load_css() function at app startup
        
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
        
        # Section 1: Available Models
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Available Models")
        
        if not available_models:
            st.info("No models found. Pull a model to get started.")
        else:
            # Create a filtered list of models to display
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
            # Create a table-style layout for model selection
            st.markdown("""
                <table class="model-selection-table">
                    <thead>
                        <tr>
                            <th style="width: 30%;">Purpose</th>
                            <th>Selected Model</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Code Generation</strong><br><span class="purpose-description">For creating Java code problems</span></td>
                            <td id="generative-model-cell"></td>
                        </tr>
                        <tr>
                            <td><strong>Review Analysis</strong><br><span class="purpose-description">For analyzing student reviews</span></td>
                            <td id="review-model-cell"></td>
                        </tr>
                        <tr>
                            <td><strong>Summary Generation</strong><br><span class="purpose-description">For generating feedback summaries</span></td>
                            <td id="summary-model-cell"></td>
                        </tr>
                        <tr>
                            <td><strong>Comparison</strong><br><span class="purpose-description">For comparing student reviews with actual issues</span></td>
                            <td id="compare-model-cell"></td>
                        </tr>
                    </tbody>
                </table>
            """, unsafe_allow_html=True)
            
            # Add model selection dropdowns to table cells
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div id="generative-model-cell">', unsafe_allow_html=True)
                generative_model = st.selectbox(
                    "Model for generating code problems",
                    options=model_options,
                    index=model_options.index(st.session_state.model_selections["generative"]) 
                    if st.session_state.model_selections["generative"] in model_options else 0,
                    key="generative_model_select",
                    label_visibility="collapsed"
                )
                st.session_state.model_selections["generative"] = generative_model
            
            with col2:
                st.markdown('<div id="review-model-cell">', unsafe_allow_html=True)
                review_model = st.selectbox(
                    "Model for analyzing student reviews",
                    options=model_options,
                    index=model_options.index(st.session_state.model_selections["review"]) 
                    if st.session_state.model_selections["review"] in model_options else 0,
                    key="review_model_select",
                    label_visibility="collapsed"
                )
                st.session_state.model_selections["review"] = review_model
            
            with col3:
                st.markdown('<div id="summary-model-cell">', unsafe_allow_html=True)
                summary_model = st.selectbox(
                    "Model for generating feedback summaries",
                    options=model_options,
                    index=model_options.index(st.session_state.model_selections["summary"]) 
                    if st.session_state.model_selections["summary"] in model_options else 0,
                    key="summary_model_select",
                    label_visibility="collapsed"
                )
                st.session_state.model_selections["summary"] = summary_model
            
            with col4:
                st.markdown('<div id="compare-model-cell">', unsafe_allow_html=True)
                compare_model = st.selectbox(
                    "Model for comparing student reviews with actual issues",
                    options=model_options,
                    index=model_options.index(st.session_state.model_selections["compare"]) 
                    if st.session_state.model_selections["compare"] in model_options else 0,
                    key="compare_model_select",
                    label_visibility="collapsed"
                )
                st.session_state.model_selections["compare"] = compare_model
            
            # Advanced settings expander
            with st.expander("Advanced Settings", expanded=False):
                # Enable/disable reasoning mode
                reasoning_mode = st.checkbox(
                    "Enable Reasoning Mode",
                    value=os.getenv("REASONING_MODE", "false").lower() == "true",
                    help="When enabled, models will use step-by-step reasoning (may use more tokens)"
                )
                
                # Temperature settings
                st.subheader("Temperature Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    generative_temp = st.slider(
                        "Generation Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(os.getenv("GENERATIVE_TEMPERATURE", "0.7")),
                        step=0.1,
                        help="Higher temperature = more creative but less predictable"
                    )
                    
                    review_temp = st.slider(
                        "Review Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(os.getenv("REVIEW_TEMPERATURE", "0.7")),
                        step=0.1
                    )
                
                with col2:
                    summary_temp = st.slider(
                        "Summary Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(os.getenv("SUMMARY_TEMPERATURE", "0.7")),
                        step=0.1
                    )
                    
                    compare_temp = st.slider(
                        "Compare Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(os.getenv("COMPARE_TEMPERATURE", "0.7")),
                        step=0.1
                    )
                
                # Reasoning temperature
                reasoning_temp = st.slider(
                    "Reasoning Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(os.getenv("REASONING_TEMPERATURE", "0.1")),
                    step=0.1,
                    help="Lower values give more predictable responses in reasoning mode"
                )
                
                # Save settings button
                if st.button("Save Settings to .env", type="primary"):
                    # Update environment variables in memory
                    os.environ["GENERATIVE_MODEL"] = st.session_state.model_selections["generative"]
                    os.environ["REVIEW_MODEL"] = st.session_state.model_selections["review"]
                    os.environ["SUMMARY_MODEL"] = st.session_state.model_selections["summary"]
                    os.environ["COMPARE_MODEL"] = st.session_state.model_selections["compare"]
                    
                    os.environ["GENERATIVE_TEMPERATURE"] = str(generative_temp)
                    os.environ["REVIEW_TEMPERATURE"] = str(review_temp)
                    os.environ["SUMMARY_TEMPERATURE"] = str(summary_temp)
                    os.environ["COMPARE_TEMPERATURE"] = str(compare_temp)
                    
                    os.environ["REASONING_MODE"] = str(reasoning_mode).lower()
                    os.environ["REASONING_TEMPERATURE"] = str(reasoning_temp)
                    
                    # Update .env file
                    try:
                        env_path = ".env"
                        if os.path.exists(env_path):
                            # Read existing .env content
                            with open(env_path, "r") as f:
                                lines = f.readlines()
                            
                            # Update existing values or add new ones
                            env_vars = {
                                "GENERATIVE_MODEL": st.session_state.model_selections["generative"],
                                "REVIEW_MODEL": st.session_state.model_selections["review"],
                                "SUMMARY_MODEL": st.session_state.model_selections["summary"],
                                "COMPARE_MODEL": st.session_state.model_selections["compare"],
                                "GENERATIVE_TEMPERATURE": str(generative_temp),
                                "REVIEW_TEMPERATURE": str(review_temp),
                                "SUMMARY_TEMPERATURE": str(summary_temp),
                                "COMPARE_TEMPERATURE": str(compare_temp),
                                "REASONING_MODE": str(reasoning_mode).lower(),
                                "REASONING_TEMPERATURE": str(reasoning_temp)
                            }
                            
                            # Update existing variables
                            updated_lines = []
                            updated_vars = set()
                            
                            for line in lines:
                                updated = False
                                for var_name, var_value in env_vars.items():
                                    if line.startswith(f"{var_name}="):
                                        updated_lines.append(f"{var_name}={var_value}\n")
                                        updated_vars.add(var_name)
                                        updated = True
                                        break
                                
                                if not updated:
                                    updated_lines.append(line)
                            
                            # Add new variables that weren't updated
                            for var_name, var_value in env_vars.items():
                                if var_name not in updated_vars:
                                    updated_lines.append(f"{var_name}={var_value}\n")
                            
                            # Write the updated content back
                            with open(env_path, "w") as f:
                                f.writelines(updated_lines)
                            
                            st.success("Settings saved to .env file!")
                        else:
                            # Create a new .env file
                            with open(env_path, "w") as f:
                                f.write(f"OLLAMA_BASE_URL={self.llm_manager.ollama_base_url}\n")
                                f.write(f"DEFAULT_MODEL={self.llm_manager.default_model}\n")
                                f.write(f"GENERATIVE_MODEL={st.session_state.model_selections['generative']}\n")
                                f.write(f"REVIEW_MODEL={st.session_state.model_selections['review']}\n")
                                f.write(f"SUMMARY_MODEL={st.session_state.model_selections['summary']}\n")
                                f.write(f"COMPARE_MODEL={st.session_state.model_selections['compare']}\n")
                                f.write(f"GENERATIVE_TEMPERATURE={generative_temp}\n")
                                f.write(f"REVIEW_TEMPERATURE={review_temp}\n")
                                f.write(f"SUMMARY_TEMPERATURE={summary_temp}\n")
                                f.write(f"COMPARE_TEMPERATURE={compare_temp}\n")
                                f.write(f"REASONING_MODE={str(reasoning_mode).lower()}\n")
                                f.write(f"REASONING_TEMPERATURE={reasoning_temp}\n")
                            
                            st.success("Created new .env file with settings!")
                    except Exception as e:
                        st.error(f"Error saving settings: {str(e)}")
        
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