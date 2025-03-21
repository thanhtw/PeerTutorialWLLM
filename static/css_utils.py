"""
CSS utility functions for loading and managing CSS in Streamlit applications.
"""
import os
import streamlit as st

def load_css(css_file=None, css_directory=None):
    """
    Load CSS from file or directory into Streamlit.
    
    Args:
        css_file: Path to single CSS file
        css_directory: Path to directory containing CSS files
        
    Returns:
        List of loaded CSS file names or empty list if none loaded
    """
    css_content = ""
    loaded_files = []
    
    # Load single file if specified
    if css_file and os.path.exists(css_file):
        try:
            with open(css_file, 'r') as f:
                css_content += f.read()
                loaded_files.append(os.path.basename(css_file))
        except Exception as e:
            st.error(f"Error loading CSS file {css_file}: {str(e)}")
    
    # Load all CSS files from directory if specified
    if css_directory and os.path.exists(css_directory) and os.path.isdir(css_directory):
        try:
            # First load base.css if it exists
            base_css_path = os.path.join(css_directory, "base.css")
            if os.path.exists(base_css_path):
                with open(base_css_path, 'r') as f:
                    css_content += f.read()
                    loaded_files.append("base.css")
            
            # Then load components.css
            components_css_path = os.path.join(css_directory, "components.css")
            if os.path.exists(components_css_path):
                with open(components_css_path, 'r') as f:
                    css_content += f.read()
                    loaded_files.append("components.css")
            
            # Then load model_manager.css
            model_manager_css_path = os.path.join(css_directory, "model_manager.css")
            if os.path.exists(model_manager_css_path):
                with open(model_manager_css_path, 'r') as f:
                    css_content += f.read()
                    loaded_files.append("model_manager.css")
            
            # Finally load tabs.css
            tabs_css_path = os.path.join(css_directory, "tabs.css")
            if os.path.exists(tabs_css_path):
                with open(tabs_css_path, 'r') as f:
                    css_content += f.read()
                    loaded_files.append("tabs.css")
            
            # Load any remaining CSS files (except main.css which is now obsolete)
            for filename in sorted(os.listdir(css_directory)):
                if (filename.endswith('.css') and 
                    filename not in ["base.css", "components.css", "model_manager.css", "tabs.css", "main.css"]):
                    file_path = os.path.join(css_directory, filename)
                    with open(file_path, 'r') as f:
                        css_content += f.read()
                        loaded_files.append(filename)
                        
        except Exception as e:
            st.error(f"Error loading CSS files from directory {css_directory}: {str(e)}")
    
    # Apply CSS if we loaded any
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        return loaded_files
    
    return []

