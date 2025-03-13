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
            
            # Then load all other CSS files
            for filename in sorted(os.listdir(css_directory)):
                if filename.endswith('.css') and filename != "base.css":
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

def inject_custom_css():
    """Force injection of critical CSS directly."""
    st.markdown("""
    <style>
    /* Critical styling for dark mode compatibility */
    .stTabs [data-baseweb="tab-list"] {
      gap: 10px;
      background-color: #262730;
      padding: 10px 10px 0 10px;
      border-radius: 10px 10px 0 0;
    }
    
    .stTabs [data-baseweb="tab"] {
      height: 50px;
      background-color: #3b4253;
      border-radius: 8px 8px 0 0;
      padding: 8px 16px;
      font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
      background-color: #4c68d7;
      color: white;
    }
    
    .tab-label {
      display: flex;
      align-items: center;
    }
    
    .tab-number {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      background-color: rgba(255, 255, 255, 0.2);
      border-radius: 50%;
      margin-right: 8px;
      font-weight: bold;
    }
    
    .model-card {
      background-color: #1e1e1e;
      border-left: 5px solid #ccc;
      padding: 12px;
      margin-bottom: 12px;
      border-radius: 4px;
    }
    
    .model-available {
      border-left-color: #4CAF50;
      background-color: rgba(76, 175, 80, 0.1);
    }
    
    button[data-testid="baseButton-primary"] {
      background-color: #4c68d7 !important;
      color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)