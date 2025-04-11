"""
Enhanced export utilities for Java Peer Review Training System.

This module provides improved functions for exporting prompts and LLM responses
to consolidated text files based on session ID for better organization.
"""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

def ensure_export_directory() -> str:
    """
    Ensure that the export directory exists.
    
    Returns:
        Path to the export directory
    """
    # Create exports directory in the project root
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exports_dir = os.path.join(current_dir, "exports")
    
    if not os.path.exists(exports_dir):
        os.makedirs(exports_dir)
        
    return exports_dir

def generate_session_id() -> str:
    """
    Generate a unique session ID.
    
    Returns:
        Unique session ID string
    """
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

def export_prompt_response(
    prompt: str, 
    response: str, 
    operation_type: str,
    session_id: Optional[str] = None,
    error_list: Optional[List[Dict[str, Any]]] = None,
    evaluation_result: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export prompt and response to a consolidated text file based on session ID.
    
    Args:
        prompt: The prompt sent to the LLM
        response: The response from the LLM
        operation_type: Type of operation (e.g., "code_generation", "evaluation")
        session_id: Session ID for grouping related exports
        error_list: Optional list of requested errors
        evaluation_result: Optional evaluation result data
        
    Returns:
        Path to the exported file
    """
    # Ensure export directory exists
    exports_dir = ensure_export_directory()
    
    # Use session ID for filename if provided, otherwise use timestamp
    if session_id:
        filename = f"session_{session_id}.txt"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{operation_type}_{timestamp}.txt"
    
    filepath = os.path.join(exports_dir, filename)
    
    # Format the content with clear section headers for easy reading
    content = f"""
{'='*80}
OPERATION: {operation_type}
TIMESTAMP: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}

PROMPT:
{prompt}

RESPONSE:
{response}
"""

    # Add error list if provided
    if error_list:
        content += f"""
{'='*80}
REQUESTED ERRORS:
{'='*80}

"""
        for i, error in enumerate(error_list, 1):
            error_type = error.get("type", "unknown").upper()
            name = error.get("name", "unknown")
            description = error.get("description", "")
            
            content += f"{i}. {error_type} - {name}\n   Description: {description}\n\n"

    # Add evaluation result if provided
    if evaluation_result:
        content += f"""
{'='*80}
EVALUATION RESULTS:
{'='*80}

Valid: {evaluation_result.get("valid", False)}

Found Errors ({len(evaluation_result.get("found_errors", []))})
------------
"""
        for i, error in enumerate(evaluation_result.get("found_errors", []), 1):
            content += f"{i}. {error}\n"
            
        content += f"""
Missing Errors ({len(evaluation_result.get("missing_errors", []))})
--------------
"""
        for i, error in enumerate(evaluation_result.get("missing_errors", []), 1):
            content += f"{i}. {error}\n"
            
        # Add feedback if available
        if "feedback" in evaluation_result:
            content += f"""
Feedback:
---------
{evaluation_result["feedback"]}
"""

    # Append to file if it exists and session_id is provided, otherwise create new
    mode = 'a' if os.path.exists(filepath) and session_id else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(content)
    
    return filepath

def get_session_log(session_id: str) -> Optional[str]:
    """
    Get the contents of a session log file.
    
    Args:
        session_id: Session ID to retrieve log for
        
    Returns:
        Contents of the log file or None if not found
    """
    exports_dir = ensure_export_directory()
    filepath = os.path.join(exports_dir, f"session_{session_id}.txt")
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    return None

def list_session_files() -> List[str]:
    """
    List all session files in the exports directory.
    
    Returns:
        List of session filenames
    """
    exports_dir = ensure_export_directory()
    return [f for f in os.listdir(exports_dir) if f.startswith("session_") and f.endswith(".txt")]