"""
Export utilities for Java Peer Review Training System.

This module provides functions for exporting prompts and LLM responses to text files
for debugging and analysis purposes.
"""

import os
import time
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

def export_prompt_response(
    prompt: str, 
    response: str, 
    operation_type: str,
    error_list: Optional[List[Dict[str, Any]]] = None,
    evaluation_result: Optional[Dict[str, Any]] = None
) -> str:
    """
    Export prompt and response to a text file.
    
    Args:
        prompt: The prompt sent to the LLM
        response: The response from the LLM
        operation_type: Type of operation (e.g., "code_generation", "evaluation")
        error_list: Optional list of requested errors
        evaluation_result: Optional evaluation result data
        
    Returns:
        Path to the exported file
    """
    # Ensure export directory exists
    exports_dir = ensure_export_directory()
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{operation_type}_{timestamp}.txt"
    filepath = os.path.join(exports_dir, filename)
    
    # Format the content
    content = f"""OPERATION: {operation_type}
TIMESTAMP: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{'='*80}
PROMPT:
{'='*80}

{prompt}

{'='*80}
RESPONSE:
{'='*80}

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

    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath