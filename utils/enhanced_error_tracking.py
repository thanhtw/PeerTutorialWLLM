"""
Enhanced error tracking module for the Java Code Review Training System.

This module provides functions for extracting and enriching error information
from generated code to improve the review evaluation process.
"""

import re
from typing import List, Dict, Any, Tuple, Optional

def extract_error_locations(
    code: str,
    errors: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract line numbers and context for errors in generated code.
    
    Args:
        code: The Java code with errors
        errors: List of error dictionaries from the error repository
        
    Returns:
        Enhanced error list with location information
    """
    enhanced_errors = []
    lines = code.splitlines()
    
    for error in errors:
        error_info = error.copy()
        
        # Default values if we can't find location information
        error_info["line_number"] = None
        error_info["line_content"] = None
        error_info["context"] = None
        
        # Try to find potential locations for this error
        error_type = error.get("type", "").lower()
        error_name = error.get("name", "").lower()
        
        # Look for both explicit error markers and code patterns
        found_locations = []
        
        # 1. Look for explicit error comment markers in the code
        for i, line in enumerate(lines):
            # Check for explicit annotations like "// ERROR TYPE: ERROR NAME" or "// TODO: Fix"
            if (("error" in line.lower() and error_name in line.lower()) or 
                ("todo" in line.lower() and error_name in line.lower())):
                # Found an annotation - check the next few lines for the actual error
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 3)
                found_locations.append({
                    "line_number": i + 1,  # 1-based line numbering
                    "line_content": line.strip(),
                    "context": "\n".join(lines[context_start:context_end]),
                    "confidence": 0.9  # High confidence for explicit annotations
                })
        
        # 2. Try to identify error patterns in code based on error types/names
        if error_type == "build":
            if "nullpointer" in error_name or "null pointer" in error_name:
                # Look for potential null pointer issues (object access without null check)
                for i, line in enumerate(lines):
                    if "." in line and "null" not in line.lower() and "check" not in line.lower() and "if" not in line.lower():
                        # Simple heuristic for potential NPE
                        context_start = max(0, i - 2)
                        context_end = min(len(lines), i + 3)
                        found_locations.append({
                            "line_number": i + 1,
                            "line_content": line.strip(),
                            "context": "\n".join(lines[context_start:context_end]),
                            "confidence": 0.6  # Medium confidence
                        })
            
            elif "missing return" in error_name.lower():
                # Look for methods that might be missing returns
                for i, line in enumerate(lines):
                    if "void" not in line.lower() and any(type_name in line for type_name in ["int", "String", "boolean", "Object", "List"]) and "(" in line and ")" in line and "{" in line:
                        # Looks like a method declaration with a return type
                        # Check if method has returns in all branches
                        context_start = max(0, i - 1)
                        # Find the method end
                        brace_count = 0
                        found_return = False
                        for j, method_line in enumerate(lines[i:], i):
                            if "{" in method_line:
                                brace_count += method_line.count("{")
                            if "}" in method_line:
                                brace_count -= method_line.count("}")
                            if "return" in method_line:
                                found_return = True
                            if brace_count == 0:
                                # End of method
                                context_end = j + 1
                                break
                        
                        if not found_return:
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": "\n".join(lines[context_start:context_end]),
                                "confidence": 0.7
                            })
        
        elif error_type == "checkstyle":
            if "naming" in error_name.lower() or "convention" in error_name.lower():
                # Look for naming convention issues
                for i, line in enumerate(lines):
                    if "class" in line or "interface" in line or "enum" in line:
                        # Check class naming (should be UpperCamelCase)
                        words = line.split()
                        for j, word in enumerate(words):
                            if word in ["class", "interface", "enum"] and j+1 < len(words):
                                class_name = words[j+1].split("{")[0].strip()
                                if class_name and (class_name[0].islower() or "_" in class_name):
                                    found_locations.append({
                                        "line_number": i + 1,
                                        "line_content": line.strip(),
                                        "context": line.strip(),
                                        "confidence": 0.8
                                    })
                    
                    # Check variable naming
                    var_declaration = re.search(r'\b(int|double|float|String|boolean|long|char)\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
                    if var_declaration:
                        var_name = var_declaration.group(2)
                        if var_name[0].isupper() or "_" in var_name:
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": line.strip(),
                                "confidence": 0.7
                            })
        
        # If no locations were found, use a generic approach
        if not found_locations:
            # Default to a generic location for this error
            # Find a representative line in the code where this error might occur
            keywords = error_name.lower().split()
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in keywords):
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 3)
                    found_locations.append({
                        "line_number": i + 1,
                        "line_content": line.strip(),
                        "context": "\n".join(lines[context_start:context_end]),
                        "confidence": 0.4  # Lower confidence
                    })
        
        # Select the best location (highest confidence)
        if found_locations:
            best_location = max(found_locations, key=lambda loc: loc["confidence"])
            error_info["line_number"] = best_location["line_number"]
            error_info["line_content"] = best_location["line_content"]
            error_info["context"] = best_location["context"]
        
        enhanced_errors.append(error_info)
    
    return enhanced_errors

def generate_problem_descriptions(enhanced_errors: List[Dict[str, Any]]) -> List[str]:
    """
    Generate detailed problem descriptions from enhanced error information.
    
    Args:
        enhanced_errors: List of error dictionaries with location information
        
    Returns:
        List of formatted problem descriptions
    """
    problem_descriptions = []
    
    for error in enhanced_errors:
        error_type = error.get("type", "unknown").upper()
        name = error.get("name", "unknown error")
        description = error.get("description", "")
        line_number = error.get("line_number", "unknown line")
        line_content = error.get("line_content", "")
        
        # Create a detailed problem description
        if line_number and line_content:
            problem = f"{error_type} ERROR - {name}: {description} (Line {line_number}: '{line_content}')"
        else:
            problem = f"{error_type} ERROR - {name}: {description}"
        
        problem_descriptions.append(problem)
    
    return problem_descriptions

def enrich_error_information(
    code: str, 
    selected_errors: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Enrich error information with location data and generate problem descriptions.
    
    Args:
        code: Generated Java code with errors
        selected_errors: Original errors from the repository
        
    Returns:
        Tuple of (enhanced_errors, problem_descriptions)
    """
    # Extract and add location information to errors
    enhanced_errors = extract_error_locations(code, selected_errors)
    
    # Generate detailed problem descriptions
    problem_descriptions = generate_problem_descriptions(enhanced_errors)
    
    return enhanced_errors, problem_descriptions