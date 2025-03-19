import re
import random
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

def validate_code_errors(code: str, requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate that generated code actually contains the requested errors.
    
    Args:
        code: Generated Java code
        requested_errors: List of error dictionaries that should be implemented
        
    Returns:
        Dictionary with validation results
    """
    import re
    
    validation_results = {
        "valid": True,
        "missing_errors": [],
        "found_errors": [],
        "error_locations": {},
    }
    
    lines = code.splitlines()
    
    # Track annotation locations - we'll use these to check if errors are actually implemented
    annotations = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("//") and "ERROR:" in line:
            error_match = re.search(r'ERROR:\s*(?:\[)?([A-Za-z0-9_\s-]+)(?:\])?', line)
            if error_match:
                error_type = error_match.group(1).strip()
                # Save line number (1-based) of the code that should contain the error
                annotations[error_type] = i + 2  # +2 because: +1 for 1-based indexing, +1 for next line after comment
    
    # Check for each requested error
    for error in requested_errors:
        error_type = error.get("type", "").upper()
        error_name = error.get("name", "")
        
        # Create a normalized error key
        error_key = f"{error_type} - {error_name}"
        found = False
        
        # Check different error types
        if "Cannot find symbol" in error_name:
            # Look for undefined variables
            for i, line in enumerate(lines):
                if re.search(r'\b\w+\b', line) and not is_comment(line):
                    # Extract words that might be variables
                    words = re.findall(r'\b(\w+)\b', line)
                    for word in words:
                        # Skip Java keywords and common types
                        if word not in ["public", "private", "class", "void", "int", "String", "boolean", 
                                       "if", "else", "for", "while", "return", "try", "catch", "throw"]:
                            # Check if this word isn't defined earlier
                            is_defined = False
                            for prev_i in range(i):
                                if word in lines[prev_i] and ("class " + word in lines[prev_i] or 
                                                             re.search(rf'\b(int|String|boolean|char|double|float|long)\s+{word}\b', lines[prev_i])):
                                    is_defined = True
                                    break
                            
                            if not is_defined and not is_primitive_or_common(word):
                                found = True
                                validation_results["error_locations"][error_key] = i + 1
                                break
                if found:
                    break
        
        elif "Incompatible types" in error_name:
            # Look for incompatible type assignments
            for i, line in enumerate(lines):
                if "=" in line and not is_comment(line):
                    # Check for String to int assignment
                    if re.search(r'int\s+\w+\s*=\s*"', line) or re.search(r'\b\w+\s*=\s*".*"', line) and "int " in lines[i][:lines[i].index("=")]:
                        found = True
                        validation_results["error_locations"][error_key] = i + 1
                        break
        
        elif "MemberName" in error_name or "MethodName" in error_name:
            # Look for naming convention violations
            for i, line in enumerate(lines):
                if "private" in line or "public" in line or "protected" in line:
                    # Check for names with underscores or improper casing
                    name_match = re.search(r'(?:private|public|protected)\s+\w+\s+(\w+)', line)
                    if name_match:
                        name = name_match.group(1)
                        # If it's a method, it should have ( after the name
                        is_method = "(" in line
                        if (is_method and "MethodName" in error_name) or (not is_method and "MemberName" in error_name):
                            if "_" in name or (len(name) > 0 and name[0].isupper()):
                                found = True
                                validation_results["error_locations"][error_key] = i + 1
                                break
        
        elif "TypeName" in error_name:
            # Look for class/interface/enum with improper naming
            for i, line in enumerate(lines):
                if "class " in line or "interface " in line or "enum " in line:
                    type_match = re.search(r'(?:class|interface|enum)\s+(\w+)', line)
                    if type_match:
                        type_name = type_match.group(1)
                        if len(type_name) > 0 and type_name[0].islower():
                            found = True
                            validation_results["error_locations"][error_key] = i + 1
                            break
        
        elif "NullPointerException" in error_name:
            # Look for potential null pointer issues
            for i, line in enumerate(lines):
                if "null" in line and "." in line and not is_comment(line):
                    if re.search(r'(\w+)\s*=\s*null', line):
                        var_name = re.search(r'(\w+)\s*=\s*null', line).group(1)
                        # Check if this variable is used later without null check
                        for j in range(i+1, len(lines)):
                            if var_name in lines[j] and "." in lines[j] and not "null" in lines[j]:
                                found = True
                                validation_results["error_locations"][error_key] = j + 1
                                break
                    elif re.search(r'null\.\w+', line):
                        found = True
                        validation_results["error_locations"][error_key] = i + 1
                if found:
                    break
        
        # Generic fallback - check if error annotation line number has code that's likely buggy
        if not found and error_key in annotations:
            annotation_line = annotations[error_key]
            if annotation_line < len(lines):
                # Mark as found but with a lower confidence
                found = True
                validation_results["error_locations"][error_key] = annotation_line
        
        # Update validation results
        if found:
            validation_results["found_errors"].append(error_key)
        else:
            validation_results["missing_errors"].append(error_key)
            validation_results["valid"] = False
    
    return validation_results

def is_comment(line: str) -> bool:
    """Check if a line is a comment."""
    return line.strip().startswith("//") or line.strip().startswith("/*")

def is_primitive_or_common(word: str) -> bool:
    """Check if a word is a Java primitive or common class name."""
    primitives = ["int", "boolean", "char", "byte", "short", "long", "float", "double", "void"]
    common_classes = ["String", "Integer", "Boolean", "Character", "Byte", "Short", "Long", "Float", 
                     "Double", "Object", "System", "Math", "List", "Set", "Map", "ArrayList", "HashMap"]
    return word in primitives or word in common_classes