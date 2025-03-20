"""
Error validation module for Java Peer Review Training System.

This module provides functions to validate that generated Java code
actually contains the required errors.
"""

import re
import random
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_code_errors(code: str, requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate that generated code actually contains the requested errors.
    
    Args:
        code: Generated Java code
        requested_errors: List of error dictionaries that should be implemented
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "valid": False,
        "missing_errors": [],
        "found_errors": [],
        "error_locations": {},
        "debug_info": {}  # Add debug information about detection attempts
    }
    
    # Handle edge cases
    if not code or not requested_errors:
        logger.warning("Empty code or no requested errors provided for validation")
        validation_results["missing_errors"] = [f"{error.get('type', '').upper()} - {error.get('name', '')}" 
                               for error in requested_errors]
        return validation_results
    
    lines = code.splitlines()
    logger.info(f"Validating code with {len(lines)} lines for {len(requested_errors)} requested errors")
    
    # Create debug dictionary to track detection attempts for each error
    debug_info = {}
    
    # Check for each requested error
    for error in requested_errors:
        error_type = error.get("type", "").upper()
        error_name = error.get("name", "")
        
        # Skip errors with missing name or type
        if not error_name or not error_type:
            logger.warning(f"Skipping error with missing name or type: {error}")
            continue
        
        # Create a normalized error key
        error_key = f"{error_type} - {error_name}"
        
        # Initialize debug tracking for this error
        debug_info[error_key] = {
            "detection_attempts": [],
            "detection_methods_tried": [],
            "found": False,
            "location": None
        }
        
        # Start with annotation detection for all error types
        annotation_location = find_error_from_annotations(lines, error_type, error_name)
        if annotation_location:
            validation_results["found_errors"].append(error_key)
            validation_results["error_locations"][error_key] = annotation_location
            debug_info[error_key]["found"] = True
            debug_info[error_key]["location"] = annotation_location
            debug_info[error_key]["detection_methods_tried"].append("annotation")
            debug_info[error_key]["detection_attempts"].append({
                "method": "annotation", 
                "success": True, 
                "line": annotation_location
            })
            # Continue to next error since we found this one
            continue
        else:
            debug_info[error_key]["detection_methods_tried"].append("annotation")
            debug_info[error_key]["detection_attempts"].append({
                "method": "annotation", 
                "success": False
            })
        
        # Use error-specific detection if annotation not found
        # Get specific detection methods based on error name and type
        detection_methods = get_detection_methods(error_name, error_type, error.get("description", ""))
        
        found = False
        for method_name, method_func in detection_methods:
            debug_info[error_key]["detection_methods_tried"].append(method_name)
            
            try:
                location = method_func(lines, error)
                if location:
                    validation_results["found_errors"].append(error_key)
                    validation_results["error_locations"][error_key] = location
                    debug_info[error_key]["found"] = True
                    debug_info[error_key]["location"] = location
                    debug_info[error_key]["detection_attempts"].append({
                        "method": method_name, 
                        "success": True, 
                        "line": location
                    })
                    found = True
                    break
                else:
                    debug_info[error_key]["detection_attempts"].append({
                        "method": method_name, 
                        "success": False
                    })
            except Exception as e:
                logger.warning(f"Error in detection method {method_name} for {error_key}: {str(e)}")
                debug_info[error_key]["detection_attempts"].append({
                    "method": method_name, 
                    "success": False,
                    "error": str(e)
                })
        
        # If error not found with specific methods, add to missing errors
        if not found:
            validation_results["missing_errors"].append(error_key)
        
    # Set valid flag based on whether we found all errors
    validation_results["valid"] = len(validation_results["found_errors"]) == len(requested_errors)
    validation_results["debug_info"] = debug_info
    
    # Log results
    logger.info(f"Validation found {len(validation_results['found_errors'])} of {len(requested_errors)} requested errors")
    for error in validation_results["found_errors"]:
        line = validation_results["error_locations"].get(error, "unknown")
        logger.info(f"  Found: {error} at line {line}")
    for error in validation_results["missing_errors"]:
        logger.info(f"  Missing: {error}")
    
    return validation_results

def find_error_from_annotations(lines: List[str], error_type: str, error_name: str) -> Optional[int]:
    """
    Find error location from error annotations in the code.
    
    Args:
        lines: List of code lines
        error_type: Type of error to find
        error_name: Name of error to find
        
    Returns:
        Line number (1-based) or None if not found
    """
    error_type_lower = error_type.lower()
    error_name_lower = error_name.lower()
    
    # Check different annotation formats
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Skip non-comments
        if not line_lower.startswith("//"):
            continue
            
        # Check standard error annotation format
        if "// error:" in line_lower:
            if error_type_lower in line_lower and error_name_lower in line_lower:
                # Return the line after the annotation (where the actual error is)
                return i + 2 if i + 1 < len(lines) else i + 1
        
        # Check other common annotation formats
        if "// todo:" in line_lower or "// fixme:" in line_lower:
            if error_type_lower in line_lower and error_name_lower in line_lower:
                return i + 2 if i + 1 < len(lines) else i + 1
                
        # Check for error name directly in comment
        if error_name_lower in line_lower and (
            "error" in line_lower or 
            "issue" in line_lower or 
            "problem" in line_lower or
            "bug" in line_lower
        ):
            return i + 2 if i + 1 < len(lines) else i + 1
            
    return None

def get_detection_methods(error_name: str, error_type: str, error_description: str = "") -> List[Tuple[str, callable]]:
    """
    Get appropriate detection methods for a specific error.
    
    Args:
        error_name: Name of the error
        error_type: Type of the error
        error_description: Description of the error
        
    Returns:
        List of (method_name, method_function) tuples
    """
    methods = []
    
    # Convert to lowercase for case-insensitive matching
    error_name_lower = error_name.lower()
    error_type_lower = error_type.lower()
    error_description_lower = error_description.lower()
    
    # Common checks for all errors
    methods.append(("generic_error_detection", detect_generic_error))
    
    # Build error type specific methods
    if error_type_lower == "build":
        # Cannot find symbol
        if "cannot find symbol" in error_name_lower or "symbol" in error_name_lower:
            methods.insert(0, ("cannot_find_symbol", detect_cannot_find_symbol))
            
        # Incompatible types
        if "incompatible types" in error_name_lower or "type mismatch" in error_name_lower:
            methods.insert(0, ("incompatible_types", detect_incompatible_types))
            
        # Missing return statement
        if "missing return" in error_name_lower or "return statement" in error_name_lower:
            methods.insert(0, ("missing_return", detect_missing_return))
        
        # Unreported exception
        if "exception" in error_name_lower and ("unreported" in error_name_lower or "checked" in error_name_lower):
            methods.insert(0, ("unreported_exception", detect_unreported_exception))
            
        # NullPointerException
        if "null" in error_name_lower or "nullpointer" in error_name_lower:
            methods.insert(0, ("null_pointer", detect_null_pointer))
            
        # String comparison using ==
        if ("string" in error_name_lower or "equal" in error_name_lower) and "==" in error_name_lower:
            methods.insert(0, ("string_equality", detect_string_comparison))
    
    # Checkstyle error type specific methods
    elif error_type_lower == "checkstyle":
        # TypeName errors
        if "typename" in error_name_lower or "classname" in error_name_lower:
            methods.insert(0, ("typename", detect_typename))
            
        # MemberName errors
        if "membername" in error_name_lower or "variablename" in error_name_lower:
            methods.insert(0, ("membername", detect_membername))
            
        # MethodName errors
        if "methodname" in error_name_lower:
            methods.insert(0, ("methodname", detect_methodname))
            
        # Whitespace errors
        if "whitespace" in error_name_lower or "spacing" in error_name_lower:
            methods.insert(0, ("whitespace", detect_whitespace))
            
        # Unnecessary imports
        if "unused import" in error_name_lower or "redundant import" in error_name_lower:
            methods.insert(0, ("unused_import", detect_unused_import))
    
    # Detect by description if available
    if error_description:
        if "null" in error_description_lower and "pointer" in error_description_lower:
            methods.insert(0, ("null_pointer_from_description", detect_null_pointer))
            
        if "return" in error_description_lower and "statement" in error_description_lower:
            methods.insert(0, ("missing_return_from_description", detect_missing_return))
            
        if "string" in error_description_lower and "==" in error_description_lower:
            methods.insert(0, ("string_equality_from_description", detect_string_comparison))
    
    return methods

def detect_cannot_find_symbol(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'Cannot find symbol' error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    # Look for variable usages that aren't defined
    defined_vars = set()
    
    # First collect all variable declarations
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Look for variable declarations
        var_decl = re.findall(r'\b(int|double|float|char|boolean|String|byte|short|long)\s+(\w+)\s*[=;]', line)
        for type_name, var_name in var_decl:
            defined_vars.add(var_name)
            
        # Also add loop variables
        for_loop_vars = re.findall(r'for\s*\(\s*\w+\s+(\w+)\s*:', line)
        defined_vars.update(for_loop_vars)
        
        # Add method parameters
        if "(" in line and ")" in line and not ";" in line:
            params = re.findall(r'\(\s*(.*?)\s*\)', line)
            if params:
                param_str = params[0]
                param_parts = param_str.split(',')
                for part in param_parts:
                    part = part.strip()
                    if part and " " in part:
                        var_name = part.split()[-1].replace(")", "")
                        defined_vars.add(var_name)
    
    # Then look for usages of undefined variables
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Skip declarations
        if "=" in line and any(type_name in line for type_name in ["int", "double", "float", "char", "boolean", "String", "byte", "short", "long"]):
            continue
            
        # Look for variable usages
        words = re.findall(r'\b([a-zA-Z]\w*)\b', line)
        for word in words:
            # Skip keywords, types, and known variables
            if (word not in JAVA_KEYWORDS and
                word not in COMMON_JAVA_TYPES and
                word not in defined_vars and
                not word[0].isupper()):  # Skip class names (start with uppercase)
                
                # Skip if it's a method call with parenthesis
                if f"{word}(" in line:
                    continue
                    
                # Found an undefined variable
                return i + 1
    
    return None

def detect_incompatible_types(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'Incompatible types' error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    # Common incompatible type patterns
    patterns = [
        # String to numeric
        (r'(int|double|float|byte|short|long)\s+\w+\s*=\s*["\'"]', "String assigned to numeric"),
        
        # Numeric to boolean
        (r'boolean\s+\w+\s*=\s*\d+', "Numeric assigned to boolean"),
        
        # Object cast to incompatible type
        (r'\(\s*(String|Integer)\s*\)\s*[^"\']*\s*[a-zA-Z][a-zA-Z0-9]*', "Invalid cast"),
        
        # Returning wrong type
        (r'return\s+["\'"].*?["\'"];\s*}\s*(\/\/.*?)?\s*public\s+int', "String returned from int method"),
        
        # Direct type check
        (r'=\s*\(\s*\w+\s*\)\s*\w+', "Casting between incompatible types")
    ]
    
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Check for incompatible type patterns
        for pattern, desc in patterns:
            if re.search(pattern, line):
                return i + 1
    
    # Check for String == comparison which is a special case of type error
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        if ("==" in line or "!=" in line) and ("\"" in line or "'" in line or "String" in line):
            # Make sure it's not comparing with null
            if not "null" in line:
                return i + 1
    
    return None

def detect_missing_return(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'Missing return statement' error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    # Join lines to handle multi-line method declarations
    code = "\n".join(lines)
    
    # Find all method declarations with non-void return types
    methods = re.finditer(r'(public|private|protected)\s+(?!void)(\w+)(?:\<.*?\>)?\s+(\w+)\s*\((.*?)\)\s*\{', code, re.DOTALL)
    
    for match in methods:
        return_type = match.group(2)
        method_name = match.group(3)
        start_pos = match.start()
        
        # Skip constructors (name matches class name)
        if return_type == method_name:
            continue
            
        # Find the matching closing brace
        open_braces = 1
        pos = start_pos + len(match.group(0))
        end_pos = -1
        
        while pos < len(code) and open_braces > 0:
            if code[pos] == '{':
                open_braces += 1
            elif code[pos] == '}':
                open_braces -= 1
                if open_braces == 0:
                    end_pos = pos
            pos += 1
            
        if end_pos == -1:
            continue
            
        # Extract method body
        method_body = code[start_pos:end_pos+1]
        
        # Check if there's a return statement for all paths
        # We'll simplify and just check for presence of a return statement
        if "return" not in method_body:
            # Find the line number
            line_count = code[:start_pos].count('\n') + 1
            return line_count
            
        # Check for if-else blocks without return in some paths
        if "if" in method_body and "return" in method_body:
            # Simple heuristic - if there's an if and a return but no else with return
            if "if" in method_body and "return" in method_body and not re.search(r'}\s*else\s*{.*return', method_body, re.DOTALL):
                # Find the line number
                line_count = code[:start_pos].count('\n') + 1
                return line_count
    
    return None

def detect_unreported_exception(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'Unreported exception' error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    # Check for checked exceptions that aren't caught or declared
    checked_exceptions = [
        "IOException", "FileNotFoundException", "SQLException",
        "ClassNotFoundException", "CloneNotSupportedException",
        "InterruptedException"
    ]
    
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Look for exception throwing statements
        for exc in checked_exceptions:
            if f"throw new {exc}" in line or f"throws {exc}" in line:
                # Check if method has throws declaration
                method_start = None
                for j in range(i, -1, -1):
                    if "public" in lines[j] or "private" in lines[j] or "protected" in lines[j]:
                        if "(" in lines[j] and ")" in lines[j]:
                            method_start = j
                            break
                
                if method_start is not None:
                    # Check if the method declares this exception
                    method_line = lines[method_start]
                    if "throws" not in method_line or exc not in method_line:
                        # Check if this exception is caught
                        is_caught = False
                        for j in range(method_start, i):
                            if "try" in lines[j] and "{" in lines[j]:
                                for k in range(j, i):
                                    if "catch" in lines[k] and exc in lines[k]:
                                        is_caught = True
                                        break
                                if is_caught:
                                    break
                        
                        if not is_caught:
                            return i + 1
    
    return None

def detect_null_pointer(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'NullPointerException' risk.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    # Pattern 1: Direct null access
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        if "null." in line or "null;" in line:
            return i + 1
    
    # Pattern 2: Setting a variable to null then using it
    var_null_assignments = {}
    
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Find null assignments
        null_assigns = re.findall(r'(\w+)\s*=\s*null', line)
        for var in null_assigns:
            var_null_assignments[var] = i
            
        # Check for usages of previously nulled variables
        for var, assign_line in list(var_null_assignments.items()):
            pattern = rf'\b{re.escape(var)}\s*\.\s*\w+'
            if re.search(pattern, line) and i > assign_line:
                # Check if there's a null check between assignment and usage
                has_null_check = False
                for j in range(assign_line + 1, i):
                    if f"if ({var} " in lines[j] or f"if({var} " in lines[j] or f"if ({var}!" in lines[j] or f"if({var}!" in lines[j]:
                        has_null_check = True
                        break
                
                if not has_null_check:
                    return i + 1
    
    return None

def detect_string_comparison(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'String comparison using ==' error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Look for == or != with strings
        if ("==" in line or "!=" in line) and ("String" in line or "\"" in line or "'" in line):
            # Make sure we're not comparing with null (which is valid with ==)
            if "null" not in line:
                # Make sure this is a string comparison, not a numeric one
                if not re.search(r'\d+\s*==|\d+\s*!=|==\s*\d+|!=\s*\d+', line):
                    return i + 1
    
    return None

def detect_typename(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'TypeName' convention error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Look for class/interface/enum declarations with lowercase first letter
        match = re.search(r'\b(class|interface|enum)\s+([a-z]\w*)', line)
        if match:
            return i + 1
    
    return None

def detect_membername(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'MemberName' convention error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Look for member variables with incorrect naming (uppercase first or containing underscores)
        if "private" in line or "protected" in line or "public" in line:
            # Skip methods
            if "(" in line:
                continue
                
            # Look for variable declarations
            match = re.search(r'\b(private|protected|public)\s+.*?\s+([A-Z]\w*|\w*_\w*)\s*[=;]', line)
            if match:
                var_name = match.group(2)
                # Skip constants (all uppercase)
                if not (var_name.isupper() and "final" in line):
                    return i + 1
    
    return None

def detect_methodname(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'MethodName' convention error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Look for method declarations with incorrect naming (uppercase first or containing underscores)
        if ("public" in line or "private" in line or "protected" in line) and "(" in line and ")" in line:
            # Skip constructors (name matches class name)
            if "class" in line:
                continue
                
            # Look for method declarations
            match = re.search(r'\b(public|private|protected)\s+\w+\s+([A-Z]\w*|\w*_\w*)\s*\(', line)
            if match:
                return i + 1
    
    return None

def detect_whitespace(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'Whitespace' convention error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    for i, line in enumerate(lines):
        if is_comment(line):
            continue
            
        # Look for missing whitespace around operators
        if re.search(r'[a-zA-Z0-9]=[a-zA-Z0-9]', line) or re.search(r'[a-zA-Z0-9]\+[a-zA-Z0-9]', line) or re.search(r'[a-zA-Z0-9]-[a-zA-Z0-9]', line):
            return i + 1
            
        # Look for incorrect whitespace in parentheses
        if "if(" in line or "for(" in line or "while(" in line:
            return i + 1
    
    return None

def detect_unused_import(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Detect 'Unused imports' error.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    imports = []
    
    # Extract imports
    for i, line in enumerate(lines):
        if line.strip().startswith("import "):
            # Extract the imported class or package
            match = re.search(r'import\s+(.*?);', line)
            if match:
                import_name = match.group(1)
                if import_name.endswith(".*"):
                    # For wildcard imports, just get the package name
                    import_package = import_name[:-2]
                    imports.append((import_package, i))
                else:
                    # For direct imports, get the class name
                    import_class = import_name.split(".")[-1]
                    imports.append((import_class, i))
    
    # Check for usages of each import
    code = "\n".join(lines)
    for import_name, line_number in imports:
        # Skip java.lang
        if import_name.startswith("java.lang"):
            return line_number + 1
            
        # For packages, we'll just check if the package name is used
        if import_name.count(".") > 0:
            continue
            
        # Count occurrences but skip the import line itself
        other_code = "\n".join(lines[:line_number] + lines[line_number+1:])
        if import_name not in other_code:
            return line_number + 1
    
    return None

def detect_generic_error(lines: List[str], error: Dict[str, Any]) -> Optional[int]:
    """
    Generic error detection for types we don't have specialized detection for.
    
    Args:
        lines: List of code lines
        error: Error dictionary
        
    Returns:
        Line number (1-based) or None if not found
    """
    error_name = error.get("name", "").lower()
    error_desc = error.get("description", "").lower()
    implementation_guide = error.get("implementation_guide", "").lower()
    
    # Special case for Array Index Out of Bounds
    if "array" in error_name and "bounds" in error_name:
        for i, line in enumerate(lines):
            if is_comment(line):
                continue
                
            if "[" in line and "]" in line:
                # Look for hardcoded array access with literal index
                if re.search(r'\[\s*\d+\s*\]', line):
                    return i + 1
        
        # Look for loop conditions that might cause bounds errors
        for i, line in enumerate(lines):
            if is_comment(line):
                continue
                
            if "for" in line and "=" in line and "<" in line:
                match = re.search(r'for\s*\(\s*\w+\s+\w+\s*=\s*\d+\s*;\s*\w+\s*(<|<=|>|>=)\s*(\w+)\.length', line)
                if match:
                    comparison = match.group(1)
                    if comparison == "<=" or comparison == ">=" or comparison == ">":
                        return i + 1
    
    # Find key terms in the code related to the error
    keywords = set()
    
    # Extract keywords from error name and description
    key_terms = []
    
    # From error name
    if error_name:
        key_terms.extend([term.strip() for term in re.findall(r'\b\w{4,}\b', error_name)])
    
    # From error description
    if error_desc:
        key_terms.extend([term.strip() for term in re.findall(r'\b\w{4,}\b', error_desc)])
    
    # From implementation guide
    if implementation_guide:
        key_terms.extend([term.strip() for term in re.findall(r'\b\w{4,}\b', implementation_guide)])
    
    # Filter common words
    filtered_terms = [term for term in key_terms if term.lower() not in COMMON_STOP_WORDS]
    
    # Look for any significant match in the code
    for term in filtered_terms:
        for i, line in enumerate(lines):
            if term.lower() in line.lower() and not is_comment(line):
                # Make sure this isn't just in a comment
                line_without_comments = re.sub(r'//.*$', '', line)
                if term.lower() in line_without_comments.lower():
                    return i + 1
    
    # Last resort - pick a likely line
    for i, line in enumerate(lines):
        if "if" in line or "for" in line or "while" in line:
            return i + 1
            
    if len(lines) > 10:
        return 10
    elif len(lines) > 0:
        return len(lines) // 2
    
    return None

def is_comment(line: str) -> bool:
    """Check if a line is a comment."""
    return line.strip().startswith("//") or line.strip().startswith("/*") or line.strip().startswith("*")

def is_primitive_or_common(word: str) -> bool:
    """Check if a word is a Java primitive or common class name."""
    return word in JAVA_KEYWORDS or word in COMMON_JAVA_TYPES

# Constants for Java language
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", 
    "class", "const", "continue", "default", "do", "double", "else", "enum", 
    "extends", "final", "finally", "float", "for", "goto", "if", "implements", 
    "import", "instanceof", "int", "interface", "long", "native", "new", "null", 
    "package", "private", "protected", "public", "return", "short", "static", 
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", 
    "transient", "try", "void", "volatile", "while", "true", "false"
}

COMMON_JAVA_TYPES = {
    "String", "Integer", "Boolean", "Character", "Byte", "Short", "Long", "Float", 
    "Double", "Object", "System", "Math", "List", "ArrayList", "Map", "HashMap",
    "Set", "HashSet", "TreeSet", "Collection", "Date", "Calendar", "Exception",
    "RuntimeException", "Error", "Thread", "Runnable", "File", "Scanner"
}

COMMON_STOP_WORDS = {
    "the", "and", "that", "this", "with", "for", "from", "not", "error", "should",
    "must", "might", "may", "could", "would", "code", "java", "class", "method", 
    "variable", "type", "statement", "function", "return", "value", "object", 
    "instance", "when", "where", "what", "which", "how", "why", "who", "null", 
    "public", "private", "protected", "static", "final", "void", "boolean", "int", 
    "double", "float", "char", "long", "byte", "short", "throws", "throw", "try", 
    "catch", "finally", "break", "continue", "else", "enum", "interface", "extends", 
    "implements", "import", "package", "super", "this", "new", "abstract", "assert", 
    "case", "default", "goto", "instanceof", "native", "strictfp", "switch", 
    "synchronized", "transient", "volatile", "true", "false"
}