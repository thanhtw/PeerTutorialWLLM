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
    print("\n========== EXTRACTING ERROR LOCATIONS ==========")
    print(f"Code length: {len(code)} characters")
    print(f"Number of errors to locate: {len(errors)}")
    
    enhanced_errors = []
    lines = code.splitlines()
    print(f"Code has {len(lines)} lines")
    
    for error in errors:
        error_info = error.copy()
        
        # Default values if we can't find location information
        error_info["line_number"] = None
        error_info["line_content"] = None
        error_info["context"] = None
        
        # Try to find potential locations for this error
        error_type = error.get("type", "").lower()
        error_name = error.get("name", "").lower()
        description = error.get("description", "").lower()
        category = error.get("category", "").lower() if "category" in error else ""
        
        print(f"\nLooking for: {error_type} - {error_name}")
        
        # Look for both explicit error markers and code patterns
        found_locations = []
        
        # 1. Look for explicit error comment markers in the code
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check for explicit annotations like "// ERROR TYPE: ERROR NAME" or "// TODO: Fix"
            if ((("error" in line_lower or "todo" in line_lower) and error_name.lower() in line_lower) or
                (error_type in line_lower and error_name.lower() in line_lower)):
                # Found an annotation - check the next few lines for the actual error
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 5)
                found_locations.append({
                    "line_number": i + 1,  # 1-based line numbering
                    "line_content": line.strip(),
                    "context": "\n".join(lines[context_start:context_end]),
                    "confidence": 0.9  # High confidence for explicit annotations
                })
                print(f"  Found explicit mention at line {i+1}: {line.strip()[:50]}...")
        
        # Look for implementation guide hints in the code
        implementation_guide = error.get("implementation_guide", "").lower()
        if implementation_guide and len(implementation_guide) > 10:
            # Extract key phrases from implementation guide (words of 5+ chars)
            key_phrases = [word for word in re.findall(r'\b\w{5,}\b', implementation_guide) 
                          if word not in ['implementation', 'should', 'could', 'would', 'instead']]
            
            if key_phrases:
                print(f"  Looking for implementation phrases: {', '.join(key_phrases[:3])}...")
                
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if any(phrase in line_lower for phrase in key_phrases):
                        context_start = max(0, i - 2)
                        context_end = min(len(lines), i + 3)
                        found_locations.append({
                            "line_number": i + 1,
                            "line_content": line.strip(),
                            "context": "\n".join(lines[context_start:context_end]),
                            "confidence": 0.75  # Good confidence for implementation matches
                        })
                        print(f"  Found implementation match at line {i+1}: {line.strip()[:50]}...")
        
        # 2. Try to identify error patterns in code based on error types/names
        if error_type == "build":
            print(f"  Checking for BUILD error patterns for: {error_name}")
            
            if "nullpointer" in error_name or "null pointer" in error_name:
                # Look for potential null pointer issues (object access without null check)
                for i, line in enumerate(lines):
                    if "." in line and "null" not in line.lower() and "check" not in line.lower() and "if" not in line.lower():
                        if re.search(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*', line):
                            # Found object access pattern
                            context_start = max(0, i - 2)
                            context_end = min(len(lines), i + 3)
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": "\n".join(lines[context_start:context_end]),
                                "confidence": 0.6  # Medium confidence
                            })
                            print(f"  Found potential NPE at line {i+1}: {line.strip()[:50]}...")
            
            elif "missing return" in error_name.lower():
                # Look for methods that might be missing returns
                for i, line in enumerate(lines):
                    if "void" not in line.lower() and any(type_name in line for type_name in 
                                                         ["int", "String", "boolean", "Object", "List"]) and \
                       "(" in line and ")" in line and "{" in line:
                        # Check if method has returns in all branches
                        context_start = max(0, i - 1)
                        # Find the method end
                        brace_count = 0
                        found_return = False
                        method_body = []
                        
                        for j, method_line in enumerate(lines[i:], i):
                            method_body.append(method_line)
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
                        
                        if not found_return and len(method_body) > 2:
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": "\n".join(lines[context_start:min(len(lines), context_end)]),
                                "confidence": 0.7
                            })
                            print(f"  Found missing return at line {i+1}: {line.strip()[:50]}...")
                            
            elif "incompatible types" in error_name.lower() or "type mismatch" in error_name.lower():
                # Look for assignments with potential type mismatches
                for i, line in enumerate(lines):
                    if "=" in line and "==" not in line and "!=" not in line:
                        # Check for assignment between different types
                        left_side = line.split("=")[0].strip()
                        right_side = line.split("=")[1].strip().rstrip(';')
                        
                        # Check for potential type mismatches
                        if ("int" in left_side and ("\"" in right_side or "String" in right_side)) or \
                           ("String" in left_side and not ("\"" in right_side or "String" in right_side)):
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": line.strip(),
                                "confidence": 0.65
                            })
                            print(f"  Found incompatible types at line {i+1}: {line.strip()[:50]}...")
                        
            elif "array" in error_name.lower() and "bounds" in error_name.lower():
                # Look for array access without bounds check
                for i, line in enumerate(lines):
                    if "[" in line and "]" in line and "length" not in line.lower() and "if" not in line.lower():
                        # Potential array without bounds check
                        context_start = max(0, i - 2)
                        context_end = min(len(lines), i + 3)
                        found_locations.append({
                            "line_number": i + 1,
                            "line_content": line.strip(),
                            "context": "\n".join(lines[context_start:context_end]),
                            "confidence": 0.7
                        })
                        print(f"  Found array bounds at line {i+1}: {line.strip()[:50]}...")
        
        elif error_type == "checkstyle":
            print(f"  Checking for CHECKSTYLE error patterns for: {error_name}")
            
            if "naming" in error_name.lower() or "convention" in error_name.lower():
                # Look for naming convention issues
                if "className" in error_name.lower() or "TypeName" in error_name or "class" in category:
                    # Class naming conventions (should be UpperCamelCase)
                    for i, line in enumerate(lines):
                        if re.search(r'\bclass\s+[a-z][a-zA-Z0-9_]*', line) or \
                           re.search(r'\binterface\s+[a-z][a-zA-Z0-9_]*', line) or \
                           re.search(r'\benum\s+[a-z][a-zA-Z0-9_]*', line):
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": line.strip(),
                                "confidence": 0.8
                            })
                            print(f"  Found class naming issue at line {i+1}: {line.strip()[:50]}...")
                            
                elif "methodName" in error_name.lower() or "MethodName" in error_name:
                    # Method naming conventions (should be lowerCamelCase)
                    for i, line in enumerate(lines):
                        if re.search(r'\b(public|private|protected)?\s+\w+\s+[A-Z][a-zA-Z0-9_]*\s*\(', line):
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": line.strip(),
                                "confidence": 0.75
                            })
                            print(f"  Found method naming issue at line {i+1}: {line.strip()[:50]}...")
                            
                elif "variableName" in error_name.lower() or "MemberName" in error_name or "LocalVariableName" in error_name:
                    # Variable naming conventions
                    for i, line in enumerate(lines):
                        # Check for variable declarations with bad names (underscores or starting with uppercase)
                        if re.search(r'\b(int|String|boolean|double|float|long|char)\s+[A-Z][a-zA-Z0-9_]*', line) or \
                           re.search(r'\b(int|String|boolean|double|float|long|char)\s+\w*_\w*', line):
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": line.strip(),
                                "confidence": 0.7
                            })
                            print(f"  Found variable naming issue at line {i+1}: {line.strip()[:50]}...")
                            
                elif "constant" in error_name.lower() or "ConstantName" in error_name:
                    # Constants should be UPPER_CASE
                    for i, line in enumerate(lines):
                        if re.search(r'\b(static\s+final|final\s+static)\s+\w+\s+[a-z][a-zA-Z0-9_]*', line):
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": line.strip(),
                                "confidence": 0.8
                            })
                            print(f"  Found constant naming issue at line {i+1}: {line.strip()[:50]}...")
            
            elif "whitespace" in error_name.lower() or "indentation" in error_name.lower():
                # Check for whitespace, formatting issues
                for i, line in enumerate(lines):
                    # Look for missing spaces around operators
                    if re.search(r'[a-zA-Z0-9_]=[a-zA-Z0-9_]', line) or \
                       re.search(r'[a-zA-Z0-9_]\+[a-zA-Z0-9_]', line) or \
                       re.search(r'[a-zA-Z0-9_]-[a-zA-Z0-9_]', line):
                        found_locations.append({
                            "line_number": i + 1,
                            "line_content": line.strip(),
                            "context": line.strip(),
                            "confidence": 0.7
                        })
                        print(f"  Found whitespace issue at line {i+1}: {line.strip()[:50]}...")
                        
            elif "brace" in error_name.lower() or "leftcurly" in error_name.lower() or "rightcurly" in error_name.lower():
                # Check for brace placement issues
                for i, line in enumerate(lines):
                    if line.strip().endswith(")") and i+1 < len(lines) and lines[i+1].strip().startswith("{"):
                        # Left brace on new line when it should be at end of previous line
                        found_locations.append({
                            "line_number": i + 2,  # Report the line with the brace
                            "line_content": lines[i+1].strip(),
                            "context": f"{line.strip()}\n{lines[i+1].strip()}",
                            "confidence": 0.75
                        })
                        print(f"  Found brace issue at line {i+2}: {lines[i+1].strip()[:50]}...")
                        
            elif "javadoc" in error_name.lower() or "comment" in error_name.lower():
                # Check for missing or invalid Javadoc comments
                for i, line in enumerate(lines):
                    if line.strip().startswith("public") and not line.strip().startswith("public class"):
                        # Check if previous lines have Javadoc
                        if i == 0 or not any("/**" in lines[j] for j in range(max(0, i-5), i)):
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": "\n".join(lines[max(0, i-1):min(len(lines), i+2)]),
                                "confidence": 0.7
                            })
                            print(f"  Found missing Javadoc at line {i+1}: {line.strip()[:50]}...")
        
        # If we didn't find any locations based on the patterns above, 
        # try a more generic text-matching approach
        if not found_locations:
            print("  No specific pattern matches found, trying generic text matching")
            # Try to match based on error name keywords in the code
            keywords = re.findall(r'\b\w{4,}\b', error_name.lower())
            if keywords:
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    matches = [keyword for keyword in keywords if keyword in line_lower]
                    if matches:
                        context_start = max(0, i - 2)
                        context_end = min(len(lines), i + 3)
                        found_locations.append({
                            "line_number": i + 1,
                            "line_content": line.strip(),
                            "context": "\n".join(lines[context_start:context_end]),
                            "confidence": 0.5 * (len(matches) / len(keywords))  # Scale confidence by match ratio
                        })
                        print(f"  Found keyword match at line {i+1}: {line.strip()[:50]}...")
            
            # If still no locations, search in broader context based on description
            if not found_locations and description:
                desc_keywords = re.findall(r'\b\w{5,}\b', description.lower())
                if desc_keywords:
                    for i, line in enumerate(lines):
                        line_lower = line.lower()
                        matches = [keyword for keyword in desc_keywords if keyword in line_lower]
                        if matches:
                            context_start = max(0, i - 2)
                            context_end = min(len(lines), i + 3)
                            found_locations.append({
                                "line_number": i + 1,
                                "line_content": line.strip(),
                                "context": "\n".join(lines[context_start:context_end]),
                                "confidence": 0.4 * (len(matches) / len(desc_keywords))  # Lower confidence
                            })
                            print(f"  Found description match at line {i+1}: {line.strip()[:50]}...")
        
        # If we still haven't found a location, place it at a reasonable default position
        if not found_locations:
            print("  No location found, using a default position")
            # Decide on a reasonable default position based on the error type and category
            reasonable_line = 1  # Default to first line
            
            if error_type == "build":
                if "method" in error_name.lower() or "return" in error_name.lower():
                    # Find a method declaration
                    for i, line in enumerate(lines):
                        if "public" in line.lower() and "(" in line and ")" in line:
                            reasonable_line = i + 1
                            break
                elif "variable" in error_name.lower() or "field" in error_name.lower():
                    # Find a variable or field declaration
                    for i, line in enumerate(lines):
                        if any(type_name in line.lower() for type_name in 
                                ["int", "string", "boolean", "double", "object"]) and "=" in line:
                            reasonable_line = i + 1
                            break
            elif error_type == "checkstyle":
                if "class" in error_name.lower():
                    # Find the class declaration
                    for i, line in enumerate(lines):
                        if "class" in line.lower():
                            reasonable_line = i + 1
                            break
                elif "javadoc" in error_name.lower() or "comment" in error_name.lower():
                    # Find a method without proper comments
                    for i, line in enumerate(lines):
                        if "public" in line.lower() and "(" in line and ")" in line:
                            reasonable_line = i + 1
                            break
            
            # Default to a position about 1/3 of the way into the file
            if reasonable_line == 1 and len(lines) > 10:
                reasonable_line = len(lines) // 3
            
            # Create a default location
            context_start = max(0, reasonable_line - 3)
            context_end = min(len(lines), reasonable_line + 3)
            found_locations.append({
                "line_number": reasonable_line,
                "line_content": lines[reasonable_line - 1] if reasonable_line <= len(lines) else "// Default position",
                "context": "\n".join(lines[context_start:context_end]),
                "confidence": 0.3  # Low confidence for default positions
            })
        
        # Select the best location (highest confidence)
        best_location = max(found_locations, key=lambda loc: loc.get("confidence", 0))
        error_info["line_number"] = best_location["line_number"]
        error_info["line_content"] = best_location["line_content"]
        error_info["context"] = best_location["context"]
        
        print(f"  Selected location - Line {error_info['line_number']}: {error_info['line_content'][:50]}...")
        enhanced_errors.append(error_info)
    
    print(f"\nCompleted error location extraction - Found locations for {len(enhanced_errors)} errors")
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
    # Print debugging info for error enrichment process
    print("\n========== STARTING ERROR ENRICHMENT ==========")
    print(f"Number of selected errors: {len(selected_errors)}")
    
    # Make sure we're working with a non-empty list of errors
    if not selected_errors or len(selected_errors) == 0:
        print("WARNING: No selected errors provided to enrich_error_information")
        return [], []
    
    # Extract and add location information to errors
    try:
        enhanced_errors = extract_error_locations(code, selected_errors)
        
        # Print the enhanced errors for debugging
        print("\nEnhanced errors after extraction:")
        for i, error in enumerate(enhanced_errors, 1):
            line_num = error.get('line_number', 'Not found')
            name = error.get('name', 'Unknown')
            print(f"  {i}. {name} - Line: {line_num}")
            
        # Generate detailed problem descriptions
        problem_descriptions = generate_problem_descriptions(enhanced_errors)
        
        # Print the problem descriptions for debugging
        print("\nGenerated problem descriptions:")
        for i, desc in enumerate(problem_descriptions, 1):
            print(f"  {i}. {desc[:100]}..." if len(desc) > 100 else f"  {i}. {desc}")
            
        # If no line numbers were found, ensure we still have basic descriptions
        if all(error.get('line_number') is None for error in enhanced_errors):
            print("\nWARNING: No line numbers found for any errors, using fallback descriptions")
            # Create fallback descriptions without line numbers
            problem_descriptions = []
            for error in selected_errors:
                error_type = error.get("type", "unknown").upper()
                name = error.get("name", "unknown error")
                description = error.get("description", "")
                category = error.get("category", "unknown")
                problem_descriptions.append(f"{error_type} ERROR - {name}: {description} (Category: {category})")
                
            # Add at least basic line information to enhanced errors
            for i, error in enumerate(enhanced_errors):
                if error.get('line_number') is None:
                    # Assign an arbitrary line based on the error index
                    # This ensures we at least have some location information
                    line_number = 10 + (i * 5)  # arbitrary spacing
                    error['line_number'] = line_number
                    error['line_content'] = f"// Location estimated for {error.get('name', 'unknown error')}"
                    
                    # Try to find a more reasonable line based on context
                    lines = code.splitlines()
                    for j, line in enumerate(lines):
                        if error.get('name', '').lower() in line.lower():
                            error['line_number'] = j + 1
                            error['line_content'] = line.strip()
                            break
            
        print("\n========== ERROR ENRICHMENT COMPLETE ==========")
        return enhanced_errors, problem_descriptions
        
    except Exception as e:
        print(f"ERROR during error enrichment: {str(e)}")
        # Provide a fallback if the enrichment process fails
        fallback_enhanced = []
        fallback_descriptions = []
        
        for i, error in enumerate(selected_errors):
            error_type = error.get("type", "unknown").upper()
            name = error.get("name", "unknown error")
            description = error.get("description", "")
            category = error.get("category", "unknown")
            
            # Create a basic enhanced error
            enhanced = error.copy()
            enhanced["line_number"] = 10 + (i * 5)  # arbitrary spacing
            enhanced["line_content"] = f"// Location not found for {name}"
            enhanced["context"] = "Context not available due to error detection failure"
            
            fallback_enhanced.append(enhanced)
            fallback_descriptions.append(f"{error_type} ERROR - {name}: {description} (Category: {category})")
        
        print("Using fallback error enrichment due to exception")
        return fallback_enhanced, fallback_descriptions