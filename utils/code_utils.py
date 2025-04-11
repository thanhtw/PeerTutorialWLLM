"""
Utility functions for code generation and processing in the Java Code Review System.

This module provides shared functionality for generating prompts, 
extracting code from responses, and handling error comments.
"""

import re
import random
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_code_generation_prompt(
    code_length: str, 
    difficulty_level: str, 
    selected_errors: List[Dict[str, Any]] = None,
    domain: str = None,
    include_error_annotations: bool = True
) -> str:
    """
    Create a prompt for generating Java code with standardized error annotations.
    
    Args:
        code_length: Length of code (short, medium, long)
        difficulty_level: Difficulty level (easy, medium, hard)
        selected_errors: List of errors to include in the code
        domain: Domain context for the code
        include_error_annotations: Whether to include error annotations
        
    Returns:
        Formatted prompt string for LLM
    """
    # Select domain if not provided
    if not domain:
        domains = ["student_management", "file_processing", "data_validation", 
                  "calculation", "inventory_system", "notification_service",
                  "logging", "banking", "e-commerce"]
        domain = random.choice(domains)
    
    # Ensure string params
    code_length_str = str(code_length) if not isinstance(code_length, str) else code_length
    difficulty_level_str = str(difficulty_level) if not isinstance(difficulty_level, str) else difficulty_level
    
    # Define complexity profile
    complexity_profile = {
        "short": "1 class with 2-4 methods and fields",
        "medium": "1 class with 4-6 methods, may include nested classes",
        "long": "2-3 classes with 5-10 methods and proper class relationships"
    }.get(code_length_str.lower(), "1 class with 4-6 methods")    
    
    
    # Create error instructions with standardized format
    error_instructions = ""
    for i, error in enumerate(selected_errors, 1):
        error_type = error["type"]
        category = error.get("category", "")
        name = error["name"]
        description = error["description"]
        implementation_guide = error.get("implementation_guide", "Create this error in a way that matches its description")
        
        error_instructions += f"{i}. {error_type.upper()} ERROR - {name}\n"
        error_instructions += f"   Category: {category}\n"
        error_instructions += f"   Description: {description}\n"
        error_instructions += f"   Implementation: {implementation_guide}\n\n"

    # Define standardized error annotation format based on preference
    annotation_instruction = ""
    if include_error_annotations:
        annotation_instruction = """
3. For each error you include, add EXACTLY ONE error annotation directly above the line with the error, using this EXACT format:
   // ERROR: [TYPE] - [NAME] - [Brief explanation]
   
   For example:
   // ERROR: BUILD - NullPointerException - Object accessed without null check
   // ERROR: CHECKSTYLE - MethodName - Method name should be lowerCamelCase
   
   Do not use any other annotation formats like "TODO", "FIXME", etc. Use only the format specified above.
"""
    else:
        annotation_instruction = """
3. Do NOT add any comments or annotations indicating where the errors are.
   The errors should be integrated into the code without any explicit markers.
"""
    
    # Check if reasoning mode is enabled
    reasoning_mode = os.getenv("REASONING_MODE", "false").lower() == "true"
    
    # Common error examples to help the LLM understand
    error_examples = """
ERROR IMPLEMENTATION EXAMPLES:
- For "Cannot find symbol" error: Use a variable name like 'count' that was never declared (e.g., 'int result = count + 5;')
- For "Incompatible types" error: Assign a String to an int (e.g., 'int value = "hello";')
- For "MemberName" error: Use a variable name that violates conventions (e.g., 'private String User_Name;' instead of 'userName')
- For "TypeName" error: Define a class with lowercase first letter (e.g., 'class myClass' instead of 'MyClass')
- For "NullPointerException" error: Call a method on an object that is null (e.g., 'String s = null; int length = s.length();')

IMPORTANT: DO NOT just write correct code with error annotations. You MUST actually introduce the specified errors into the code itself.
"""
     
    prompt = f"""You are an expert Java programming educator who creates code review exercises with intentional errors.

Please create a {code_length} Java code example for a {domain} system with {complexity_profile}.
The code should be realistic, well-structured, and MUST CONTAIN the following specific errors:

{error_instructions}

{error_examples}

Let's think through this step by step:

1. First, I'll design the overall structure of the Java application for a {domain} system.
2. Then, I'll identify where each error should be placed to create a realistic learning scenario.
3. Next, I'll implement the code with these intentional errors in a way that maintains realism.
4. Finally, I'll review the code to ensure all required errors are present and the code is otherwise valid.

Requirements:
1. Write a complete, compilable Java code (except for the intentional errors)
2. Make the code realistic and representative of actual Java applications
3. ACTUALLY IMPLEMENT the errors in the code according to their implementation guides{annotation_instruction}
4. The difficulty level should be {difficulty_level}, appropriate for students learning Java
5. Return your final code in a code block with ``` delimiters

I'll now create the Java code with the required errors:"""
    
    
    return prompt

def extract_code_from_response(response: str) -> str:
    """
    Extract Java code from LLM response.
    
    Args:
        response: Text response from LLM
        
    Returns:
        Extracted Java code
    """
    # Check for None or empty response
    if not response:
        return ""
        
    # Try to extract code from code blocks
    import re
    
    # First try to match code blocks with proper Java syntax highlighting
    code_blocks = re.findall(r'```java\s*(.*?)\s*```', response, re.DOTALL)
    if code_blocks:
        # Return the largest code block
        return max(code_blocks, key=len)
    
    # If that fails, try to match any code blocks
    code_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_blocks:
        # Return the largest code block
        return max(code_blocks, key=len)
    
    # If no code blocks are found, assume the entire response is code
    return response.strip()

def add_error_comments(code: str, errors: List[Dict[str, Any]]) -> str:
    """
    Add fallback error comments to code.
    
    Args:
        code: Java code snippet
        errors: List of error dictionaries
        
    Returns:
        Code with added error comments
    """
    lines = code.split('\n')
    
    # Add comments at strategic positions
    for i, error in enumerate(errors):
        error_type = error.get("type", "unknown")
        name = error.get("name", "unknown error")
        description = error.get("description", "")
        
        # Find a reasonable position
        position = min(5 + i * 3, len(lines) - 1)
        
        # Create error comment
        comment = f"// TODO: Fix {error_type} error: {name} - {description}"
        
        # Insert comment
        lines.insert(position, comment)
    
    return '\n'.join(lines)

def get_error_count_for_difficulty(difficulty: str) -> int:
    """
    Get appropriate error count based on difficulty level.
    
    Args:
        difficulty: Difficulty level (easy, medium, hard)
        
    Returns:
        Number of errors to include
    """
    difficulty_map = {
        "easy": 2,
        "medium": 4,
        "hard": 6
    }
    return difficulty_map.get(str(difficulty).lower(), 4)

def format_list(items: List[str]) -> str:
    """
    Format a list of items as a bullet list.
    
    Args:
        items: List of string items
        
    Returns:
        Formatted bullet list as string
    """
    return "\n".join([f"- {item}" for item in items])

def add_line_numbers(code: str) -> str:
    """
    Add line numbers to code snippet.
    
    Args:
        code: The code snippet to add line numbers to
        
    Returns:
        Code with line numbers
    """
    lines = code.splitlines()
    max_line_num = len(lines)
    padding = len(str(max_line_num))
    
    # Create a list of lines with line numbers
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        # Format line number with consistent padding
        line_num = str(i).rjust(padding)
        numbered_lines.append(f"{line_num} | {line}")
    
    return "\n".join(numbered_lines)

def generate_comparison_report(known_problems: List[str], review_analysis: Dict[str, Any]) -> str:
    """
    Generate a comparison report between student review and known problems.
    
    Args:
        known_problems: List of known problems in the code
        review_analysis: Analysis of the student review
        
    Returns:
        Formatted comparison report
    """
    # Create report header
    report = "# Detailed Comparison: Your Review vs. Actual Issues\n\n"
    
    # New section explaining the comparison method
    report += "## How Reviews Are Compared\n\n"
    report += "Your review is compared to the known problems in the code using a semantic matching approach. "
    report += "This means we look for whether you've identified the key aspects of each issue, rather than requiring exact matching phrases.\n\n"
    
    report += "For each issue, the system checks if your comments include:\n"
    report += "1. **The correct location** of the error (line numbers, method names, etc.)\n"
    report += "2. **The appropriate error type or category** (e.g., NullPointerException, naming convention)\n"
    report += "3. **A clear explanation** of why it's problematic\n\n"
    
    report += "A problem is considered 'identified' if you correctly mentioned its key aspects. "
    report += "Partial credit may be given for partially identified issues. "
    report += "False positives are issues you reported that don't match any actual problems in the code.\n\n"
    
    # Problems section
    report += "## Code Issues Analysis\n\n"
    
    # Safely extract data from review analysis
    identified_problems = review_analysis.get("identified_problems", [])
    missed_problems = review_analysis.get("missed_problems", [])
    false_positives = review_analysis.get("false_positives", [])
    
    # Ensure all problems are properly converted to strings
    known_problems_str = [str(p) if not isinstance(p, str) else p for p in known_problems]
    identified_problems_str = [str(p) if not isinstance(p, str) else p for p in identified_problems]
    missed_problems_str = [str(p) if not isinstance(p, str) else p for p in missed_problems]
    false_positives_str = [str(p) if not isinstance(p, str) else p for p in false_positives]
    
    # Issues found correctly
    if identified_problems_str:
        report += "### Issues You Identified Correctly\n\n"
        for i, problem in enumerate(identified_problems_str, 1):
            report += f"**{i}. {problem}**\n\n"
            report += "Great job finding this issue! "
            report += "This demonstrates your understanding of this type of problem.\n\n"
    
    # Issues missed with detailed guidance
    if missed_problems_str:
        report += "### Issues You Missed\n\n"
        for i, problem in enumerate(missed_problems_str, 1):
            report += f"**{i}. {problem}**\n\n"
            
            # Enhanced guidance with example comment format
            problem_lower = problem.lower()
            report += "**How to identify this issue:**\n\n"
            
            if "null" in problem_lower or "nullpointer" in problem_lower:
                report += "When reviewing Java code, look for variables that might be null before being accessed. "
                report += "Check for null checks before method calls or field access. Missing null checks often lead to NullPointerExceptions at runtime.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [NullPointerException Risk] - The variable 'name' is accessed without a null check, which could cause a runtime exception`\n\n"
            elif "naming" in problem_lower or "convention" in problem_lower:
                report += "Check that class names use UpperCamelCase, while methods and variables use lowerCamelCase. "
                report += "Constants should use UPPER_SNAKE_CASE. Consistent naming improves code readability and maintainability.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Naming Convention] - The variable 'user_name' should use lowerCamelCase format (userName)`\n\n"
            elif "equal" in problem_lower or "==" in problem_lower:
                report += "String and object comparisons should use the .equals() method instead of the == operator, which only compares references. "
                report += "Using == for content comparison is a common error that can lead to unexpected behavior.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Object Comparison] - String comparison uses == operator instead of .equals() method`\n\n"
            elif "array" in problem_lower or "index" in problem_lower:
                report += "Always verify that array indices are within valid ranges before accessing elements. "
                report += "Check for potential ArrayIndexOutOfBoundsException risks, especially in loops.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Array Bounds] - Array access without bounds checking could cause ArrayIndexOutOfBoundsException`\n\n"
            elif "whitespace" in problem_lower or "indent" in problem_lower:
                report += "Look for consistent indentation and proper whitespace around operators and keywords. "
                report += "Proper formatting makes code more readable and maintainable.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Formatting] - Inconsistent indentation makes the code hard to read`\n\n"
            else:
                report += "When identifying issues, be specific about the location, type of error, and why it's problematic. "
                report += "Include line numbers and detailed explanations in your comments.\n\n"
                report += "**Example comment format:**\n\n"
                report += "`Line X: [Error Type] - Description of the issue and why it's problematic`\n\n"
    
    # False positives
    if false_positives_str:
        report += "### Issues You Incorrectly Identified\n\n"
        for i, problem in enumerate(false_positives_str, 1):
            report += f"**{i}. {problem}**\n\n"
            report += "This wasn't actually an issue in the code. "
            report += "Be careful not to flag correct code as problematic.\n\n"
    
    # Calculate some metrics
    total_problems = len(known_problems_str)
    identified_count = len(identified_problems_str)
    missed_count = len(missed_problems_str)
    false_positive_count = len(false_positives_str)
    
    accuracy = (identified_count / total_problems * 100) if total_problems > 0 else 0
    
    # Overall assessment
    report += "### Overall Assessment\n\n"
    
    if accuracy >= 80:
        report += "**Excellent review!** You found most of the issues in the code.\n\n"
    elif accuracy >= 60:
        report += "**Good review.** You found many issues, but missed some important ones.\n\n"
    elif accuracy >= 40:
        report += "**Fair review.** You found some issues, but missed many important ones.\n\n"
    else:
        report += "**Needs improvement.** You missed most of the issues in the code.\n\n"
    
    report += f"- You identified {identified_count} out of {total_problems} issues ({accuracy:.1f}%)\n"
    report += f"- You missed {missed_count} issues\n"
    report += f"- You incorrectly identified {false_positive_count} non-issues\n\n"
    
    # Add improvement tips
    report += "## Tips for Improvement\n\n"
    
    if missed_problems_str:
        # Categories of missed issues
        missed_categories = []
        
        for problem in missed_problems_str:
            problem_lower = problem.lower()
            if "null" in problem_lower:
                missed_categories.append("null pointer handling")
            elif "naming" in problem_lower or "convention" in problem_lower:
                missed_categories.append("naming conventions")
            elif "javadoc" in problem_lower or "comment" in problem_lower:
                missed_categories.append("documentation")
            elif "exception" in problem_lower or "throw" in problem_lower:
                missed_categories.append("exception handling")
            elif "loop" in problem_lower or "condition" in problem_lower:
                missed_categories.append("logical conditions")
            elif "whitespace" in problem_lower or "indentation" in problem_lower:
                missed_categories.append("code formatting")
            elif "array" in problem_lower or "index" in problem_lower:
                missed_categories.append("array handling")
        
        # Remove duplicates and sort
        missed_categories = sorted(set(missed_categories))
        
        if missed_categories:
            report += "Based on your review, focus on these areas in future code reviews:\n\n"
            for category in missed_categories:
                report += f"- **{category.title()}**\n"
            report += "\n"
    
    # Add systematic approach suggestion
    report += """### Systematic Review Approach

For more thorough code reviews, try this systematic approach:

1. **First pass**: Check for syntax errors, compilation issues, and obvious bugs
2. **Second pass**: Examine naming conventions, code style, and documentation
3. **Third pass**: Analyze logical flow, edge cases, and potential runtime errors
4. **Final pass**: Look for performance issues, security concerns, and maintainability problems

By following a structured approach, you'll catch more issues and provide more comprehensive reviews.
"""
    
    # Add effective comment format
    report += """
### Effective Comment Format

When writing code review comments, use this format for clarity and consistency:

```
Line X: [Error Type] - Description of the issue and why it's problematic
```

For example:
```
Line 42: [NullPointerException Risk] - The 'user' variable could be null here, add a null check before calling methods
```

This format helps others quickly understand the location, type, and impact of each issue.
"""
    
    return report

def strip_error_annotations(code: str) -> str:
    """
    Remove error annotation comments from code while preserving string literals.
    Focused on standardized format: // ERROR: [TYPE] - [NAME] - [Description]
    but still handles legacy formats for backward compatibility.
    Preserves comments that start with "// Intentional error:" as these indicate
    deliberate errors for educational purposes.
    
    Args:
        code: The code with error annotations
        
    Returns:
        Clean code without error annotations (except for "// Intentional error:" comments)
    """
    import re
    
    # Skip processing if code is empty
    if not code:
        return code
    
    # Standard format pattern (should match most generated annotations)
    standard_pattern = r'^\s*//\s*ERROR:\s*\[.*?\].*\n'
    
    # Match and remove specific error annotation patterns
    error_patterns = [
        # Primary standardized format
        r'^\s*//\s*ERROR:\s*.*-.*\n',
        
        # Legacy formats for backward compatibility
        r'^\s*//\s*ERROR TYPE:.*\n',
        r'^\s*//\s*ERROR NAME:.*\n',
        r'^\s*//\s*TODO: Fix.*\n',
        r'^\s*//\s*FIXME:.*\n',
        
        # Very specific patterns that indicate error annotations
        r'^\s*//.*BUILD ERROR.*\n',
        r'^\s*//.*CHECKSTYLE ERROR.*\n',
        r'^\s*//.*LOGICAL ERROR.*\n',
        r'^\s*//.*RUNTIME ERROR.*\n',
        
        # General case for problem area annotations
        r'^\s*//\s*Problem area:.*\n',
    ]
    
    # Process the code line by line to preserve "// Intentional error:" comments
    lines = code.splitlines()
    result_lines = []
    
    for line in lines:
        # Skip lines that match our error patterns unless they start with "// Intentional error:"
        if re.match(r'^\s*//\s*Intentional error:', line):
            # Keep this line - it's an intentional error marker we want to preserve
            result_lines.append(line)
            continue
            
        # Check if this line matches any of our error annotation patterns
        is_error_annotation = False
        
        # Check against standard pattern
        if re.match(standard_pattern, line):
            is_error_annotation = True
        else:
            # Check against other patterns
            for pattern in error_patterns:
                if re.match(pattern, line):
                    is_error_annotation = True
                    break
        
        # Keep the line if it's not an error annotation
        if not is_error_annotation:
            result_lines.append(line)
    
    # Join the lines back together
    result = '\n'.join(result_lines)
    
    # Clean up any empty lines created by removing annotations
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    
    return result