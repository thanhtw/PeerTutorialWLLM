"""
Code Generator module for Java Peer Review Training System.

This module provides the CodeGenerator class which dynamically generates
Java code snippets based on the selected difficulty level and code length,
eliminating the reliance on predefined templates.
"""

import random
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.language_models import BaseLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeGenerator:
    """
    Generates Java code snippets dynamically without relying on predefined templates.
    This class creates realistic Java code based on specified complexity and length.
    """
    def __init__(self, llm: BaseLanguageModel = None):
        """
        Initialize the CodeGenerator with an optional language model.
        
        Args:
            llm: Language model to use for code generation
        """
        self.llm = llm
        
        # Define complexity profiles for different code lengths
        self.complexity_profiles = {
            "short": {
                "class_count": 1,
                "method_count_range": (2, 4),
                "field_count_range": (2, 4),
                "imports_count_range": (0, 2),
                "nested_class_prob": 0.1,
                "interface_prob": 0.0
            },
            "medium": {
                "class_count": 1,
                "method_count_range": (3, 6),
                "field_count_range": (3, 6),
                "imports_count_range": (1, 4),
                "nested_class_prob": 0.3,
                "interface_prob": 0.2
            },
            "long": {
                "class_count": 2,
                "method_count_range": (5, 10),
                "field_count_range": (4, 8),
                "imports_count_range": (2, 6),
                "nested_class_prob": 0.5,
                "interface_prob": 0.4
            }
        }
        
        # Common Java domains to make code more realistic
        self.domains = [
            "user_management", "file_processing", "data_validation", 
            "calculation", "inventory_system", "notification_service",
            "logging", "banking", "e-commerce", "student_management"
        ]
    
    def generate_java_code(self, 
                           code_length: str = "medium", 
                           difficulty_level: str = "medium",
                           domain: str = None) -> str:
        """
        Generate Java code with specified length and complexity.
        
        Args:
            code_length: Desired code length (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            domain: Optional domain for the code context
            
        Returns:
            Generated Java code as a string
        """
        if self.llm:
            return self._generate_with_llm(code_length, difficulty_level, domain)
        else:
            return self._generate_programmatically(code_length, difficulty_level, domain)
    
    def _generate_with_llm(self, code_length: str, difficulty_level: str, domain: str = None) -> str:
        """
        Generate Java code using the language model.
        
        Args:
            code_length: Desired code length (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            domain: Optional domain for the code context
            
        Returns:
            Generated Java code as a string
        """
        if not self.llm:
            logger.warning("No LLM provided, falling back to programmatic generation")
            return self._generate_programmatically(code_length, difficulty_level, domain)
        
        # Select a domain if not provided
        if not domain:
            domain = random.choice(self.domains)
        
        # Create a detailed prompt for the LLM
        prompt = self._create_generation_prompt(code_length, difficulty_level, domain)
        
        try:
            # Generate the code using the LLM
            logger.info(f"Generating Java code with LLM: {code_length} length, {difficulty_level} difficulty, {domain} domain")
            response = self.llm.invoke(prompt)
            
            # Extract the Java code from the response
            code = self._extract_code_from_response(response)
            
            if not code or len(code.strip()) < 50:  # Minimal validation
                logger.warning("LLM generated invalid or too short code, falling back to programmatic generation")
                return self._generate_programmatically(code_length, difficulty_level, domain)
            
            return code
            
        except Exception as e:
            logger.error(f"Error generating code with LLM: {str(e)}")
            return self._generate_programmatically(code_length, difficulty_level, domain)
    
    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Java code from LLM response.
        
        Args:
            response: Full response from the LLM
            
        Returns:
            Extracted Java code
        """
        # Try to extract code blocks
        import re
        code_blocks = re.findall(r'```(?:java)?\s*(.*?)\s*```', response, re.DOTALL)
        
        if code_blocks:
            # Return the largest code block
            return max(code_blocks, key=len)
        
        return response  # If no code blocks found, return the full response
    
    def _create_generation_prompt(self, code_length: str, difficulty_level: str, domain: str) -> str:
        """
        Create a detailed prompt for the LLM to generate Java code.
        
        Args:
            code_length: Desired code length (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            domain: Domain for the code context
            
        Returns:
            Formatted prompt
        """
        complexity_profile = self.complexity_profiles.get(code_length, self.complexity_profiles["medium"])
        
        prompt = f"""
You are a Java programming expert. Create a realistic, working Java code snippet for a {domain} system.

The code should be {code_length} in length and {difficulty_level} in complexity.

Requirements:
- Create approximately {complexity_profile["class_count"]} main class(es)
- Include {complexity_profile["method_count_range"][0]}-{complexity_profile["method_count_range"][1]} methods
- Define {complexity_profile["field_count_range"][0]}-{complexity_profile["field_count_range"][1]} fields/properties
- Use {complexity_profile["imports_count_range"][0]}-{complexity_profile["imports_count_range"][1]} imports
- Include appropriate comments and documentation
- Follow standard Java naming conventions and best practices
- Make the code realistic and representative of real-world Java applications
- Do NOT include any intentional errors or problems

Return only the Java code with no additional explanations.
```java
// Your code here
```
"""
        return prompt
    
    def _generate_programmatically(self, code_length: str, difficulty_level: str, domain: str = None) -> str:
        """
        Generate Java code programmatically as a fallback method.
        
        Args:
            code_length: Desired code length (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            domain: Optional domain for the code context
            
        Returns:
            Generated Java code as a string
        """
        # Select domain if not provided
        if not domain:
            domain = random.choice(self.domains)
            
        # Get complexity profile
        profile = self.complexity_profiles.get(code_length, self.complexity_profiles["medium"])
        
        # Generate class name and imports
        class_name = self._generate_class_name(domain)
        imports = self._generate_imports(profile["imports_count_range"])
        
        # Generate fields
        fields_count = random.randint(*profile["field_count_range"])
        fields = self._generate_fields(fields_count, domain)
        
        # Generate methods
        methods_count = random.randint(*profile["method_count_range"])
        methods = self._generate_methods(methods_count, domain, difficulty_level)
        
        # Assemble the code
        code = "\n".join(imports) if imports else ""
        
        # Add class declaration and javadoc
        code += f"""
/**
 * {class_name} - A class for handling {domain.replace('_', ' ')} operations.
 */
public class {class_name} {{
    {self._format_code_block(fields)}
    
    /**
     * Default constructor for {class_name}.
     */
    public {class_name}() {{
        // Initialize components
    }}
    
    {self._format_code_block(methods)}
}}
"""
        return code
    
    def _generate_class_name(self, domain: str) -> str:
        """Generate a class name based on the domain."""
        domain_parts = domain.split('_')
        return ''.join(part.capitalize() for part in domain_parts) + random.choice(["Manager", "Service", "Handler", "Processor", "Controller"])
    
    def _generate_imports(self, count_range: Tuple[int, int]) -> List[str]:
        """Generate import statements."""
        common_imports = [
            "import java.util.List;",
            "import java.util.ArrayList;",
            "import java.util.Map;",
            "import java.util.HashMap;",
            "import java.util.Set;",
            "import java.util.HashSet;",
            "import java.io.File;",
            "import java.io.IOException;",
            "import java.time.LocalDate;",
            "import java.time.LocalDateTime;",
            "import java.util.stream.Collectors;",
            "import java.util.Optional;",
            "import java.util.function.Function;",
            "import java.util.concurrent.atomic.AtomicInteger;",
            "import java.util.concurrent.locks.Lock;",
            "import java.util.concurrent.locks.ReentrantLock;"
        ]
        
        count = random.randint(*count_range)
        return random.sample(common_imports, min(count, len(common_imports)))
    
    def _generate_fields(self, count: int, domain: str) -> List[str]:
        """Generate class fields."""
        field_types = ["int", "String", "boolean", "double", "List<String>", "Map<String, Integer>", "LocalDate"]
        field_prefixes = {
            "user_management": ["user", "role", "permission", "account"],
            "file_processing": ["file", "path", "content", "reader", "writer"],
            "data_validation": ["validator", "rule", "constraint", "pattern"],
            "calculation": ["value", "result", "sum", "factor", "multiplier"],
            "inventory_system": ["item", "product", "stock", "inventory"],
            "notification_service": ["message", "notification", "alert", "recipient"],
            "logging": ["logger", "level", "message", "format"],
            "banking": ["account", "balance", "transaction", "customer"],
            "e-commerce": ["product", "order", "cart", "customer"],
            "student_management": ["student", "course", "grade", "enrollment"]
        }
        
        prefixes = field_prefixes.get(domain, ["data", "item", "value", "count"])
        
        fields = []
        for _ in range(count):
            field_type = random.choice(field_types)
            prefix = random.choice(prefixes)
            suffix = random.choice(["Id", "Name", "Count", "Value", "Data", "Info", "Flag", "Status", "List", "Map"])
            field_name = prefix + suffix
            
            # Convert to camelCase
            field_name = field_name[0].lower() + field_name[1:]
            
            # Add modifiers
            modifiers = random.choice(["private ", "private final ", "private static "])
            
            fields.append(f"{modifiers}{field_type} {field_name};")
        
        return fields
    
    def _generate_methods(self, count: int, domain: str, difficulty: str) -> List[str]:
        """Generate class methods."""
        method_prefixes = {
            "user_management": ["get", "add", "update", "delete", "find", "validate"],
            "file_processing": ["read", "write", "parse", "save", "load"],
            "data_validation": ["validate", "check", "verify", "ensure"],
            "calculation": ["calculate", "compute", "add", "subtract", "multiply"],
            "inventory_system": ["add", "remove", "update", "check", "list"],
            "notification_service": ["send", "notify", "alert", "schedule"],
            "logging": ["log", "debug", "info", "warn", "error"],
            "banking": ["deposit", "withdraw", "transfer", "calculate"],
            "e-commerce": ["add", "remove", "checkout", "calculate", "find"],
            "student_management": ["enroll", "grade", "register", "calculate", "find"]
        }
        
        prefixes = method_prefixes.get(domain, ["process", "handle", "get", "set", "update"])
        
        methods = []
        for _ in range(count):
            prefix = random.choice(prefixes)
            suffix = random.choice(["Data", "Item", "Value", "Information", "Status", "Record", "Entry", "Result"])
            method_name = prefix + suffix
            
            # Convert to camelCase
            method_name = method_name[0].lower() + method_name[1:]
            
            # Generate method parameters and body based on difficulty
            if difficulty == "easy":
                params = random.choice(["", "int id", "String name", "boolean flag"])
                return_type = random.choice(["void", "boolean", "int", "String"])
                body = self._generate_method_body(return_type, "easy")
            elif difficulty == "medium":
                params = random.choice([
                    "int id, String name", 
                    "String data, boolean validate", 
                    "List<String> items",
                    "Map<String, Integer> values"
                ])
                return_type = random.choice(["boolean", "int", "String", "List<String>", "Map<String, Object>"])
                body = self._generate_method_body(return_type, "medium")
            else:  # hard
                params = random.choice([
                    "int id, String name, boolean validate", 
                    "String data, Map<String, Object> options, boolean strict",
                    "List<String> items, int startIndex, int maxResults",
                    "Map<String, Integer> values, Function<String, Integer> transformer"
                ])
                return_type = random.choice([
                    "boolean", "int", "String", "List<String>", 
                    "Map<String, Object>", "Optional<String>", "<T> List<T>"
                ])
                body = self._generate_method_body(return_type, "hard")
            
            # Add javadoc
    #         javadoc = f"""/**
    #  * {prefix.capitalize()}s the {suffix.lower()} based on the provided parameters.
    #  *
    #  * @param {params.replace(', ', '\n     * @param ')}
    #  * @return {return_type} {self._generate_return_description(return_type)}
    #  */"""
            
            method = f"""{javadoc}
    public {return_type} {method_name}({params}) {{
{body}
    }}"""
            
            methods.append(method)
        
        return methods
    
    def _generate_method_body(self, return_type: str, difficulty: str) -> str:
        """Generate method body based on return type and difficulty."""
        if difficulty == "easy":
            if return_type == "void":
                return "        // Simple implementation\n        System.out.println(\"Processing data...\");"
            elif return_type == "boolean":
                return "        // Simple boolean return\n        return true;"
            elif return_type == "int":
                return "        // Simple integer return\n        return 42;"
            elif return_type == "String":
                return "        // Simple string return\n        return \"Result\";"
        elif difficulty == "medium":
            if return_type == "void":
                return "        // Medium complexity implementation\n        try {\n            System.out.println(\"Processing data...\");\n            Thread.sleep(100);  // Simulate processing\n        } catch (InterruptedException e) {\n            Thread.currentThread().interrupt();\n        }"
            elif return_type == "boolean":
                return "        // Check condition before returning\n        if (System.currentTimeMillis() % 2 == 0) {\n            return true;\n        }\n        return false;"
            elif return_type == "int":
                return "        // Calculate integer result\n        int result = 0;\n        for (int i = 0; i < 10; i++) {\n            result += i;\n        }\n        return result;"
            elif return_type == "String":
                return "        // Build string result\n        StringBuilder result = new StringBuilder();\n        result.append(\"Result: \");\n        result.append(System.currentTimeMillis());\n        return result.toString();"
            elif "List" in return_type:
                return "        // Create and populate list\n        List<String> result = new ArrayList<>();\n        result.add(\"Item 1\");\n        result.add(\"Item 2\");\n        result.add(\"Item 3\");\n        return result;"
            elif "Map" in return_type:
                return "        // Create and populate map\n        Map<String, Object> result = new HashMap<>();\n        result.put(\"key1\", \"value1\");\n        result.put(\"key2\", 42);\n        result.put(\"key3\", true);\n        return result;"
        else:  # hard
            if return_type == "void":
                return "        // Complex implementation with error handling\n        try {\n            System.out.println(\"Processing data...\");\n            for (int i = 0; i < 5; i++) {\n                // Simulate complex processing\n                Thread.sleep(50);\n                System.out.println(\"Step \" + (i + 1) + \" completed\");\n            }\n        } catch (InterruptedException e) {\n            Thread.currentThread().interrupt();\n            System.err.println(\"Processing interrupted: \" + e.getMessage());\n        } catch (Exception e) {\n            System.err.println(\"Error during processing: \" + e.getMessage());\n        }"
            elif return_type == "boolean":
                return "        // Complex condition checking\n        try {\n            long timestamp = System.currentTimeMillis();\n            boolean condition1 = timestamp % 2 == 0;\n            boolean condition2 = timestamp % 3 == 0;\n            \n            if (condition1 && condition2) {\n                return true;\n            } else if (condition1 || condition2) {\n                return String.valueOf(timestamp).length() > 13;\n            }\n            return false;\n        } catch (Exception e) {\n            System.err.println(\"Error checking conditions: \" + e.getMessage());\n            return false;\n        }"
            elif return_type == "int":
                return "        // Complex integer calculation\n        try {\n            int result = 0;\n            int iterations = (int) (System.currentTimeMillis() % 10) + 5;\n            \n            for (int i = 0; i < iterations; i++) {\n                if (i % 3 == 0) {\n                    result += i * 2;\n                } else if (i % 3 == 1) {\n                    result += i;\n                } else {\n                    result += i / 2;\n                }\n            }\n            \n            return Math.max(result, 10);\n        } catch (Exception e) {\n            System.err.println(\"Calculation error: \" + e.getMessage());\n            return -1;\n        }"
            elif return_type == "String":
                return "        // Complex string building with formatting\n        try {\n            StringBuilder result = new StringBuilder();\n            long timestamp = System.currentTimeMillis();\n            \n            result.append(\"Result generated at: \")\n                  .append(java.time.LocalDateTime.now())\n                  .append(\"\\n\");\n                  \n            if (timestamp % 2 == 0) {\n                result.append(\"Even timestamp: \")\n                      .append(timestamp)\n                      .append(\"\\n\");\n            } else {\n                result.append(\"Odd timestamp: \")\n                      .append(timestamp)\n                      .append(\"\\n\");\n            }\n            \n            // Add some pseudorandom data\n            int checksum = 0;\n            for (char c : result.toString().toCharArray()) {\n                checksum += c;\n            }\n            \n            result.append(\"Checksum: \")\n                  .append(checksum);\n                  \n            return result.toString();\n        } catch (Exception e) {\n            return \"Error generating result: \" + e.getMessage();\n        }"
            elif "Optional" in return_type:
                return "        // Return optional value based on conditions\n        try {\n            long timestamp = System.currentTimeMillis();\n            \n            if (timestamp % 3 == 0) {\n                // One-third of the time return empty\n                return Optional.empty();\n            }\n            \n            String result = \"Result-\" + timestamp;\n            return Optional.of(result);\n        } catch (Exception e) {\n            System.err.println(\"Error generating optional: \" + e.getMessage());\n            return Optional.empty();\n        }"
            elif "List" in return_type:
                return "        // Generate complex list with filtering\n        try {\n            List<String> items = new ArrayList<>();\n            int count = (int) (System.currentTimeMillis() % 5) + 3;\n            \n            for (int i = 0; i < count; i++) {\n                items.add(\"Item-\" + (i + 1) + \"-\" + System.nanoTime() % 1000);\n            }\n            \n            // Filter items based on condition\n            return items.stream()\n                       .filter(item -> item.length() > 10)\n                       .collect(Collectors.toList());\n        } catch (Exception e) {\n            System.err.println(\"Error generating list: \" + e.getMessage());\n            return new ArrayList<>();\n        }"
            elif "Map" in return_type:
                return "        // Generate complex map with nested structure\n        try {\n            Map<String, Object> result = new HashMap<>();\n            long timestamp = System.currentTimeMillis();\n            \n            result.put(\"timestamp\", timestamp);\n            result.put(\"generated\", java.time.LocalDateTime.now().toString());\n            \n            // Add nested map\n            Map<String, Integer> stats = new HashMap<>();\n            stats.put(\"count\", (int) (timestamp % 100));\n            stats.put(\"status\", timestamp % 2 == 0 ? 1 : 0);\n            stats.put(\"code\", 200);\n            \n            result.put(\"statistics\", stats);\n            \n            // Add some conditional data\n            if (timestamp % 3 == 0) {\n                result.put(\"message\", \"Special case detected\");\n            }\n            \n            return result;\n        } catch (Exception e) {\n            Map<String, Object> error = new HashMap<>();\n            error.put(\"error\", e.getMessage());\n            return error;\n        }"
        
        # Default fallback
        return "        // Method implementation\n        return null;"
    
    def _generate_return_description(self, return_type: str) -> str:
        """Generate a description for the method return value."""
        if return_type == "void":
            return "This method does not return a value"
        elif return_type == "boolean":
            return "True if the operation was successful, false otherwise"
        elif return_type == "int":
            return "The calculated numeric result"
        elif return_type == "String":
            return "A string representation of the result"
        elif "List" in return_type:
            return "A list containing the requested elements"
        elif "Map" in return_type:
            return "A map containing the key-value pairs of the result"
        elif "Optional" in return_type:
            return "An optional containing the result, or empty if no result is available"
        else:
            return "The result of the operation"
    
    def _format_code_block(self, lines: List[str]) -> str:
        """Format a list of code lines into a proper indented block."""
        if not lines:
            return ""
        return "\n    ".join(lines)