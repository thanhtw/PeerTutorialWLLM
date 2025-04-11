"""
MCP Client module for Java Peer Review Training System.

This module implements the client side of the Model Control Protocol (MCP)
to interact with the tools provided by the MCP server.
"""

import logging
import requests
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPClient:
    """
    Client for interacting with the MCP server to access tools for code generation,
    evaluation, and feedback in the Java Peer Review Training System.
    """
    
    def __init__(self, host: str = None, port: int = None):
        """
        Initialize the MCP client.
        
        Args:
            host: Host address of the MCP server
            port: Port of the MCP server
        """
        load_dotenv()
        
        self.host = host or os.getenv("MCP_HOST", "localhost")
        self.port = port or int(os.getenv("MCP_PORT", "8000"))
        self.base_url = f"http://{self.host}:{self.port}/api"
        
        logger.info(f"Initialized MCP client connecting to {self.base_url}")
    
    def check_server_status(self) -> Dict[str, Any]:
        """
        Check if the MCP server is running.
        
        Returns:
            Status information
        """
        try:
            response = requests.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                return {"status": "online", "details": response.json()}
            return {"status": "error", "details": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"status": "offline", "error": str(e)}
    
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            params: Parameters for the tool
            
        Returns:
            Result of the tool call
        """
        try:
            url = f"{self.base_url}/{tool_name}"
            response = requests.post(url, json=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error calling tool {tool_name}: {response.status_code} - {response.text}")
                return {"error": f"Status code: {response.status_code}", "message": response.text}
        except Exception as e:
            logger.error(f"Exception calling tool {tool_name}: {str(e)}")
            return {"error": str(e)}
    
    def generate_java_code(self, code_length: str, difficulty_level: str, 
                          error_types: List[Dict[str, Any]], domain: str = "student_management") -> str:
        """
        Generate Java code with specific errors.
        
        Args:
            code_length: Length of code (short, medium, long)
            difficulty_level: Difficulty level (easy, medium, hard)
            error_types: List of error types to include
            domain: Domain context for the code
            
        Returns:
            Generated Java code with errors
        """
        params = {
            "code_length": code_length,
            "difficulty_level": difficulty_level,
            "error_types": error_types,
            "domain": domain
        }
        
        result = self.call_tool("generate_java_code", params)
        
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Error generating Java code: {result['error']}")
            return f"// Error generating code: {result.get('error', 'Unknown error')}"
        
        return result
    
    def evaluate_java_code(self, code: str, requested_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate Java code to check if it contains the requested errors.
        
        Args:
            code: Java code to evaluate
            requested_errors: List of errors that should be in the code
            
        Returns:
            Evaluation results
        """
        params = {
            "code": code,
            "requested_errors": requested_errors
        }
        
        return self.call_tool("evaluate_java_code", params)
    
    def analyze_student_review(self, code_snippet: str, known_problems: List[str], 
                             student_review: str) -> Dict[str, Any]:
        """
        Analyze a student's code review.
        
        Args:
            code_snippet: Java code that was reviewed
            known_problems: List of known problems in the code
            student_review: The student's review text
            
        Returns:
            Analysis results
        """
        params = {
            "code_snippet": code_snippet,
            "known_problems": known_problems,
            "student_review": student_review
        }
        
        return self.call_tool("analyze_student_review", params)
    
    def generate_targeted_guidance(self, code_snippet: str, known_problems: List[str],
                                 student_review: str, review_analysis: Dict[str, Any],
                                 iteration_count: int, max_iterations: int) -> str:
        """
        Generate targeted guidance for a student based on their review.
        
        Args:
            code_snippet: Java code that was reviewed
            known_problems: List of known problems in the code
            student_review: The student's review text
            review_analysis: Analysis of the student review
            iteration_count: Current iteration number
            max_iterations: Maximum number of iterations
            
        Returns:
            Targeted guidance text
        """
        params = {
            "code_snippet": code_snippet,
            "known_problems": known_problems,
            "student_review": student_review,
            "review_analysis": review_analysis,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations
        }
        
        return self.call_tool("generate_targeted_guidance", params)
    
    def generate_final_feedback(self, code_snippet: str, known_problems: List[str], 
                              review_history: List[Dict[str, Any]]) -> str:
        """
        Generate final feedback after all review iterations.
        
        Args:
            code_snippet: Java code that was reviewed
            known_problems: List of known problems in the code
            review_history: History of review attempts
            
        Returns:
            Final feedback text
        """
        params = {
            "code_snippet": code_snippet,
            "known_problems": known_problems,
            "review_history": review_history
        }
        
        return self.call_tool("generate_final_feedback", params)
    
    def generate_comparison_report(self, known_problems: List[str], review_analysis: Dict[str, Any]) -> str:
        """
        Generate a comparison report between known problems and student review.
        
        Args:
            known_problems: List of known problems in the code
            review_analysis: Analysis of the student's review
            
        Returns:
            Comparison report
        """
        params = {
            "known_problems": known_problems,
            "review_analysis": review_analysis
        }
        
        return self.call_tool("generate_comparison_report", params)
    
    def check_gpu_availability(self) -> Dict[str, Any]:
        """
        Check if GPU acceleration is available.
        
        Returns:
            Dictionary with GPU availability information
        """
        return self.call_tool("check_gpu_availability", {})
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models.
        
        Returns:
            List of available models with details
        """
        return self.call_tool("get_available_models", {})
    
    def download_model(self, model_name: str) -> Dict[str, Any]:
        """
        Download a model.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            Result of the download operation
        """
        params = {
            "model_name": model_name
        }
        
        return self.call_tool("download_model", params)