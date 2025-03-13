"""
LLM Manager module for Java Peer Review Training System.

This module provides the LLMManager class for handling model initialization,
configuration, and management of Ollama models.
"""

import os
import requests
import time
import logging
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from dotenv import load_dotenv
from functools import lru_cache

# Update import to use the newer package
try:
    from langchain_community.llms.ollama import Ollama
except ImportError:
    # Fallback to old import if the new one is not available
    from langchain_community.llms import Ollama

from langchain_core.language_models import BaseLanguageModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMManager:
    """
    LLM Manager for handling model initialization, configuration and management.
    Provides caching and error recovery for Ollama models.
    """
    
    def __init__(self):
        """Initialize the LLM Manager with environment variables."""
        load_dotenv()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = os.getenv("DEFAULT_MODEL", "llama3:1b")
        
        # Track initialized models
        self.initialized_models = {}
        
        # Track model pull status
        self.pull_status = {}
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available Ollama models.
        
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        # Standard models that can be pulled
        library_models = [
            {"id": "llama3", "name": "Llama 3", "description": "Meta's Llama 3 model", "pulled": False},
            {"id": "llama3:8b", "name": "Llama 3 (8B)", "description": "Meta's Llama 3 8B model", "pulled": False},
            {"id": "llama3:1b", "name": "Llama 3 (1B)", "description": "Meta's Llama 3 1B model", "pulled": False},
            {"id": "phi3:mini", "name": "Phi-3 Mini", "description": "Microsoft Phi-3 model", "pulled": False},
            {"id": "gemma:2b", "name": "Gemma 2B", "description": "Google's lightweight Gemma model", "pulled": False},
            {"id": "mistral", "name": "Mistral 7B", "description": "Mistral AI's 7B model", "pulled": False},
            {"id": "codellama:7b", "name": "CodeLlama 7B", "description": "Meta's CodeLlama model for code generation", "pulled": False},
            {"id": "deepseek-coder:6.7b", "name": "DeepSeek Coder 6.7B", "description": "DeepSeek Coder model for programming tasks", "pulled": False}
        ]
        
        # Check Ollama API for available models
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                pulled_models = response.json().get("models", [])
                pulled_ids = [model["name"] for model in pulled_models]
                
                # Mark models as pulled if they exist locally
                for model in library_models:
                    if model["id"] in pulled_ids:
                        model["pulled"] = True
                
                # Add any pulled models that aren't in our standard list
                for pulled_model in pulled_models:
                    model_id = pulled_model["name"]
                    if not any(model["id"] == model_id for model in library_models):
                        library_models.append({
                            "id": model_id,
                            "name": model_id,
                            "description": f"Size: {pulled_model.get('size', 'Unknown')}",
                            "pulled": True
                        })
            
            return library_models
                
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            # Return list with local models marked as pulled
            return library_models
    
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model details
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/show?name={model_name}", timeout=5)
            
            if response.status_code == 200:
                model_info = response.json()
                
                # Format model info for display
                details = {
                    "name": model_info.get("model", model_name),
                    "size": self._format_size(model_info.get("size", 0)),
                    "modified": model_info.get("modified", "Unknown"),
                    "parameters": model_info.get("parameters", "Unknown"),
                    "template": model_info.get("template", "Unknown"),
                    "context_length": model_info.get("details", {}).get("context_length", "Unknown"),
                    "license": model_info.get("license", "Unknown"),
                    "modelfile": model_info.get("modelfile", "")
                }
                
                return details
            
            return {"name": model_name, "error": f"Status code: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error getting model details: {str(e)}")
            return {"name": model_name, "error": str(e)}
    
    def _format_size(self, size_in_bytes: int) -> str:
        """Format size in human-readable format."""
        if not isinstance(size_in_bytes, (int, float)):
            return "Unknown"
            
        size = float(size_in_bytes)
        
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
            
        return f"{size:.2f} TB"
            
    def download_ollama_model(self, model_name: str) -> bool:
        """
        Download a model using Ollama.
        
        Args:
            model_name (str): Name of the model to download
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.pull_status[model_name] = {
                "status": "pulling",
                "progress": 0,
                "error": None
            }
            
            # Start the pull operation
            response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to start model download: {response.text}")
                self.pull_status[model_name] = {
                    "status": "failed",
                    "progress": 0,
                    "error": f"Failed to start download: {response.text}"
                }
                return False
            
            logger.info(f"Started downloading {model_name}...")
            
            # Poll for completion
            model_ready = False
            start_time = time.time()
            max_wait_time = 600  # 10 minute timeout
            
            while not model_ready and (time.time() - start_time) < max_wait_time:
                try:
                    # Check if model exists in list of models
                    check_response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                    if check_response.status_code == 200:
                        models = check_response.json().get("models", [])
                        if any(model["name"] == model_name for model in models):
                            model_ready = True
                            self.pull_status[model_name] = {
                                "status": "completed",
                                "progress": 100,
                                "error": None
                            }
                            logger.info(f"Model {model_name} downloaded successfully!")
                            break
                    
                    # Update progress (simulated)
                    elapsed = time.time() - start_time
                    progress = min(95, int(elapsed / (max_wait_time * 0.8) * 100))
                    
                    self.pull_status[model_name] = {
                        "status": "pulling",
                        "progress": progress,
                        "error": None
                    }
                    
                    time.sleep(2)  # Check every 2 seconds
                except Exception as e:
                    # Log error but continue polling
                    logger.warning(f"Error checking model status: {str(e)}")
                    self.pull_status[model_name] = {
                        "status": "pulling",
                        "progress": self.pull_status[model_name].get("progress", 0),
                        "error": f"Error checking status: {str(e)}"
                    }
                    time.sleep(5)
            
            if not model_ready:
                logger.warning(f"Download timeout for {model_name}. It may still be downloading.")
                self.pull_status[model_name] = {
                    "status": "timeout",
                    "progress": self.pull_status[model_name].get("progress", 0),
                    "error": "Download timeout. The model may still be downloading."
                }
                return False
            
            return True
                
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            self.pull_status[model_name] = {
                "status": "failed",
                "progress": 0,
                "error": str(e)
            }
            return False
    
    def get_pull_status(self, model_name: str) -> Dict[str, Any]:
        """
        Get the current pull status of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with pull status information
        """
        if model_name not in self.pull_status:
            return {
                "status": "unknown",
                "progress": 0,
                "error": None
            }
            
        return self.pull_status[model_name]
    
    def check_ollama_connection(self) -> Tuple[bool, str]:
        """
        Check if Ollama service is running and accessible.
        
        Returns:
            Tuple[bool, str]: (is_connected, message)
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True, "Connected to Ollama successfully"
            else:
                return False, f"Connected to Ollama but received status code {response.status_code}"
        except requests.ConnectionError:
            return False, f"Failed to connect to Ollama at {self.ollama_base_url}"
        except Exception as e:
            return False, f"Error checking Ollama connection: {str(e)}"
    
    def check_model_availability(self, model_name: str) -> bool:
        """
        Check if a specific model is available in Ollama.
        
        Args:
            model_name (str): Name of the model to check
            
        Returns:
            bool: True if the model is available, False otherwise
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"] == model_name for model in models)
            return False
        except Exception:
            return False
    
    def initialize_model(self, model_name: str, model_params: Dict[str, Any] = None) -> Optional[BaseLanguageModel]:
        """
        Initialize an Ollama model with GPU support if available.
        
        Args:
            model_name (str): Name of the model to initialize
            model_params (Dict[str, Any], optional): Model parameters
            
        Returns:
            Optional[BaseLanguageModel]: Initialized LLM or None if initialization fails
        """
        # Create a unique key for caching based on model name and params
        cache_key = model_name
        
        if model_name in self.initialized_models:
            logger.info(f"Using cached model: {model_name}")
            return self.initialized_models[model_name]
                
        # Apply default model parameters if none provided
        if model_params is None:
            model_params = self._get_default_params(model_name)
        
        # Enable GPU acceleration if available
        model_params = self.enable_gpu_for_model(model_params)
        
        # Initialize Ollama model
        try:
            # Check if model is available
            if not self.check_model_availability(model_name):
                logger.warning(f"Model {model_name} not found. Attempting to pull...")
                if self.download_ollama_model(model_name):
                    logger.info(f"Successfully pulled model {model_name}")
                else:
                    logger.error(f"Failed to pull model {model_name}")
                    return None
            
            # Initialize Ollama model with parameters
            temperature = model_params.get("temperature", 0.7)
            
            # Extract additional parameters
            additional_params = {k: v for k, v in model_params.items() if k != "temperature"}
            
            # Remove potentially problematic parameters (max_tokens)
            if "max_tokens" in additional_params:
                logger.info(f"Removing max_tokens parameter to avoid Ollama validation error")
                del additional_params["max_tokens"]
            
            llm = Ollama(
                base_url=self.ollama_base_url,
                model=model_name,
                temperature=temperature,
                **additional_params
            )
            
            # Check if reasoning mode is enabled
            reasoning_mode = os.getenv("REASONING_MODE", "false").lower() == "true"
            
            # Test the model with a simple query
            try:
                if reasoning_mode:
                    # For reasoning mode, use a test prompt that encourages reasoning
                    test_prompt = "Let's think step by step: What is 2+2?"
                    _ = llm.invoke(test_prompt)
                else:
                    _ = llm.invoke("hello")
                    
                # If successful, cache the model
                self.initialized_models[model_name] = llm
                logger.info(f"Successfully initialized model {model_name} with GPU support")
                return llm
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {str(e)}")
                return None
                    
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {str(e)}")
            return None
    
    def initialize_model_from_env(self, model_key: str, temperature_key: str) -> Optional[BaseLanguageModel]:
        """
        Initialize a model using environment variables.
        
        Args:
            model_key (str): Environment variable key for model name
            temperature_key (str): Environment variable key for temperature
            
        Returns:
            Optional[BaseLanguageModel]: Initialized LLM or None if initialization fails
        """
        model_name = os.getenv(model_key, self.default_model)
        temperature = float(os.getenv(temperature_key, "0.7"))
        
        # Check if reasoning mode is enabled
        reasoning_mode = os.getenv("REASONING_MODE", "false").lower() == "true"
        
        model_params = {
            "temperature": temperature
        }
        
        # If reasoning mode is enabled, add specific parameters for reasoning
        if reasoning_mode:
            # Override temperature with reasoning temperature if specified
            reasoning_temp = os.getenv("REASONING_TEMPERATURE")
            if reasoning_temp:
                model_params["temperature"] = float(reasoning_temp)
            
            # Modify the model name to use a larger model if available
            if "1b" in model_name:  # If using 1B model, try to use 8B if available
                larger_model = model_name.replace("1b", "8b")
                if self.check_model_availability(larger_model):
                    model_name = larger_model
                    logger.info(f"Reasoning mode: Upgraded to {model_name}")
        
        return self.initialize_model(model_name, model_params)
    
    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Default parameters for the model
        """
        # Basic defaults - REMOVED max_tokens to avoid validation errors
        params = {
            "temperature": 0.7,
            # Removed max_tokens parameter - caused validation errors with some Ollama versions
        }
        
        # Add appropriate system message for specific model types
        if "code" in model_name.lower() or any(code_model in model_name.lower() for code_model in ["codellama", "deepseek-coder"]):
            # For code-oriented models
            params["temperature"] = 0.5  # Lower temperature for code generation
            
        # Adjust based on model name and role
        if "generative" in model_name or any(gen in model_name for gen in ["llama3", "llama-3"]):
            params["temperature"] = 0.8  # Slightly higher creativity for generative tasks
            
        elif "review" in model_name or any(rev in model_name for gen in ["mistral", "deepseek"]):
            params["temperature"] = 0.3  # Lower temperature for review tasks
            
        elif "summary" in model_name:
            params["temperature"] = 0.4  # Moderate temperature for summary tasks
            
        elif "compare" in model_name:
            params["temperature"] = 0.5  # Balanced temperature for comparison tasks
        
        return params
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from Ollama.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.delete(
                f"{self.ollama_base_url}/api/delete",
                json={"name": model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully deleted model: {model_name}")
                
                # Remove from initialized models if present
                if model_name in self.initialized_models:
                    del self.initialized_models[model_name]
                
                # Remove from pull status if present
                if model_name in self.pull_status:
                    del self.pull_status[model_name]
                
                return True
            else:
                logger.error(f"Failed to delete model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            return False
        
    def check_gpu_availability(self) -> Dict[str, Any]:
        """
        Check if GPU is available for Ollama.
        
        Returns:
            Dict with GPU availability information
        """
        try:
            # Check Ollama API for hardware information
            response = requests.get(f"{self.ollama_base_url}/api/hardware", timeout=5)
            
            if response.status_code == 200:
                hardware_info = response.json()
                gpu_info = hardware_info.get("gpu", {})
                
                # Check if GPU is available
                has_gpu = bool(gpu_info)
                
                # Format GPU information
                if has_gpu:
                    gpu_name = gpu_info.get("name", "Unknown")
                    gpu_memory = gpu_info.get("memory", {}).get("total", "Unknown")
                    
                    return {
                        "has_gpu": True,
                        "gpu_name": gpu_name,
                        "gpu_memory": gpu_memory,
                        "message": f"GPU detected: {gpu_name} with {gpu_memory} memory"
                    }
                else:
                    return {
                        "has_gpu": False,
                        "message": "No GPU detected for Ollama"
                    }
            else:
                return {
                    "has_gpu": False,
                    "message": f"Failed to get hardware info: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error checking GPU availability: {str(e)}")
            return {
                "has_gpu": False,
                "message": f"Error checking GPU: {str(e)}"
            }
        
    def enable_gpu_for_model(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enable GPU acceleration for model parameters.
        
        Args:
            model_params: Model parameters dictionary
            
        Returns:
            Updated model parameters with GPU acceleration
        """
        # Check GPU availability first
        gpu_info = self.check_gpu_availability()
        
        if gpu_info["has_gpu"]:
            # Add GPU parameters to model configuration
            updated_params = model_params.copy()
            
            # Set n_gpu_layers to -1 to use all available GPU layers
            updated_params["n_gpu_layers"] = -1
            
            # You can also adjust other GPU-related parameters based on the specific model
            # For example, for larger models with high memory requirements:
            # updated_params["f16_kv"] = True  # Use half-precision for key/value cache
            
            logger.info(f"Enabled GPU acceleration with {gpu_info['gpu_name']}")
            return updated_params
        else:
            logger.warning(f"GPU not available: {gpu_info['message']}")
            return model_params
           