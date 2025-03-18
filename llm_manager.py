"""
LLM Manager module for Java Peer Review Training System.

This module provides the LLMManager class for handling model initialization,
configuration, and management of Ollama models with enhanced GPU support.
"""

import os
import requests
import time
import logging
import json
import psutil
import subprocess
from typing import Dict, Any, Optional, List, Union, Tuple
from dotenv import load_dotenv
import inspect
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
    Provides caching and error recovery for Ollama models with enhanced GPU support.
    """
    
    def __init__(self):
        """Initialize the LLM Manager with environment variables."""
        load_dotenv()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = os.getenv("DEFAULT_MODEL", "llama3:1b")
        
        # Force GPU usage by default if available
        self.force_gpu = os.getenv("ENABLE_GPU", "true").lower() == "true"
        self.gpu_layers = int(os.getenv("GPU_LAYERS", "-1"))  # -1 means use all available layers
        
        # Track initialized models
        self.initialized_models = {}
        
        # Track model pull status
        self.pull_status = {}
        
        # Cache GPU info
        self._gpu_info = None
        
        # Initialize GPU data
        self.refresh_gpu_info()
    
    def refresh_gpu_info(self):
        """Refresh GPU information with extended details."""
        self._gpu_info = self.check_gpu_availability(extended=True)
        return self._gpu_info
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available Ollama models.
        
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        # Standard models that can be pulled
        library_models = [
            {"id": "llama3", "name": "Llama 3", "description": "Meta's Llama 3 model", "pulled": False, "gpu_optimized": True},
            {"id": "llama3:8b", "name": "Llama 3 (8B)", "description": "Meta's Llama 3 8B model", "pulled": False, "gpu_optimized": True},
            {"id": "llama3:1b", "name": "Llama 3 (1B)", "description": "Meta's Llama 3 1B model", "pulled": False, "gpu_optimized": True},
            {"id": "phi3:mini", "name": "Phi-3 Mini", "description": "Microsoft Phi-3 model", "pulled": False, "gpu_optimized": True},
            {"id": "gemma:2b", "name": "Gemma 2B", "description": "Google's lightweight Gemma model", "pulled": False, "gpu_optimized": True},
            {"id": "mistral", "name": "Mistral 7B", "description": "Mistral AI's 7B model", "pulled": False, "gpu_optimized": True},
            {"id": "codellama:7b", "name": "CodeLlama 7B", "description": "Meta's CodeLlama model for code generation", "pulled": False, "gpu_optimized": True},
            {"id": "deepseek-coder:6.7b", "name": "DeepSeek Coder 6.7B", "description": "DeepSeek Coder model for programming tasks", "pulled": False, "gpu_optimized": True}
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
                        
                        # Check if model has GPU-specific parameters
                        try:
                            model_info = self.get_model_details(model["id"])
                            modelfile = model_info.get("modelfile", "").lower()
                            model["gpu_optimized"] = "gpu" in modelfile or model["gpu_optimized"]
                        except:
                            pass
                
                # Add any pulled models that aren't in our standard list
                for pulled_model in pulled_models:
                    model_id = pulled_model["name"]
                    if not any(model["id"] == model_id for model in library_models):
                        # Get model details to check GPU optimization
                        gpu_optimized = False
                        try:
                            model_info = self.get_model_details(model_id)
                            modelfile = model_info.get("modelfile", "").lower()
                            gpu_optimized = "gpu" in modelfile or "cuda" in modelfile
                        except:
                            pass
                            
                        library_models.append({
                            "id": model_id,
                            "name": model_id,
                            "description": f"Size: {pulled_model.get('size', 'Unknown')}",
                            "pulled": True,
                            "gpu_optimized": gpu_optimized
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
                    "modelfile": model_info.get("modelfile", ""),
                    "gpu_optimized": "gpu" in model_info.get("modelfile", "").lower() or "cuda" in model_info.get("modelfile", "").lower()
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
    
    def get_gpu_memory_usage(self) -> Dict[str, Any]:
        """
        Get current GPU memory usage if available.
        
        Returns:
            Dictionary with GPU memory usage information
        """
        try:
            # Try to get GPU info using nvidia-smi if NVIDIA GPU
            gpu_info = {}
            
            try:
                # Try nvidia-smi for NVIDIA GPUs
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                       capture_output=True, text=True, check=True)
                
                if result.returncode == 0:
                    gpu_data = result.stdout.strip().split(',')
                    if len(gpu_data) >= 3:
                        mem_used = float(gpu_data[0].strip())
                        mem_total = float(gpu_data[1].strip())
                        gpu_util = float(gpu_data[2].strip())
                        
                        gpu_info = {
                            "memory_used": mem_used,
                            "memory_total": mem_total,
                            "memory_used_percent": (mem_used / mem_total) * 100 if mem_total > 0 else 0,
                            "utilization": gpu_util,
                            "source": "nvidia-smi"
                        }
                        return gpu_info
            except Exception as e:
                logger.debug(f"Error getting NVIDIA GPU info: {str(e)}")
            
            # Try rocm-smi for AMD GPUs if nvidia-smi failed
            try:
                result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], capture_output=True, text=True, check=True)
                
                if result.returncode == 0:
                    # Parse rocm-smi output (more complex, may need adjustment)
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        mem_used = 0
                        mem_total = 0
                        
                        for line in lines:
                            if 'Used' in line:
                                used_parts = line.split(':')
                                if len(used_parts) > 1:
                                    try:
                                        mem_used = float(used_parts[1].strip().split()[0])
                                    except:
                                        pass
                            if 'Total' in line:
                                total_parts = line.split(':')
                                if len(total_parts) > 1:
                                    try:
                                        mem_total = float(total_parts[1].strip().split()[0])
                                    except:
                                        pass
                        
                        gpu_info = {
                            "memory_used": mem_used,
                            "memory_total": mem_total,
                            "memory_used_percent": (mem_used / mem_total) * 100 if mem_total > 0 else 0,
                            "utilization": None,  # rocm-smi requires different commands for utilization
                            "source": "rocm-smi"
                        }
                        return gpu_info
            except Exception as e:
                logger.debug(f"Error getting AMD GPU info: {str(e)}")
            
            # If we get here, we couldn't get GPU info
            return {
                "memory_used": 0,
                "memory_total": 0,
                "memory_used_percent": 0,
                "utilization": 0,
                "source": "none"
            }
            
        except Exception as e:
            logger.error(f"Error getting GPU memory usage: {str(e)}")
            return {
                "memory_used": 0,
                "memory_total": 0,
                "memory_used_percent": 0,
                "utilization": 0,
                "error": str(e),
                "source": "error"
            }
    
    def get_system_memory_usage(self) -> Dict[str, Any]:
        """
        Get current system memory usage.
        
        Returns:
            Dictionary with system memory usage information
        """
        try:
            memory = psutil.virtual_memory()
            return {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
                "formatted_total": self._format_size(memory.total),
                "formatted_used": self._format_size(memory.used)
            }
        except Exception as e:
            logger.error(f"Error getting system memory usage: {str(e)}")
            return {
                "total": 0,
                "available": 0,
                "used": 0,
                "percent": 0,
                "error": str(e)
            }
    
    def initialize_model(self, model_name: str, model_params: Dict[str, Any] = None) -> Optional[BaseLanguageModel]:
        """
        Initialize an Ollama model with enhanced GPU support.
        
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
        
        # Enable GPU acceleration if available and forced
        if self.force_gpu:
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
            
            # Separate Ollama-specific parameters from other parameters
            ollama_params = {}
            
            # Only include supported parameters
            # Note: Different versions of the Ollama client support different parameters
            supported_params = [
                "temperature", 
                "model", 
                "base_url", 
                "keep_alive",
                "num_ctx",
                "repeat_penalty",
                "top_k",
                "top_p"
            ]
            
            # Extract supported parameters
            for key, value in model_params.items():
                if key in supported_params:
                    ollama_params[key] = value
            
            # Handle GPU layers separately as a model kwarg
            if "n_gpu_layers" in model_params:
                # Some versions use num_gpu instead of n_gpu_layers
                gpu_layers = model_params["n_gpu_layers"]
                
                # Try to use num_gpu parameter via model_kwargs if available
                if hasattr(Ollama, "model_kwargs") or "model_kwargs" in inspect.signature(Ollama.__init__).parameters:
                    ollama_params["model_kwargs"] = {"num_gpu": gpu_layers}
                else:
                    # Older versions - try direct parameter
                    logger.info(f"Using older Ollama client style for GPU layers: {gpu_layers}")
                    # Only include if positive to avoid validation errors
                    if gpu_layers > 0:
                        ollama_params["num_gpu"] = gpu_layers
            
            # Log the parameters being used
            logger.info(f"Initializing model {model_name} with params: {ollama_params}")
            
            # Create the Ollama model 
            try:
                llm = Ollama(
                    base_url=self.ollama_base_url,
                    model=model_name,
                    temperature=temperature,
                    **ollama_params
                )
            except TypeError as e:
                # If the above fails due to unexpected parameters, try with minimal params
                logger.warning(f"Error with full params: {str(e)}, trying minimal params")
                llm = Ollama(
                    base_url=self.ollama_base_url,
                    model=model_name,
                    temperature=temperature
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
                
                # Log GPU info if available
                gpu_info = self.check_gpu_availability()
                if gpu_info.get("has_gpu", False):
                    logger.info(f"Successfully initialized model {model_name} with GPU support")
                else:
                    logger.info(f"Successfully initialized model {model_name} (CPU only)")
                    
                return llm
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {str(e)}")
                return None
                    
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {str(e)}")
            return None
    
    def initialize_model_from_env(self, model_key: str, temperature_key: str) -> Optional[BaseLanguageModel]:
        """
        Initialize a model using environment variables with enhanced GPU support.
        
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
        
        # Enhanced GPU parameters
        if self.force_gpu:
            gpu_info = self.check_gpu_availability()
            if gpu_info.get("has_gpu", False):
                # Set GPU-specific parameters
                model_params["n_gpu_layers"] = self.gpu_layers
                
                # Add additional GPU optimizations
                model_params["f16_kv"] = True  # Use half-precision for key/value cache
                model_params["logits_all"] = False  # Don't compute logits for all tokens (faster)
                
                logger.info(f"Enabling GPU acceleration for model {model_name}")
        
        return self.initialize_model(model_name, model_params)
    
    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific model with enhanced GPU support.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Default parameters for the model
        """
        # Basic defaults - without problematic parameters
        params = {
            "temperature": 0.7,
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
        
        # Add enhanced GPU parameters if GPU is available and enabled
        if self.force_gpu:
            gpu_info = self.check_gpu_availability()
            if gpu_info.get("has_gpu", False):
                # Add GPU-specific parameters - n_gpu_layers will be handled separately
                params["n_gpu_layers"] = self.gpu_layers  # Use specified number of GPU layers
                
                # We'll remove these parameters that cause validation errors with some clients
                # params["f16_kv"] = True  # Removed to avoid validation errors
                # params["logits_all"] = False  # Removed to avoid validation errors
        
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
        
    def check_gpu_availability(self, extended: bool = False) -> Dict[str, Any]:
        """
        Check if GPU is available for Ollama with enhanced details and fallback detection.
        
        Args:
            extended: Whether to include extended GPU info
            
        Returns:
            Dict with GPU availability information
        """
        try:
            # First try the hardware endpoint (newer Ollama versions)
            try:
                response = requests.get(f"{self.ollama_base_url}/api/hardware", timeout=5)
                
                if response.status_code == 200:
                    hardware_info = response.json()
                    gpu_info = hardware_info.get("gpu", {})
                    
                    # Check if GPU is available
                    has_gpu = bool(gpu_info)
                    
                    # Format GPU information
                    if has_gpu:
                        gpu_name = gpu_info.get("name", "Unknown")
                        gpu_memory = gpu_info.get("memory", {})
                        memory_total = gpu_memory.get("total", "Unknown")
                        memory_used = gpu_memory.get("used", "Unknown")
                        
                        result = {
                            "has_gpu": True,
                            "gpu_name": gpu_name,
                            "memory_total": memory_total,
                            "memory_used": memory_used,
                            "memory_free": memory_total - memory_used if isinstance(memory_total, (int, float)) and isinstance(memory_used, (int, float)) else "Unknown",
                            "formatted_total": self._format_size(memory_total) if isinstance(memory_total, (int, float)) else "Unknown",
                            "formatted_used": self._format_size(memory_used) if isinstance(memory_used, (int, float)) else "Unknown",
                            "message": f"GPU detected: {gpu_name} with {self._format_size(memory_total) if isinstance(memory_total, (int, float)) else 'Unknown'} memory"
                        }
                        
                        # Add extended GPU info if requested
                        if extended:
                            # Try to get GPU utilization and temperature
                            gpu_usage = self.get_gpu_memory_usage()
                            result.update({
                                "utilization": gpu_usage.get("utilization", 0),
                                "memory_used_percent": gpu_usage.get("memory_used_percent", 0),
                                "system_memory": self.get_system_memory_usage()
                            })
                        
                        return result
                    
                    return {
                        "has_gpu": False,
                        "message": "No GPU detected for Ollama"
                    }
                
                elif response.status_code == 404:
                    # Hardware endpoint not available - try alternative detection methods
                    #logger.info("Ollama /api/hardware endpoint not found. Using alternative GPU detection.")
                    return self._detect_gpu_alternative(extended)
            except:
                # API endpoint might not exist in this Ollama version
                return self._detect_gpu_alternative(extended)
                    
            # If we reach here, fall back to alternative methods
            return self._detect_gpu_alternative(extended)
                
        except Exception as e:
            logger.error(f"Error checking GPU availability: {str(e)}")
            return {
                "has_gpu": False,
                "message": f"Error checking GPU: {str(e)}"
            }

    def _detect_gpu_alternative(self, extended: bool = False) -> Dict[str, Any]:
        """Alternative methods to detect GPU when the /api/hardware endpoint isn't available"""
        # Method 1: Check for NVIDIA GPU using host command
        try:
            # Try to use nvidia-smi to check for GPU
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'], 
                                capture_output=True, text=True, check=False)
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse nvidia-smi output
                gpu_data = result.stdout.strip().split(',')
                if len(gpu_data) >= 3:
                    gpu_name = gpu_data[0].strip()
                    memory_total = float(gpu_data[1].strip()) * 1024 * 1024  # Convert to bytes
                    memory_used = float(gpu_data[2].strip()) * 1024 * 1024  # Convert to bytes
                    
                    result = {
                        "has_gpu": True,
                        "gpu_name": gpu_name,
                        "memory_total": memory_total,
                        "memory_used": memory_used,
                        "memory_free": memory_total - memory_used,
                        "formatted_total": self._format_size(memory_total),
                        "formatted_used": self._format_size(memory_used),
                        "message": f"GPU detected: {gpu_name} with {self._format_size(memory_total)} memory",
                        "detection_method": "nvidia-smi"
                    }
                    
                    # Add extended GPU info if requested
                    if extended:
                        gpu_usage = self.get_gpu_memory_usage()
                        result.update({
                            "utilization": gpu_usage.get("utilization", 0),
                            "memory_used_percent": gpu_usage.get("memory_used_percent", 0),
                            "system_memory": self.get_system_memory_usage()
                        })
                    
                    return result
        except:
            pass
        
        # Method 2: Generate with GPU flag and check response
        try:
            # Try a simple generate request with GPU flag to see if it works
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={"model": self.default_model, "prompt": "test", "options": {"num_gpu": 1}},
                timeout=5
            )
            
            if response.status_code == 200:
                # If we get a successful response, GPU might be available
                return {
                    "has_gpu": True,
                    "gpu_name": "GPU Detected",
                    "message": "GPU appears to be working with Ollama",
                    "detection_method": "generation-test"
                }
        except:
            pass
        
        # Method 3: Check environment for GPU
        if os.getenv("ENABLE_GPU", "false").lower() == "true" and self.force_gpu:
            # User has explicitly configured GPU, so assume it's available even if we can't detect it
            return {
                "has_gpu": True,
                "gpu_name": "GPU (Assumed from environment)",
                "message": "GPU assumed available based on environment configuration",
                "detection_method": "environment-config"
            }
        
        # If all methods fail, assume no GPU
        return {
            "has_gpu": False,
            "message": "No GPU detected for Ollama (after trying multiple detection methods)",
            "system_memory": self.get_system_memory_usage() if extended else {}
        }    
    
    def enable_gpu_for_model(self, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enable GPU acceleration for model parameters with enhanced settings.
        
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
            
            # Set number of GPU layers - this will be handled by the initialize_model method
            updated_params["n_gpu_layers"] = self.gpu_layers  # Use specified number of GPU layers
            
            # Don't add f16_kv directly as it's not supported by some Ollama client versions
            # Instead, we'll use it as a model parameter if supported
            
            # If we have GPU memory information, adjust parameters based on available memory
            if "memory_total" in gpu_info and isinstance(gpu_info["memory_total"], (int, float)):
                memory_gb = gpu_info["memory_total"] / (1024 * 1024 * 1024)
                
                if memory_gb < 4:
                    # Very limited GPU memory - be more conservative
                    updated_params["n_gpu_layers"] = min(24, self.gpu_layers if self.gpu_layers > 0 else 999)
                elif memory_gb < 8:
                    # Limited GPU memory
                    updated_params["n_gpu_layers"] = min(32, self.gpu_layers if self.gpu_layers > 0 else 999)
            
            logger.info(f"Enabled GPU acceleration with {gpu_info['gpu_name']}")
            return updated_params
        else:
            logger.warning(f"GPU not available: {gpu_info['message']}")
            return model_params
    
    def get_active_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active models being used by the system.
        
        Returns:
            Dictionary mapping roles to model information
        """
        active_models = {}
        
        # Get current model selections
        roles = ["generative", "review", "summary", "compare"]
        
        for role in roles:
            model_name = os.getenv(f"{role.upper()}_MODEL", self.default_model)
            
            # Get model details
            model_info = {
                "name": model_name,
                "role": role,
                "temperature": float(os.getenv(f"{role.upper()}_TEMPERATURE", "0.7")),
                "uses_gpu": False,
                "details": {}
            }
            
            # Check if model is using GPU
            if model_name in self.initialized_models:
                # Model is initialized, check if it has GPU parameters
                llm = self.initialized_models[model_name]
                if hasattr(llm, "client"):
                    client_params = getattr(llm.client, "_client_params", {})
                    if client_params.get("n_gpu_layers", 0) != 0:
                        model_info["uses_gpu"] = True
            
            # Try to get model details
            try:
                details = self.get_model_details(model_name)
                model_info["details"] = details
                model_info["gpu_optimized"] = details.get("gpu_optimized", False)
            except:
                pass
            
            active_models[role] = model_info
        
        return active_models