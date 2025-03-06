"""
LLM Manager for Peer Review Agent.

This module handles LLM initialization, configuration, and management
for the peer code review agent, providing a unified interface to interact
with Ollama models.
"""

import os
import requests
import time
import logging
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
            {"id": "mistral", "name": "Mistral 7B", "description": "Mistral AI's 7B model", "pulled": False}
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
            
    def download_ollama_model(self, model_name: str) -> bool:
        """
        Download a model using Ollama.
        
        Args:
            model_name (str): Name of the model to download
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Start the pull operation
            response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to start model download: {response.text}")
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
                            logger.info(f"Model {model_name} downloaded successfully!")
                            break
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    # Log error but continue polling
                    logger.warning(f"Error checking model status: {str(e)}")
                    time.sleep(5)
            
            if not model_ready:
                logger.warning(f"Download timeout for {model_name}. It may still be downloading.")
                return False
            
            return True
                
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return False
    
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
        Initialize an Ollama model.
        
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
            
            # Initialize Ollama model
            temperature = model_params.get("temperature", 0.7)
            
            llm = Ollama(
                base_url=self.ollama_base_url,
                model=model_name,
                temperature=temperature
            )
            
            # Test the model with a simple query
            try:
                _ = llm.invoke("hello")
                # If successful, cache the model
                self.initialized_models[model_name] = llm
                logger.info(f"Successfully initialized model {model_name}")
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
        
        model_params = {
            "temperature": temperature
        }
        
        return self.initialize_model(model_name, model_params)
    
    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Default parameters for the model
        """
        # Basic defaults
        params = {
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        # Adjust based on model name
        if "generative" in model_name or any(gen in model_name for gen in ["llama3", "llama-3"]):
            params["temperature"] = 0.8  # Slightly higher creativity for generative tasks
            
        elif "review" in model_name or any(rev in model_name for rev in ["mistral", "deepseek"]):
            params["temperature"] = 0.3  # Lower temperature for review tasks
            
        elif "summary" in model_name:
            params["temperature"] = 0.4  # Moderate temperature for summary tasks
            
        elif "compare" in model_name:
            params["temperature"] = 0.5  # Balanced temperature for comparison tasks
        
        return params