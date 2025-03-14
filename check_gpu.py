#!/usr/bin/env python3
"""
GPU Check for Ollama

This script directly checks if GPU acceleration is available for Ollama
using multiple detection methods.
"""
import requests
import subprocess
import json
import os
import sys

def check_gpu():
    """Check if GPU acceleration is available for Ollama"""
    print("Checking for GPU availability with Ollama...\n")
    
    # Define Ollama API URL
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"Using Ollama URL: {ollama_url}")
    
    # Method 1: Check /api/hardware endpoint (newer Ollama versions)
    print("\n1. Checking /api/hardware endpoint...")
    try:
        response = requests.get(f"{ollama_url}/api/hardware", timeout=5)
        if response.status_code == 200:
            hardware_info = response.json()
            gpu_info = hardware_info.get("gpu", {})
            if gpu_info:
                print("✅ GPU detected via API:")
                print(f"   Name: {gpu_info.get('name', 'Unknown')}")
                memory = gpu_info.get("memory", {})
                total = memory.get("total", 0)
                used = memory.get("used", 0)
                print(f"   Memory: {used/(1024*1024*1024):.2f} GB / {total/(1024*1024*1024):.2f} GB")
                return True
            else:
                print("❌ No GPU detected via /api/hardware endpoint")
        else:
            print(f"❌ /api/hardware endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking /api/hardware: {str(e)}")
    
    # Method 2: Check Ollama model list for GPU-enabled models
    print("\n2. Checking for GPU-enabled models...")
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            gpu_models = 0
            for model in models:
                try:
                    model_info = requests.get(f"{ollama_url}/api/show?name={model['name']}", timeout=5).json()
                    modelfile = model_info.get("modelfile", "").lower()
                    if "gpu" in modelfile or "cuda" in modelfile:
                        gpu_models += 1
                        print(f"✅ GPU-enabled model found: {model['name']}")
                except:
                    pass
            if gpu_models > 0:
                print(f"✅ Found {gpu_models} GPU-enabled models")
                return True
            else:
                print("❌ No GPU-enabled models found")
    except Exception as e:
        print(f"❌ Error checking models: {str(e)}")
    
    # Method 3: Try a generate request with GPU parameters
    print("\n3. Testing generation with GPU parameters...")
    try:
        # Try a simple generation with num_gpu set
        models = requests.get(f"{ollama_url}/api/tags", timeout=5).json().get("models", [])
        if models:
            test_model = models[0]["name"]
            print(f"   Testing with model: {test_model}")
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": test_model, "prompt": "hello", "options": {"num_gpu": 1}},
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Generation with GPU parameters succeeded")
                return True
            else:
                print(f"❌ Generation failed with status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        else:
            print("❌ No models available for testing")
    except Exception as e:
        print(f"❌ Error testing generation: {str(e)}")
    
    # Method 4: Check host system for NVIDIA GPU
    print("\n4. Checking host system for NVIDIA GPU...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected on host system")
            print(f"   {result.stdout.split('|')[1].strip()}")
            return True
        else:
            print("❌ No NVIDIA GPU detected or nvidia-smi not available")
    except Exception as e:
        print(f"❌ Error checking for NVIDIA GPU: {str(e)}")
    
    print("\nConclusion: NO GPU acceleration detected for Ollama")
    return False

if __name__ == "__main__":
    has_gpu = check_gpu()
    sys.exit(0 if has_gpu else 1)