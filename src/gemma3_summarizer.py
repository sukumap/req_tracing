"""
Gemma3 Summarizer module that extends LLMSummarizer to use the Gemma3 model via Ollama.
"""
import os
import sys
import json
import requests
from typing import Dict, Optional

from src.utils.logger import get_logger
from src.cpp_parser import CppFunction
from src.llm_summarizer import LLMSummarizer

logger = get_logger()

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

class Gemma3Summarizer(LLMSummarizer):
    """
    Generates summaries of C++ functions using the Gemma3 model via Ollama.
    
    This class extends the base LLMSummarizer but uses Gemma3 instead of other models,
    providing a direct comparison with TinyLlama's performance.
    """
    
    def __init__(self, cache_file="gemma3_summaries.json", use_cache=True):
        """
        Initialize the Gemma3 summarizer.
        
        Args:
            cache_file (str): Path to the cache file for function summaries
            use_cache (bool): Whether to use cache for function summaries
        """
        # Use 'gemma3:1b' as the model name for Ollama
        super().__init__(model="gemma3:1b", cache_file=cache_file, use_fallback=False, use_cache=use_cache)
        
        # Set specific attributes for Gemma3
        self.model_name = "gemma3:1b"
        
        if not use_cache:
            logger.info("Cache disabled for Gemma3 - all functions will be summarized with the model")
        
        # Load cache if exists
        self._load_cache()
        
        # Check if Ollama is available
        self._check_ollama_availability()
        
        logger.info(f"Initialized Gemma3 summarizer with model: {self.model_name}")
    
    def _check_ollama_availability(self):
        """Check if Ollama is available and the gemma3:1b model is installed."""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if self.model_name in model_names:
                    logger.info(f"Ollama is available with {self.model_name} model")
                else:
                    logger.warning(f"{self.model_name} model not found in Ollama. Available models: {model_names}")
                    logger.warning("Please run 'ollama pull gemma3:1b' to download the model")
            else:
                logger.error(f"Ollama server error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama server. Make sure it's running on port 11434")
        except Exception as e:
            logger.error(f"Error checking Ollama: {str(e)}")
    
    def generate_summary(self, function: CppFunction) -> str:
        """
        Generate a summary for a C++ function using Gemma3 model via Ollama.
        
        Args:
            function (CppFunction): The function to summarize
            
        Returns:
            str: Function summary
        """
        # Check if we have a cached summary
        if function.qualified_name in self.cache:
            return self.cache[function.qualified_name]
        
        # Format the prompt
        prompt = f"""Summarize this code function in a concise way that captures its main purpose and functionality:

```
{function.content}
```

Provide a clear, specific Function Summary that highlights what the code actually does:"""
        
        try:
            # Call Ollama API directly
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.text}")
                summary = self._generate_fallback_summary(function)
            else:
                summary_json = response.json()
                summary = summary_json.get("response", "").strip()
                
                # Basic validation - require at least 10 characters
                if len(summary) < 10:
                    logger.warning(f"Model returned unusable summary for {function.qualified_name}, using fallback")
                    summary = self._generate_fallback_summary(function)
        except Exception as e:
            logger.error(f"Error generating summary with Gemma3: {str(e)}")
            summary = self._generate_fallback_summary(function)
        
        # Cache the result
        self.cache[function.qualified_name] = summary
        
        # Periodically save cache to disk
        if len(self.cache) % 10 == 0:
            self._save_cache()
        
        return summary
