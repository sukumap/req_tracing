"""
Gemma3 Mapper module that extends MappingEngine to use the Gemma3 model via Ollama.
"""
import os
import sys
import requests
from typing import Dict, List, Set, Tuple

from src.cpp_parser import CppFunction
from src.mapping_engine import MappingEngine
from src.utils.logger import get_logger

logger = get_logger()

# Ollama API URL
OLLAMA_API_URL = "http://localhost:11434/api/generate"

class Gemma3Mapper(MappingEngine):
    """
    Maps requirements to functions using the Gemma3 model via Ollama.
    
    This class extends the base MappingEngine but uses Gemma3 instead of other models,
    providing a direct comparison with TinyLlama's performance.
    """
    
    def __init__(self, similarity_threshold=0.5):
        """
        Initialize the Gemma3 mapper.
        
        Args:
            similarity_threshold (float): Threshold for considering a match
        """
        # Use 'gemma3:1b' as the model name for Ollama
        super().__init__(similarity_threshold=similarity_threshold, model="gemma3:1b")
        
        # Set specific attributes for Gemma3
        self.model_name = "gemma3:1b"
        
        # Check if Ollama is available
        self._check_ollama_availability()
        
        logger.info(f"Initialized Gemma3 mapper with model: {self.model_name}")
    
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
    
    def get_llm_confidence(self, requirement_desc: str, module_id: str, function_name: str, 
                          function_summary: str, function_content: str) -> float:
        """
        Use the Gemma3 model to determine confidence in matching a requirement to a function.
        
        Args:
            requirement_desc (str): The requirement description
            module_id (str): The module identifier
            function_name (str): The name of the function
            function_summary (str): The summary of the function
            function_content (str): The content of the function
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        try:
            # Use the exact same prompt format as used in TinyLlama training for fair comparison
            prompt = f"""Rate how well the function summary implements the given requirement.

Requirement: {requirement_desc}

Function Summary: {function_summary}

Think step by step about whether the function summary matches the requirement:
1. What functionality does the requirement specify?
2. What functionality does the summary describe?
3. How much overlap exists between these functionalities?

Scoring Guide:
- 0-10: No relationship whatsoever
- 11-30: Slightly related but incorrect implementation
- 31-60: Partially implements the requirement but with significant gaps
- 61-89: Mostly implements the requirement with minor issues
- 90-100: Correctly and completely implements the requirement

Based on the scoring guide, provide a single integer between 0 and 100 that indicates how well the function implements the requirement.

Rating:"""
            
            # Call Ollama API directly
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": 0.1,  # Use low temperature for more deterministic output
                    "max_tokens": 10,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.text}")
                return 0.2  # Return modest confidence on error
            
            # Extract and parse the response
            response_json = response.json()
            rating_text = response_json.get("response", "").strip()
            
            # Parse rating
            import re
            number_match = re.search(r'\b(\d+)\b', rating_text)
            if number_match:
                rating = int(number_match.group(1))
                # Clamp between 0 and 100
                rating = min(100, max(0, rating))
                # Normalize to 0-1 range
                confidence = rating / 100.0
            else:
                # Fallback if no number found
                logger.warning(f"Could not find a rating number in Gemma3 response: '{rating_text}'")
                confidence = 0.2  # Default modest confidence
            
            logger.info(f"Gemma3 confidence for matching '{function_name}' to requirement: {confidence:.2f}")
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting Gemma3 confidence: {str(e)}")
            return 0.2  # Return modest confidence on error
