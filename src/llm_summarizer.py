"""
LLM Summarizer module for generating function summaries using an LLM.
"""
import os
import json
from typing import Dict, List, Optional
import ollama

from src.utils.logger import get_logger
from src.cpp_parser import CppFunction

logger = get_logger()

class LLMSummarizer:
    """
    Generates summaries of C++ functions using an LLM model via Ollama.
    
    This class handles:
    1. Generating prompts for function summarization
    2. Caching results to avoid redundant LLM calls
    3. Ensuring the model is available in Ollama
    """
    
    def __init__(self, model="gemma3:1b", cache_file="function_summaries.json", use_fallback=False, use_cache=True):
        """
        Initialize the LLM summarizer.
        
        Args:
            model (str): Name of the model to use with Ollama
            cache_file (str): Path to the cache file for function summaries
            use_fallback (bool): Force using fallback mode without LLM
            use_cache (bool): Whether to use the cache for function summaries
        """
        self.model = model
        self.cache_file = cache_file
        self.cache = {}
        self.use_fallback = use_fallback
        self.use_cache = use_cache
        
        if not self.use_cache:
            logger.info("Cache disabled for function summarization - all functions will be summarized using LLM")
        
        # Load cache if exists
        self._load_cache()
        
        if not self.use_fallback:
            try:
                # Ensure model is available
                self._ensure_model_available()
                logger.info(f"Initialized LLM summarizer with model: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM, using fallback mode: {str(e)}")
                self.use_fallback = True
        
        if self.use_fallback:
            logger.info("Using fallback mode for function summarization (no LLM)")
    
    def _ensure_model_available(self):
        """Ensure the specified model is available in Ollama."""
        try:
            logger.info(f"Checking if model '{self.model}' is available")
            ollama.list()
            logger.info("Successfully connected to Ollama")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            raise RuntimeError(f"Failed to connect to Ollama: {str(e)}")
        
        try:
            # Check if we need to pull the model
            models = ollama.list()
            
            # Handle different response formats from different Ollama versions
            model_names = []
            
            if 'models' in models:
                # Newer Ollama versions have a 'models' key with a list of model objects
                for model_obj in models.get('models', []):
                    if isinstance(model_obj, dict) and 'name' in model_obj:
                        model_names.append(model_obj['name'])
                    elif hasattr(model_obj, 'name'):
                        model_names.append(model_obj.name)
            else:
                # Older versions might return a list directly
                for model_obj in models:
                    if isinstance(model_obj, dict) and 'name' in model_obj:
                        model_names.append(model_obj['name'])
                    elif hasattr(model_obj, 'name'):
                        model_names.append(model_obj.name)
            
            logger.info(f"Available models: {', '.join(model_names) if model_names else 'none'}")
            
            if not model_names or self.model not in model_names:
                logger.info(f"Model '{self.model}' not found, pulling from Ollama...")
                ollama.pull(self.model)
                logger.info(f"Successfully pulled model '{self.model}'")
            else:
                logger.info(f"Model '{self.model}' is already available")
        except Exception as e:
            logger.error(f"Error ensuring model availability: {str(e)}")
            raise
    
    def _load_cache(self):
        """Load the function summary cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} function summaries from cache")
            except Exception as e:
                logger.warning(f"Error loading cache file: {str(e)}")
                self.cache = {}
        else:
            logger.info("No cache file found, starting with empty cache")
            self.cache = {}
    
    def _save_cache(self):
        """Save the function summary cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Saved {len(self.cache)} function summaries to cache")
        except Exception as e:
            logger.warning(f"Error saving cache file: {str(e)}")
    
    def _format_prompt(self, function_content: str) -> str:
        """
        Format a prompt for the LLM to summarize a function.
        
        Args:
            function_content (str): The content of the function to summarize
            
        Returns:
            str: Formatted prompt
        """
        return f"""You are a technical writer documenting C++ code.
Write a concise, accurate summary of the following C++ function. 
Focus on explaining WHAT the function does, not HOW it does it.
Use at most 2 sentences.

```cpp
{function_content}
```

Your summary: """
    
    def generate_summary(self, function: CppFunction) -> str:
        """
        Generate a summary for a C++ function.
        
        Args:
            function (CppFunction): The function to summarize
            
        Returns:
            str: Function summary
        """
        # Check cache first
        if function.qualified_name in self.cache:
            logger.debug(f"Using cached summary for {function.qualified_name}")
            return self.cache[function.qualified_name]
        
        if self.use_fallback:
            # Generate a basic summary from function name if in fallback mode
            summary = self._generate_fallback_summary(function)
        else:
            # Generate summary using LLM
            try:
                prompt = self._format_prompt(function.content)
                response = ollama.chat(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                
                summary = response['message']['content'].strip()
                
                # Basic validation - require at least 10 characters
                if len(summary) < 10:
                    logger.warning(f"LLM returned unusable summary for {function.qualified_name}, using fallback")
                    summary = self._generate_fallback_summary(function)
                
            except Exception as e:
                logger.error(f"Error generating summary for {function.qualified_name}: {str(e)}")
                summary = self._generate_fallback_summary(function)
        
        # Cache the result
        self.cache[function.qualified_name] = summary
        
        # Periodically save cache to disk
        if len(self.cache) % 10 == 0:
            self._save_cache()
        
        return summary
    
    def _generate_fallback_summary(self, function: CppFunction) -> str:
        """
        Generate a basic summary from function name and content without using LLM.
        
        Args:
            function (CppFunction): The function to summarize
            
        Returns:
            str: A basic summary
        """
        # Extract the function name from qualified name
        name_parts = function.qualified_name.split('::')
        simple_name = name_parts[-1] if name_parts else function.name
        
        # Generate a summary based on function name
        if simple_name.startswith('get') and len(simple_name) > 3:
            return f"Retrieves the {simple_name[3:]} value."
        elif simple_name.startswith('set') and len(simple_name) > 3:
            return f"Sets the {simple_name[3:]} value."
        elif simple_name.startswith('is') and len(simple_name) > 2:
            return f"Checks if {simple_name[2:]} condition is met."
        elif simple_name.startswith('has') and len(simple_name) > 3:
            return f"Checks if {simple_name[3:]} is present."
        elif simple_name.startswith('calculate') or simple_name.startswith('compute'):
            return f"Calculates and returns {simple_name.replace('calculate', '').replace('compute', '')}."
        elif simple_name.startswith('create'):
            return f"Creates a new {simple_name.replace('create', '')} instance."
        elif simple_name.startswith('validate'):
            return f"Validates {simple_name.replace('validate', '')}."
        elif simple_name.startswith('parse'):
            return f"Parses {simple_name.replace('parse', '')}."
        elif simple_name.startswith('initialize'):
            return f"Initializes {simple_name.replace('initialize', '')}."
        elif simple_name.startswith('update'):
            return f"Updates {simple_name.replace('update', '')}."
        else:
            # Check if function has a short comment that can be used
            first_line = function.content.split('\n')[0] if function.content else ""
            if '//' in first_line:
                comment = first_line.split('//')[1].strip()
                if len(comment) > 10:
                    return comment
                
            # Default fallback
            return f"Implements {simple_name} functionality."
    
    def summarize_functions(self, functions: List[CppFunction]) -> Dict[str, str]:
        """
        Generate summaries for a list of functions.
        
        Args:
            functions (List[CppFunction]): List of functions to summarize
            
        Returns:
            Dict[str, str]: Map of function qualified names to summaries
        """
        summaries = {}
        
        total = len(functions)
        logger.info(f"Generating summaries for {total} functions...")
        
        for i, function in enumerate(functions):
            if i % 10 == 0 and i > 0:
                logger.info(f"Progress: {i}/{total} functions summarized")
            
            summary = self.generate_summary(function)
            summaries[function.qualified_name] = summary
        
        # Save the final cache
        self._save_cache()
        
        logger.info(f"Generated summaries for {len(summaries)} functions")
        return summaries
