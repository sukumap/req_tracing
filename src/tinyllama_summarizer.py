"""
TinyLlama Summarizer module that extends LLMSummarizer to use the trained TinyLlama model with LoRA weights.
"""
import os
import torch
import sys
from typing import Dict, List, Optional

from src.utils.logger import get_logger
from src.cpp_parser import CppFunction
from src.llm_summarizer import LLMSummarizer

logger = get_logger()

class TinyLlamaSummarizer(LLMSummarizer):
    """
    Generates summaries of C++ functions using the trained TinyLlama model with LoRA weights.
    
    This class extends the base LLMSummarizer but uses local TinyLlama model instead of Ollama,
    leveraging the trained LoRA weights for better code summarization.
    """
    
    def __init__(self, model_path=None, cache_file="tinyllama_summaries.json", use_cache=True):
        """
        Initialize the TinyLlama summarizer.
        
        Args:
            model_path (str): Path to the trained TinyLlama model with LoRA weights
                              (if None, will use the latest epoch from rl_weights)
            cache_file (str): Path to the cache file for function summaries
            use_cache (bool): Whether to use cache for function summaries
        """
        # Set default attributes to prevent attribute errors
        self.model_path = model_path
        self.local_model = None
        self.local_tokenizer = None
        self.use_fallback = False
        
        # Initialize with a dummy model name since we'll override the LLM usage
        # This sets up the cache and other base functionality
        super().__init__(model="dummy", cache_file=cache_file, use_fallback=False, use_cache=use_cache)
        
        if not use_cache:
            logger.info("Cache disabled for TinyLlama - all functions will be summarized with the model")
        
        # Load cache if exists
        self._load_cache()
        
        # Try to initialize the model now to catch early errors
        try:
            # Find the most recent epoch if model_path is not specified
            if self.model_path is None:
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                rl_weights_dir = os.path.join(root_dir, "rl_weights")
                
                if not os.path.exists(rl_weights_dir):
                    logger.error(f"RL weights directory not found: {rl_weights_dir}")
                    self.use_fallback = True
                else:
                    # Find most recent epoch directory
                    epoch_dirs = [d for d in os.listdir(rl_weights_dir) if d.startswith("epoch-")]
                    if epoch_dirs:
                        epoch_nums = [int(d.split("-")[1]) for d in epoch_dirs]
                        latest_epoch = max(epoch_nums)
                        self.model_path = os.path.join(rl_weights_dir, f"epoch-{latest_epoch}")
                        logger.info(f"Using most recent epoch: {self.model_path}")
                    else:
                        # If no epoch directories, use the base directory
                        self.model_path = rl_weights_dir
                        logger.info(f"No epoch directories found, using base directory: {self.model_path}")
                        
        except Exception as e:
            logger.warning(f"Error finding model path: {str(e)}, will try again during first use")
            
        # Don't load the model now to save memory - we'll load it on first use
        logger.info(f"Initialized TinyLlama summarizer with model path: {self.model_path}")
    
    def _ensure_model_available(self):
        """Ensure the TinyLlama model with LoRA weights is available and loaded."""
        # If model is already loaded, return immediately
        if self.local_model is not None:
            return
            
        # Set fallback flag to track if anything goes wrong
        fallback_needed = False
            
        try:
            # Find the most recent epoch if model_path is not specified
            if self.model_path is None:
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                rl_weights_dir = os.path.join(root_dir, "rl_weights")
                
                if not os.path.exists(rl_weights_dir):
                    logger.error(f"RL weights directory not found: {rl_weights_dir}")
                    fallback_needed = True
                else:  
                    # Find most recent epoch directory
                    epoch_dirs = [d for d in os.listdir(rl_weights_dir) if d.startswith("epoch-")]
                    if epoch_dirs:
                        epoch_nums = [int(d.split("-")[1]) for d in epoch_dirs]
                        latest_epoch = max(epoch_nums)
                        self.model_path = os.path.join(rl_weights_dir, f"epoch-{latest_epoch}")
                        logger.info(f"Using most recent epoch: {self.model_path}")
                    else:
                        # If no epoch directories, use the base directory
                        self.model_path = rl_weights_dir
                        logger.info(f"No epoch directories found, using base directory: {self.model_path}")
            
            # Only try to load the model if we have a valid path
            if not fallback_needed and self.model_path:
                # Try to import directly from local path first
                try:
                    # First try to import from local directory
                    sys_path = sys.path.copy()
                    
                    # Add parent directory to path to find inference module
                    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                        
                    # Try different import approaches
                    try:
                        # Try direct import first
                        from inference import load_adapted_model
                        logger.info("Successfully imported inference module directly")
                    except ImportError:
                        # Try as a relative import
                        try:
                            from ..inference import load_adapted_model
                            logger.info("Successfully imported inference module as relative import")
                        except ImportError:
                            # Try from src package
                            try:
                                from src.inference import load_adapted_model
                                logger.info("Successfully imported inference module from src package")
                            except ImportError:
                                logger.error("Failed to import inference module from all paths")
                                raise ImportError("Could not import inference module")
                    
                    # Determine device
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Loading TinyLlama model from {self.model_path} on {device}...")
                    
                    # Load model and tokenizer
                    self.local_model, self.local_tokenizer = load_adapted_model(self.model_path, device=device)
                    logger.info(f"Successfully loaded TinyLlama model with LoRA weights")
                    
                    # Restore original sys.path
                    sys.path = sys_path
                    
                except ImportError as ie:
                    logger.error(f"Failed to import inference module: {str(ie)}")
                    fallback_needed = True
                except Exception as e:
                    logger.error(f"Error loading TinyLlama model: {str(e)}")
                    fallback_needed = True
        except Exception as e:
            logger.error(f"Error ensuring model availability: {str(e)}")
            fallback_needed = True
            
        # Set fallback flag if anything went wrong
        self.use_fallback = fallback_needed
        
        # If we couldn't load the model, log a clear message
        if self.use_fallback:
            logger.warning("Falling back to basic summarization due to errors loading the TinyLlama model")
    
    def generate_summary(self, function: CppFunction) -> str:
        """
        Generate a summary for a C++ function using trained TinyLlama model.
        
        Args:
            function (CppFunction): The function to summarize
            
        Returns:
            str: Function summary
        """
        # Check if we have a cached summary
        if function.qualified_name in self.cache:
            return self.cache[function.qualified_name]
        
        # Use fallback if needed
        if self.use_fallback:
            summary = self._generate_fallback_summary(function)
            self.cache[function.qualified_name] = summary
            return summary
        
        # Load model if not already loaded
        self._ensure_model_available()
        
        # If model loading failed, use fallback
        if self.use_fallback or self.local_model is None:
            summary = self._generate_fallback_summary(function)
            self.cache[function.qualified_name] = summary
            return summary
        
        # Format the prompt
        prompt = self._format_prompt(function.content)
        
        try:
            # Import required function for generation
            from inference import generate_function_summary
            
            # Generate the summary
            summary = generate_function_summary(
                self.local_model, 
                self.local_tokenizer, 
                function.content
            )
            
            # Basic validation - require at least 10 characters
            if len(summary) < 10:
                logger.warning(f"Model returned unusable summary for {function.qualified_name}, using fallback")
                summary = self._generate_fallback_summary(function)
                
        except Exception as e:
            logger.error(f"Error generating summary with TinyLlama: {str(e)}")
            summary = self._generate_fallback_summary(function)
        
        # Cache the result
        self.cache[function.qualified_name] = summary
        
        # Periodically save cache to disk
        if len(self.cache) % 10 == 0:
            self._save_cache()
        
        return summary
