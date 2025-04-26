"""
TinyLlama Mapper module that extends MappingEngine to use the trained TinyLlama model with LoRA weights.
"""
import os
import torch
import sys
from typing import Dict, List, Set, Tuple

from src.cpp_parser import CppFunction
from src.mapping_engine import MappingEngine
from src.utils.logger import get_logger

logger = get_logger()

# Flag to check if Ollama was imported - will be set to False if import fails
OLLAMA_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    # Silently continue if Ollama is not available
    pass

class TinyLlamaMapper(MappingEngine):
    """
    Maps requirements to functions using the trained TinyLlama model with LoRA weights.
    
    This class extends the base MappingEngine but uses local TinyLlama model instead of Ollama,
    leveraging the trained LoRA weights for better requirement-to-function matching.
    """
    
    def __init__(self, similarity_threshold=0.5, model_path=None, strict_mode=False):
        """
        Initialize the TinyLlama mapper.
        
        Args:
            similarity_threshold (float): Threshold for considering a match
            model_path (str): Path to the trained TinyLlama model with LoRA weights
                             (if None, will use the latest epoch from rl_weights)
            strict_mode (bool): If True, never fall back to Ollama even if model fails to load
        """
        # Set default attributes to prevent attribute errors
        self.model_path = model_path
        self.local_model = None
        self.local_tokenizer = None
        self.use_fallback = False
        
        # Check if running with --use-tinyllama flag to enforce strict mode
        self.strict_mode = strict_mode or "--use-tinyllama" in sys.argv
        
        # Initialize with a dummy model name since we'll override the LLM usage
        super().__init__(similarity_threshold=similarity_threshold, model="dummy")
        
        # Don't load the model now to save memory - we'll load it on first use
        if self.strict_mode:
            logger.info(f"Initialized TinyLlama mapper in STRICT mode - will NOT use Ollama fallback")
        else:
            logger.info(f"Initialized TinyLlama mapper in standard mode - may use Ollama as fallback if needed")
            
        logger.info(f"TinyLlama model path: {model_path}")
    
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
        self.use_fallback = fallback_needed and not self.strict_mode
        
        # Verify if model was actually loaded correctly
        if self.local_model is not None:
            # Force fallback to false since model was loaded successfully
            self.use_fallback = False
            logger.info("TinyLlama model loaded successfully, will NOT fall back to Ollama")
        
        # If we couldn't load the model, log a clear message
        if fallback_needed:
            if self.strict_mode:
                error_msg = "Failed to load TinyLlama model and strict mode is enabled. No fallback will be used."
                logger.error(error_msg)
                # Use a default low confidence for all mappings instead of failing
                logger.warning("Using default low confidence scores for all mappings")
            else:
                logger.warning("Falling back to Ollama for requirement mapping due to errors loading the TinyLlama model")
    
    def get_llm_confidence(self, requirement_desc: str, module_id: str, function_name: str, 
                           function_summary: str, function_content: str) -> float:
        """
        Use the trained TinyLlama model to determine confidence in matching a requirement to a function.
        
        Args:
            requirement_desc (str): The requirement description
            module_id (str): The module identifier
            function_name (str): The name of the function
            function_summary (str): The summary of the function
            function_content (str): The content of the function
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        # First load the model to ensure it's available
        if self.local_model is None:
            logger.info(f"Loading TinyLlama model for '{function_name}'...")
            self._ensure_model_available()
        
        # Check if model is actually available (explicit log for debugging)
        if self.local_model is None:
            logger.warning(f"TinyLlama model is still None for '{function_name}', will use fallback")
            self.use_fallback = True
            
        # Now make the decision on which implementation to use
        if self.use_fallback or self.local_model is None:
            if OLLAMA_AVAILABLE:
                logger.info(f"Using Ollama fallback for '{function_name}' with requirement: '{requirement_desc[:30]}...'")
                try:
                    # Use the parent implementation which uses Ollama
                    return super().get_llm_confidence(
                        requirement_desc, module_id, function_name, function_summary, function_content
                    )
                except Exception as e:
                    logger.error(f"Error using Ollama fallback: {str(e)}")
                    return 0.2  # Return a modest confidence on error
            else:
                # Ollama is not available, so just return a default score
                logger.warning(f"Ollama not available as fallback for '{function_name}', using default confidence")
                return 0.2  # Return a modest default confidence
                
        # Use the TinyLlama model since it's available
        try:
            # Import the required function for prediction
            from inference import predict_requirement_match
            
            # Log that we're using TinyLlama
            logger.info(f"Using TinyLlama model for '{function_name}' with requirement: '{requirement_desc[:30]}...'")
            logger.info(f"Using TinyLlama model for '{function_name}' with summary: '{function_summary[:50]}...'")
            # Use our trained model to predict confidence
            prediction = predict_requirement_match(
                self.local_model,
                self.local_tokenizer,
                requirement_desc,
                function_summary
            )
            
            # Get the confidence score (normalized from 0-100 to 0-1)
            confidence = prediction.get('rating', 0) / 100.0
            
            logger.info(f"TinyLlama confidence for matching '{function_name}' to requirement: {confidence:.2f}")
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting TinyLlama confidence: {str(e)}")
            
            # Only try Ollama if available and fallback is allowed
            if OLLAMA_AVAILABLE and not self.strict_mode:
                logger.warning("Attempting Ollama fallback after TinyLlama error")
                try:
                    return super().get_llm_confidence(
                        requirement_desc, module_id, function_name, function_summary, function_content
                    )
                except Exception as fallback_error:
                    logger.error(f"Error using Ollama fallback: {str(fallback_error)}")
            
            # Return a default low confidence if no fallback or fallback failed
            return 0.1
