"""
Gemma Fine-Tuned Mapper module that extends MappingEngine to use the fine-tuned Gemma 3 model.
"""
import os
import torch
import sys
from typing import Dict, List, Set, Tuple
import re

from src.cpp_parser import CppFunction
from src.mapping_engine import MappingEngine
from src.utils.logger import get_logger

logger = get_logger()

class GemmaFineTunedMapper(MappingEngine):
    """
    Maps requirements to functions using the fine-tuned Gemma 3 model.
    
    This class extends the base MappingEngine but uses a locally fine-tuned Gemma 3 model
    with LoRA weights for better requirement-to-function matching.
    """
    
    def __init__(self, similarity_threshold=0.5, model_path=None):
        """
        Initialize the fine-tuned Gemma mapper.
        
        Args:
            similarity_threshold (float): Threshold for considering a match
            model_path (str): Path to the fine-tuned Gemma model with LoRA weights
                             (if None, will use the latest epoch from models/gemma_lora_weights)
        """
        # Set default attributes to prevent attribute errors
        self.model_path = model_path
        self.local_model = None
        self.local_tokenizer = None
        
        # Initialize the parent class
        super().__init__(similarity_threshold=similarity_threshold, model="dummy")
        
        # Don't load the model now to save memory - we'll load it on first use
        logger.info(f"Initialized Gemma fine-tuned mapper - will error out if model can't be loaded")
        logger.info(f"Gemma fine-tuned model path: {model_path}")
    
    def _ensure_model_available(self):
        """
        Ensure the fine-tuned Gemma model is available. Raises an error if the model cannot be loaded.
        No fallbacks are used - if the fine-tuned model can't be loaded, the function will error out.
        """
        # If model is already loaded, return
        if self.local_model is not None and self.local_tokenizer is not None:
            return
        
        # Find the model path if not provided
        if self.model_path is None:
            # Look for the model in the default location
            model_weights_dir = os.path.join(os.getcwd(), "models", "gemma_lora_weights")
            
            if not os.path.exists(model_weights_dir):
                error_msg = f"Model weights directory not found: {model_weights_dir}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Find the latest epoch directory
            epoch_dirs = [d for d in os.listdir(model_weights_dir) if d.startswith("epoch_")]
            
            if epoch_dirs:
                # Extract epoch numbers and find the latest
                epoch_nums = [int(d.split("_")[1]) for d in epoch_dirs]
                latest_epoch = max(epoch_nums)
                self.model_path = os.path.join(model_weights_dir, f"epoch_{latest_epoch}")
                logger.info(f"Using most recent epoch: {self.model_path}")
            else:
                # If no epoch directories, use the base directory
                self.model_path = model_weights_dir
                logger.info(f"No epoch directories found, using base directory: {self.model_path}")
        
        # Validate model path
        if not os.path.exists(self.model_path):
            error_msg = f"Model path does not exist: {self.model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Import required libraries for loading the model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Check for CUDA availability - require GPU, don't fall back to CPU
        if not torch.cuda.is_available():
            error_msg = "CUDA is not available. GPU is required for Gemma fine-tuned model inference."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        # Set up GPU device
        logger.info("CUDA is available, using GPU for inference")
        device = "cuda"
        # Set default CUDA device
        torch.cuda.set_device(0)
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Clear cache before loading model
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Load the tokenizer
        logger.info(f"Loading tokenizer from fine-tuned model path: {self.model_path}")
        try:
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            logger.warning(f"Error loading tokenizer from fine-tuned model path: {str(e)}")
            logger.info(f"Trying to load tokenizer from base model")
            try:
                self.local_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
            except Exception as e2:
                error_msg = f"Failed to load tokenizer from both fine-tuned and base model: {str(e2)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        # Check if adapter config exists in the model path
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            logger.error(f"Adapter config not found at: {adapter_config_path}")
            logger.info(f"Available files in model path: {os.listdir(self.model_path)}")
            raise FileNotFoundError(f"No adapter_config.json found in {self.model_path}")
        
        # Load the base model - GPU only, no CPU fallback
        base_model_id = "google/gemma-3-4b-it"
        logger.info(f"Loading base model from: {base_model_id}")
        try:
            # Force the model to use CUDA by explicitly setting device_map
            logger.info("Loading model with CUDA optimizations")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better memory efficiency
                device_map={"":0},           # Force model to use CUDA:0
                trust_remote_code=True,      # Trust the model's remote code
                low_cpu_mem_usage=True       # Optimize for low CPU memory usage
            )
        except Exception as e:
            error_msg = f"Failed to load base model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Load the fine-tuned model with LoRA weights
        logger.info(f"Loading fine-tuned Gemma model from: {self.model_path}")
        try:
            self.local_model = PeftModel.from_pretrained(
                base_model, 
                self.model_path,
                is_trainable=False
            )
            
            # Explicitly verify and log the device of the model
            model_device = next(self.local_model.parameters()).device
            logger.info(f"Fine-tuned model loaded on device: {model_device}")
            
            # Verify the model is on CUDA - error out if not
            if str(model_device) != "cuda:0":
                logger.error(f"Model not loaded on CUDA (current device: {model_device})")
                logger.info("Attempting to explicitly move model to CUDA...")
                try:
                    self.local_model = self.local_model.to("cuda:0")
                    model_device = next(self.local_model.parameters()).device
                    if str(model_device) != "cuda:0":
                        raise RuntimeError(f"Failed to move model to CUDA, still on {model_device}")
                    logger.info(f"Successfully moved model to CUDA: {model_device}")
                except Exception as e:
                    error_msg = f"Failed to move model to CUDA: {str(e)}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"Failed to load fine-tuned model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Set to evaluation mode
        self.local_model.eval()
        
        # Log model information
        logger.info(f"Model device: {next(self.local_model.parameters()).device}")
        logger.info(f"Model type: {type(self.local_model).__name__}")
        
        # Log memory usage after loading
        if torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        logger.info(f"Successfully loaded fine-tuned Gemma model")
    
    def get_llm_confidence(self, requirement_desc: str, module_id: str, function_name: str, 
                           function_summary: str, function_content: str) -> float:
        """
        Get the confidence score for a requirement-function match using the fine-tuned Gemma model.
        
        Args:
            requirement_desc: Description of the requirement
            module_id: Module ID of the requirement
            function_name: Name of the function
            function_summary: Summary of the function
            function_content: Content of the function
            
        Returns:
            Confidence score between 0 and 1
        """
        # Ensure the model is available - this will raise an error if the model can't be loaded
        self._ensure_model_available()
        
        # Use the fine-tuned Gemma model
        try:
            # Log that we're using fine-tuned Gemma
            logger.info(f"Using fine-tuned Gemma model for '{function_name}' with requirement: '{requirement_desc[:30]}...'")
            
            # Create the prompt in the same format used during fine-tuning
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
            
            # Prepare for inference with proper GPU memory management
            # Clear cache to free up memory
            torch.cuda.empty_cache()
            logger.info(f"GPU memory before inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Get the model's device and verify it's on CUDA
            device = next(self.local_model.parameters()).device
            logger.info(f"Model is on device: {device}")
            
            # Ensure model is on CUDA for inference
            if str(device) != "cuda:0":
                error_msg = f"Model must be on CUDA for inference, but found device: {device}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Tokenize the input
            try:
                inputs = self.local_tokenizer(prompt, return_tensors="pt")
                logger.info(f"Input tokens: {len(inputs['input_ids'][0])}")
                
                # Move inputs to the same device as the model
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                logger.error(f"Error preparing inputs: {str(e)}")
                raise RuntimeError(f"Failed to prepare inputs: {str(e)}")
            
            # Generate the output with error handling
            try:
                with torch.no_grad():
                    # Use optimized generation settings
                    generation_config = {
                        "max_new_tokens": 10,      # We only need a short response with the score
                        "temperature": 0.1,       # Low temperature for more deterministic output
                        "do_sample": False,       # Don't sample for deterministic output
                        "num_beams": 1            # Use greedy decoding to save memory
                    }
                    
                    # Remove parameters not supported by the model
                    if "use_cache" in generation_config and device == "cuda":
                        logger.info("Removing 'use_cache' parameter as it's not supported by Gemma3")
                        del generation_config["use_cache"]
                    
                    # Generate output
                    logger.info(f"Generating with config: {generation_config}")
                    outputs = self.local_model.generate(**inputs, **generation_config)
                    
                # Log memory usage after generation
                if torch.cuda.is_available():
                    logger.info(f"GPU memory after generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(f"Failed during generation: {str(e)}")
            
            # Decode the output with error handling
            try:
                output_text = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Generated output: {output_text[-50:]}")
            except Exception as e:
                logger.error(f"Error decoding output: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(f"Failed to decode output: {str(e)}")
            
            # Extract the rating from the output
            # First, remove the prompt from the output
            response = output_text[len(prompt):].strip()
            
            # Parse rating
            number_match = re.search(r'\b(\d+)\b', response)
            if number_match:
                rating = int(number_match.group(1))
                # Clamp between 0 and 100
                rating = min(100, max(0, rating))
                # Normalize to 0-1 range
                confidence = rating / 100.0
            else:
                # Fallback if no number found
                logger.warning(f"Could not find a rating number in fine-tuned Gemma response: '{response}'")
                confidence = 0.2  # Default modest confidence
            
            logger.info(f"Fine-tuned Gemma confidence for matching '{function_name}' to requirement: {confidence:.2f}")
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting fine-tuned Gemma confidence: {str(e)}")
            
            # Only try Ollama if fallback is allowed
            if not self.strict_mode:
                logger.warning("Attempting Ollama fallback after fine-tuned Gemma error")
                try:
                    return super().get_llm_confidence(
                        requirement_desc, module_id, function_name, function_summary, function_content
                    )
                except Exception as fallback_error:
                    logger.error(f"Error using Ollama fallback: {str(fallback_error)}")
            
            # Return a default low confidence if no fallback or fallback failed
            return 0.1
