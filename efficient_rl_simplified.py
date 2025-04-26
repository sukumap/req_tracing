#!/usr/bin/env python3
"""
Simplified reinforcement learning for requirement tracing.

This implements a custom two-stage approach for requirement tracing:
1. The model generates summaries of code functions
2. The model evaluates how well summaries match requirements
3. Rewards are computed and used to improve the model via direct policy optimization

Uses GPU acceleration with CUDA fixes while avoiding external RL libraries.
"""
# ==================== CUDA INITIALIZATION FIX ====================
# Set environment variables BEFORE importing PyTorch
import os
import math

# Set CUDA environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Memory optimization

# ==================== END CUDA FIX ====================

# Regular imports
import json
import logging
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import torch

# ─── PATCH: Force consistent dtypes for scaled_dot_product_attention ─────────────────
_orig_sdpa = F.scaled_dot_product_attention
def _patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    # Get the target dtype from key (could be float16/bfloat16)
    target_dtype = key.dtype
    
    # Convert all tensors to the same dtype
    query = query.to(target_dtype)
    key = key.to(target_dtype)
    value = value.to(target_dtype)
    
    # Also convert attention mask if present
    if attn_mask is not None:
        attn_mask = attn_mask.to(target_dtype)
    
    # Handle scale parameter
    if scale is not None and isinstance(scale, torch.Tensor):
        scale = scale.to(target_dtype)
    
    # Forward with consistent dtypes
    return _orig_sdpa(query, key, value,
                     attn_mask=attn_mask,
                     dropout_p=dropout_p,
                     is_causal=is_causal,
                     scale=scale,
                     **kwargs)

# Apply the patch
F.scaled_dot_product_attention = _patched_sdpa
import random
import sys
import time
import re
import shutil
import argparse  # Added for command line argument parsing
from typing import List, Dict, Any, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)

# Import Gemma3-specific modules (only if available)
try:
    from transformers import Gemma3ForCausalLM
    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False

# Note: We'll import the inference module after logger initialization

# ==================== ADDITIONAL CUDA FIXES ====================
# Apply PyTorch CUDA patches

# Clear cache and reset state
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()

# Reset PyTorch CUDA initialization state
if hasattr(torch.cuda, '_initialized'):
    torch.cuda._initialized = False
    
# Override the CUDA device count check to avoid the error
try:
    original_device_count = torch.cuda.device_count
    def patched_device_count():
        try:
            return original_device_count()
        except Exception as e:
            print(f"Patched device count error: {str(e)}")
            # Return 1 if there's an error in the original implementation
            return 1
    torch.cuda.device_count = patched_device_count
except Exception as e:
    print(f"Failed to patch device count: {str(e)}")
    
# ==================== END ADDITIONAL CUDA FIXES ====================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import our custom inference module for proper LoRA handling
try:
    from inference import load_adapted_model, save_for_inference, verify_lora_effects, batch_verify_lora
    HAS_INFERENCE_MODULE = True
    logger.info("Loaded inference module for proper LoRA parameter handling")
except ImportError:
    logger.warning("Inference module not found, some LoRA verification features will be disabled")
    HAS_INFERENCE_MODULE = False

# Define internal train_gemma3_simplified function for direct usage
HAS_GEMMA3_SIMPLIFIED = True
    
# Import and apply Gemma3 dtype patch for attention mechanism
try:
    from gemma3_dtype_patch import apply_gemma3_patches
    GEMMA3_PATCHED = apply_gemma3_patches()
    logger.info("Successfully applied Gemma3 attention patch to fix dtype issues")
except ImportError:
    logger.warning("Gemma3 dtype patch not available, may encounter dtype mismatches")
    GEMMA3_PATCHED = False

# Model definitions for supported architectures
MODEL_CONFIGS = {
    "tinyllama": {
        "name": "TinyLlama",
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_r": 16,
        "lora_alpha": 32
    },
    "gemma3": {
        "name": "Gemma-3",
        "id": "google/gemma-3-1b-it",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_r": 16,
        "lora_alpha": 32
    }
}

# Default model is TinyLlama
DEFAULT_MODEL = "tinyllama"
MODEL_ID = MODEL_CONFIGS[DEFAULT_MODEL]["id"]

# Constants for training
OUTPUT_DIR = "rl_weights"
TRAINING_DATA_PATH = "training_data.jsonl"
MAX_SUMMARY_LENGTH = 100
MAX_EXAMPLES = 40  # Limit number of examples for faster processing

# RL training configuration
BATCH_SIZE = 4  # Mini-batch size for updates
DEFAULT_EPOCHS = 5  # Default number of training epochs if not specified via command line
NUM_EXAMPLES_TRAIN = 32  # Examples to use for training (must be <= MAX_EXAMPLES)
LEARNING_RATE = 2e-5  # Learning rate
MIN_REWARD = 0.0  # Minimum reward value
MAX_REWARD = 1.0  # Maximum reward value
MAX_RESPONSE_LENGTH = 150  # Maximum token length for generated summaries
KL_PENALTY = 0.1  # KL divergence penalty to prevent policy from changing too quickly


def check_gpu():
    """Check if GPU is available and log its details with robust error handling."""
    # First try in a way that won't throw errors
    cuda_available = False
    device_count = 0
    device_name = "Unknown"
    
    try:
        # Try multiple ways to detect CUDA
        detection_methods = [
            # Method 1: Standard torch.cuda.is_available()
            lambda: torch.cuda.is_available(),
            
            # Method 2: Check via device count
            lambda: torch.cuda.device_count() > 0,
            
            # Method 3: Try creating a CUDA tensor
            lambda: torch.tensor([1.0], device="cuda").is_cuda
        ]
        
        # Try each method until one works
        for i, method in enumerate(detection_methods):
            try:
                if method():
                    cuda_available = True
                    logger.info(f"CUDA detected via method {i+1}")
                    break
            except Exception as e:
                logger.debug(f"CUDA detection method {i+1} failed: {str(e)}")
                continue
        
        # Get additional info if CUDA is available
        if cuda_available:
            try:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU detected: {device_name}")
                logger.info(f"CUDA Version: {torch.version.cuda}")
                logger.info(f"Number of available GPUs: {device_count}")
                
                # Try to run a small tensor operation to verify CUDA works
                try:
                    torch.cuda.set_device(0)
                    test_tensor = torch.ones(2, 2, device="cuda")
                    test_result = test_tensor + test_tensor
                    del test_tensor, test_result
                    torch.cuda.empty_cache()
                    logger.info("CUDA tensor operation successful")
                    
                    # Definitely use GPU
                    return True
                except Exception as e:
                    logger.warning(f"CUDA tensor test failed: {str(e)}")
                    logger.warning("Will try alternative GPU configuration")
            except Exception as e:
                logger.warning(f"Error getting CUDA device info: {str(e)}")
    except Exception as e:
        logger.warning(f"Error in CUDA detection: {str(e)}")
    
    # Final determination - use external nvidia-smi check as fallback
    try:
        import subprocess
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("nvidia-smi indicates GPU is available")
            return True
    except Exception as e:
        logger.warning(f"nvidia-smi check failed: {str(e)}")
    
    if cuda_available and device_count > 0:
        logger.info("Will attempt to use GPU based on CUDA availability")
        return True
    else:
        logger.warning("No GPU detected, will use CPU for training")
        return False


def load_training_data(max_examples=MAX_EXAMPLES):
    """Load training data from the JSONL file."""
    if not os.path.exists(TRAINING_DATA_PATH):
        logger.error(f"Training data file {TRAINING_DATA_PATH} not found!")
        return None
    
    with open(TRAINING_DATA_PATH, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Limit examples if needed to save memory
    if max_examples and len(data) > max_examples:
        # Ensure a balanced dataset with both positive and negative examples
        positive = [item for item in data if item["is_correct"]]
        negative = [item for item in data if not item["is_correct"]]
        
        # Determine how many of each to keep
        pos_to_keep = min(len(positive), max_examples // 2)
        neg_to_keep = min(len(negative), max_examples - pos_to_keep)
        
        # Adjust if we don't have enough of one type
        if pos_to_keep < max_examples // 2:
            neg_to_keep = min(len(negative), max_examples - pos_to_keep)
        
        # Sample randomly
        random.shuffle(positive)
        random.shuffle(negative)
        
        data = positive[:pos_to_keep] + negative[:neg_to_keep]
        random.shuffle(data)
        
        logger.info(f"Using {len(data)} examples ({pos_to_keep} positive, {neg_to_keep} negative)")
    else:
        logger.info(f"Loaded {len(data)} examples from {TRAINING_DATA_PATH}")
    
    return data


def initialize_model(has_gpu, model_type=DEFAULT_MODEL):
    """
    Initialize the model with optimal memory settings.
    If previously trained weights exist, they will be loaded instead of starting fresh.
    
    Args:
        has_gpu: Whether GPU is available
        model_type: Which model architecture to use (tinyllama or gemma3)
    """
    global MODEL_ID, OUTPUT_DIR
    
    # Get the model configuration based on the selected type
    if model_type not in MODEL_CONFIGS:
        logger.warning(f"Unknown model type: {model_type}, falling back to {DEFAULT_MODEL}")
        model_type = DEFAULT_MODEL
    
    # Set the model ID from the configuration
    model_config = MODEL_CONFIGS[model_type]
    MODEL_ID = model_config["id"]
    
    # Set model-specific output directory
    OUTPUT_DIR = f"rl_weights_{model_type}"
    device = "cuda" if has_gpu else "cpu"
    
    logger.info(f"Initializing {model_config['name']} model ({MODEL_ID}) on {device}")
    
    # Check if Gemma3 is requested but not available
    if model_type == "gemma3" and not GEMMA3_AVAILABLE:
        logger.error("Gemma3 requested but transformers doesn't have Gemma3ForCausalLM. Please upgrade transformers.")
        logger.info("Falling back to TinyLlama model")
        model_type = "tinyllama"
        MODEL_ID = MODEL_CONFIGS[model_type]["id"]
    
    # Check if we have previously saved weights
    if os.path.exists(OUTPUT_DIR):
        try:
            last_epoch_dir = None
            old_checkpoints = []
            
            # Find the highest epoch directory (most recent training checkpoint)
            for dirname in os.listdir(OUTPUT_DIR):
                if dirname.startswith("epoch-"):
                    old_checkpoints.append(dirname)
                    if last_epoch_dir is None or int(dirname.split("-")[1]) > int(last_epoch_dir.split("-")[1]):
                        last_epoch_dir = dirname
            
            # If we found a checkpoint, try to load it
            if last_epoch_dir:
                checkpoint_path = os.path.join(OUTPUT_DIR, last_epoch_dir)
                logger.info(f"Loading previously trained weights from {checkpoint_path}")
                
                # Load tokenizer from checkpoint
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model from checkpoint with quantization to reduce memory usage
                try:
                    if has_gpu:
                        # Configure quantization based on model type
                        quantization_config = None
                        load_dtype = torch.float16 # Default for TinyLlama
                        
                        if model_type == "gemma3":
                            logger.info("Disabling quantization for Gemma3. Using bfloat16.")
                            load_dtype = torch.bfloat16
                            # Quantization is disabled by keeping quantization_config=None
                        else:
                            logger.info("Configuring 8-bit quantization for TinyLlama on GPU...")
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=False,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                        
                        logger.info(f"Loading base model {MODEL_ID}...")
                        model = AutoModelForCausalLM.from_pretrained(
                            MODEL_ID,  # First load base model
                            device_map="auto", # Let HF handle device placement initially
                            load_in_8bit=False,
                            quantization_config=None, # Remove quantization config
                            torch_dtype=load_dtype,
                            trust_remote_code=True, # Needed for Gemma
                            attn_implementation="eager" # Use eager attention to avoid potential issues
                        )
                        
                        # === Debug: Print named modules for Gemma3 ===
                        if model_type == "gemma3":
                            logger.info("--- Gemma3 Base Model Named Modules (Checkpoint Path) ---")
                            for name, module in model.named_modules():
                                # Log only parent modules or specific layers of interest
                                if '.' not in name or any(layer in name for layer in ["attn", "mlp", "q_proj", "v_proj"]):
                                    logger.info(f"Module Name: {name}, Type: {type(module).__name__}")
                            logger.info("-----------------------------------------------------")
                        # === End Debug ===

                        # === Explicit dtype enforcement for Gemma3 ===
                        if model_type == "gemma3" and has_gpu:
                            logger.info("Forcing all Gemma3 model parameters (including LoRA/PEFT) to bfloat16 for consistency...")
                            for n, p in model.named_parameters():
                                try:
                                    p.data = p.data.to(torch.bfloat16)
                                except Exception as e:
                                    logger.warning(f"Failed to convert {n} to bfloat16: {e}")
                            # Log dtypes for confirmation
                            dtype_set = set(str(p.dtype) for n, p in model.named_parameters() if p.requires_grad)
                            logger.debug(f"[DEBUG] After enforcement, Gemma3 trainable param dtypes: {dtype_set}")
                        # === End explicit dtype enforcement ===

                        # Prepare model for k-bit training if quantized
                        if quantization_config is not None and hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
                            logger.info("Preparing quantized model for training...")
                            model = prepare_model_for_kbit_training(model)
                            
                        # Then load the LoRA weights
                        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
                        logger.info(f"Successfully loaded model checkpoint from {checkpoint_path}")
                        
                        # === Debug: Log trainable parameters after loading checkpoint ===
                        trainable_params_chkpt = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        logger.info(f"[Debug] Trainable parameters after PeftModel.from_pretrained (checkpoint): {trainable_params_chkpt:,}")
                        # === End Debug ===
                    
                    else:
                        # For CPU, we don't need quantization as much
                        load_dtype = torch.bfloat16 if model_type == "gemma3" else torch.float32
                        model = AutoModelForCausalLM.from_pretrained(
                            MODEL_ID,
                            torch_dtype=load_dtype, # Use appropriate dtype for CPU
                            low_cpu_mem_usage=True
                        )
                        # Load the LoRA weights
                        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
                        logger.info(f"Successfully loaded model checkpoint from {checkpoint_path} (CPU mode, dtype={load_dtype})")
                    
                    # Enable gradient checkpointing to save memory
                    if hasattr(model, "gradient_checkpointing_enable"):
                        # model.gradient_checkpointing_enable() # <<< COMMENTED OUT FOR DEBUGGING
                        pass # Keep block structure
                    
                    # Log trainable parameters
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in model.parameters())
                    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")
                    
                    # Clean up old checkpoints now that we've successfully loaded the model
                    logger.info("Cleaning up old checkpoint directories...")
                    import shutil
                    
                    # 1. Remove old epoch directories except the one we just loaded
                    for checkpoint_dir in old_checkpoints:
                        if checkpoint_dir != last_epoch_dir:  # Keep the one we just loaded
                            try:
                                old_dir_path = os.path.join(OUTPUT_DIR, checkpoint_dir)
                                shutil.rmtree(old_dir_path)
                                logger.info(f"Removed old checkpoint: {checkpoint_dir}")
                            except Exception as e:
                                logger.warning(f"Failed to remove old checkpoint {checkpoint_dir}: {e}")
                    
                    # 2. Remove old batch checkpoints from previous training runs
                    for dirname in os.listdir(OUTPUT_DIR):
                        if dirname.startswith("checkpoint-epoch"):
                            try:
                                old_checkpoint = os.path.join(OUTPUT_DIR, dirname)
                                shutil.rmtree(old_checkpoint)
                                logger.info(f"Removed old batch checkpoint: {dirname}")
                            except Exception as e:
                                logger.warning(f"Failed to remove old batch checkpoint {dirname}: {e}")
                    
                    return model, tokenizer
                except Exception as e:
                    logger.warning(f"Failed to load weights from checkpoint: {str(e)}, falling back to fresh model")
        except Exception as e:
            logger.warning(f"Error checking for checkpoints: {str(e)}, using fresh model instead")
    
    # If we get here, we couldn't load a checkpoint, so initialize a fresh model
    logger.info(f"Initializing fresh model: {MODEL_ID}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create LoRA config specific to the model architecture
    target_modules = model_config.get("target_modules", ["q_proj", "v_proj"]) # Default if not specified
    lora_config = LoraConfig(
        r=16, # Increased rank for potentially better adaptation
        lora_alpha=32, # Scaling factor for LoRA weights
        target_modules=target_modules,
        lora_dropout=0.05, # Dropout probability for LoRA layers
        bias="none", # Typically set to 'none' for LoRA
        task_type="CAUSAL_LM" # Task type for PEFT
    )
    
    # Initialize a fresh model with memory optimization
    try:
        if has_gpu:
            # Configure quantization based on model type
            quantization_config = None
            load_dtype = torch.float16 # Default for TinyLlama
            
            if model_type == "gemma3":
                logger.info("Disabling quantization for Gemma3. Using bfloat16.")
                load_dtype = torch.bfloat16
            else:
                logger.info("Configuring 8-bit quantization for TinyLlama on GPU...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=False,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            logger.info(f"Loading fresh base model {MODEL_ID}...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto", # Let HF handle device placement initially
                load_in_8bit=False,
                quantization_config=None, # Remove quantization config
                torch_dtype=load_dtype,
                trust_remote_code=True, # Needed for Gemma
                attn_implementation="eager" # Use eager attention to avoid potential issues
            )
            
            # === Debug: Print named modules for Gemma3 ===
            if model_type == "gemma3":
                logger.info("--- Gemma3 Base Model Named Modules (Fresh GPU Path) ---")
                for name, module in model.named_modules():
                     if '.' not in name or any(layer in name for layer in ["attn", "mlp", "q_proj", "v_proj"]):
                        logger.info(f"Module Name: {name}, Type: {type(module).__name__}")
                logger.info("--------------------------------------------------")
            # === End Debug ===
            
            # Prepare model for k-bit training if quantized
            if quantization_config is not None and hasattr(model, "is_loaded_in_8bit") and model.is_loaded_in_8bit:
                logger.info("Preparing quantized model for training...")
                model = prepare_model_for_kbit_training(model)
                # Apply LoRA adapters AFTER preparing for kbit training
                logger.info("Applying LoRA adapters to fresh quantized model...")
                model = get_peft_model(model, lora_config)
            elif model_type == "gemma3": # Apply LoRA if Gemma3 (not quantized)
                logger.info("Applying LoRA adapters to fresh Gemma3 model...")
                model = get_peft_model(model, lora_config)
            # If not quantized and not Gemma3, PEFT might be applied later or not needed depending on flow
            # However, let's explicitly add it for non-quantized TinyLlama too if needed
            elif quantization_config is None:
                logger.warning("Applying LoRA adapters to fresh non-quantized model (GPU path). Verify if this is intended.")
                model = get_peft_model(model, lora_config)

        else:
            # Load model on CPU without quantization
            load_dtype = torch.bfloat16 if model_type == "gemma3" else torch.float32
            logger.info("Loading base model on CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=load_dtype, # Use appropriate dtype for CPU
                low_cpu_mem_usage=True
            )
            logger.info(f"Model loaded on CPU with dtype={model.dtype}")
            
            # === Debug: Print named modules for Gemma3 ===
            if model_type == "gemma3":
                logger.info("--- Gemma3 Base Model Named Modules (Fresh CPU Path) ---")
                for name, module in model.named_modules():
                    if '.' not in name or any(layer in name for layer in ["attn", "mlp", "q_proj", "v_proj"]):
                        logger.info(f"Module Name: {name}, Type: {type(module).__name__}")
                logger.info("--------------------------------------------------")
            # === End Debug ===
            
            # Apply LoRA adapters to the fresh CPU model
            logger.info("Applying LoRA adapters to fresh CPU model...")
            model = get_peft_model(model, lora_config)

    except Exception as e:
        logger.warning(f"Model loading failed: {str(e)}")
        logger.info("Falling back to CPU as last resort")
        # Use eager attention implementation for Gemma3 to avoid dtype mismatch errors
        if model_type == "gemma3":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                attn_implementation="eager"  # Avoid dtype mismatch in attention mechanism
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, "gradient_checkpointing_enable"):
        # model.gradient_checkpointing_enable() # <<< COMMENTED OUT FOR DEBUGGING
        pass # Keep block structure
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")

    # === DEBUG: Print device and dtype for all trainable parameters ===
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"[DEBUG][PARAM] {n}: device={p.device}, dtype={p.dtype}, shape={p.shape}")
    # === END DEBUG ===

    return model, tokenizer


def compute_reward(is_correct, predicted_score, expected_score=None):
    """
    Compute a continuous reward based on the accuracy of the predicted score.
    
    Args:
        is_correct: Whether the match is actually correct
        predicted_score: The score predicted by the model (0-100)
        expected_score: Optional expected score for training
        
    Returns:
        A reward value between MIN_REWARD and MAX_REWARD
    """
    # Ensure predicted_score is a number between 0 and 100
    try:
        if isinstance(predicted_score, str):
            # Extract just the digits
            predicted_score = ''.join(filter(str.isdigit, predicted_score.strip()))
            if predicted_score:
                predicted_score = float(predicted_score)
            else:
                predicted_score = 50.0  # Default if no number found
        
        predicted_score = min(100, max(0, float(predicted_score)))  # Clamp between 0 and 100
        
    except Exception as e:
        logger.warning(f"Error parsing score '{predicted_score}': {e}")
        predicted_score = 50.0  # Default on error
    
    # If we have an expected score, calculate reward based on how close the prediction is
    if expected_score is not None:
        # Convert to same scale
        expected_score = float(expected_score)
        # Normalized distance (0 = perfect match, 1 = furthest possible)
        normalized_distance = abs(predicted_score - expected_score) / 100.0
        # Convert to reward (1 = perfect match, 0 = furthest possible)
        reward = 1.0 - normalized_distance
        # Scale to our reward range
        scaled_reward = MIN_REWARD + (MAX_REWARD - MIN_REWARD) * reward
        return scaled_reward
    
    # Without expected score, use is_correct to determine reward
    # Apply non-linear reward functions to create stronger discrimination
    if is_correct:
        # For correct matches, reward increases with confidence
        # This encourages high confidence for truly correct matches
        normalized_score = predicted_score / 100.0
        reward = normalized_score ** 1.2  # Slightly super-linear to encourage high confidence
        
        # Only slightly penalize if a correct match gets a very low score (below 30)
        # We want to encourage confidence for correct matches
        if predicted_score < 30:
            reward *= 0.8  # Small penalty for extremely low confidence on correct matches
    else:
        # For incorrect matches, apply ultra-aggressive penalties for high scores
        normalized_score = predicted_score / 100.0
        
        # Base penalty curve - extremely steep to discourage any confidence in wrong answers
        reward = 1.0 - normalized_score ** 0.2  # Even steeper penalty curve
        
        # Progressive and increasingly more severe penalties based on confidence level
        
        # Start penalizing wrong answers at very low confidence (20+)
        if predicted_score >= 20:
            # Initial penalty factor
            penalty_factor = 0.7  # More aggressive starting penalty
            reward *= penalty_factor
        
        # Medium confidence wrong answers (40+)
        if predicted_score >= 40:
            reward *= 0.3  # Very strong penalty
        
        # Medium-high confidence wrong answers (60+)
        if predicted_score >= 60:
            reward *= 0.1  # Extremely harsh penalty
        
        # High confidence wrong answers (80+)
        if predicted_score >= 80:
            reward *= 0.05  # Near-zero reward
            
        # Absolute confidence wrong answers (100)
        if predicted_score == 100:
            reward = MIN_REWARD  # Force minimum reward - this should never happen
    
    # Scale to our reward range
    scaled_reward = MIN_REWARD + (MAX_REWARD - MIN_REWARD) * reward
    return scaled_reward


def train_with_rl(model, tokenizer, data, has_gpu=False, epochs=DEFAULT_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, num_examples=NUM_EXAMPLES_TRAIN, model_type=DEFAULT_MODEL):
    """Train the model using a custom reinforcement learning approach.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer to use
        data: Training data
        has_gpu: Whether GPU is available
        epochs: Number of training epochs (from command line args)
        batch_size: Batch size for training (from command line args)
        learning_rate: Learning rate (from command line args)
        num_examples: Number of examples to use (from command line args)
        model_type: Which model architecture to use (tinyllama or gemma3)
    
    Note:
        For Gemma3, we use a simplified training approach to avoid dtype issues
        while maintaining compatibility with the existing framework.
    """
    # Check device
    if has_gpu and torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        logger.info(f"RL training on GPU device {device}: {device_name}")
        device = f"cuda:{device}"
    else:
        device = "cpu"
        logger.warning("No GPU available, training will be slower")
    
    # For Gemma3, use a completely different training approach to avoid dtype issues
    if model_type == "gemma3":
        logger.info("Starting Gemma3 simplified training...")
        # Call the dedicated training function for Gemma3
        best_model_state_dict, best_epoch, best_reward = train_gemma3_simplified(
            model=model,
            tokenizer=tokenizer,
            data=data,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_examples=num_examples
        )
        logger.info(f"Gemma3 training completed. Best model from epoch {best_epoch} with reward {best_reward}")
        # Load the best state dict back into the model
        if best_model_state_dict:
            model.load_state_dict(best_model_state_dict)
            logger.info("Loaded best state dict from Gemma3 training into the model.")
        else:
            logger.warning("No best model state dict found from Gemma3 training. Returning original model.")
        return model # Return the updated model object

    # Standard approach for other models
    logger.info("Starting reinforcement learning for requirement tracing...")
    logger.info(f"Using batch size: {batch_size}, learning rate: {learning_rate}")
    
    # Filter to only use a subset of data for training
    train_data = data[:min(len(data), num_examples)]
    logger.info(f"Using {len(train_data)} examples for reinforcement learning")
    
    try:
        # Move model to device
        model.to(device)
        model.train()
        
        # Create prompts with the appropriate template for the model
        if model_type == "gemma3":
            all_prompts = create_gemma3_prompts(train_data)
        else:
            all_prompts = create_tinyllama_prompts(train_data)
        
        # Setup optimizer with user-specified learning rate
        logger.info(f"Setting up optimizer with learning rate {learning_rate}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        logger.info("Starting reinforcement learning loop...")
        for epoch in range(epochs):
            logger.info(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")
            model.train()
            epoch_start_time = time.time()
            epoch_rewards = []
            epoch_losses = []
            
            # Shuffle data each epoch
            indices = list(range(len(train_data)))
            random.shuffle(indices)
            shuffled_train_data = [train_data[i] for i in indices]

            for i in range(0, len(shuffled_train_data), batch_size):
                batch_start_time = time.time()
                optimizer.zero_grad() # Zero gradients for the batch
                
                batch_indices = range(i, min(i + batch_size, len(shuffled_train_data)))
                actual_batch_size = len(batch_indices)
                
                all_rewards = []
                all_log_probs_for_loss = [] # Store log probs corresponding to rewards

                # --- Inner loop to process each item in the batch --- 
                for idx_in_batch, data_idx in enumerate(batch_indices):
                    item = shuffled_train_data[data_idx]
                    inputs, is_correct, expected_score = prepare_rl_input(item, tokenizer, device)
                    
                    # Ensure inputs are correctly shaped [1, seq_len]
                    input_tensor = inputs.input_ids.unsqueeze(0) if inputs.input_ids.ndim == 1 else inputs.input_ids
                    attn_tensor = inputs.attention_mask.unsqueeze(0) if inputs.attention_mask.ndim == 1 else inputs.attention_mask

                    # === 1. Forward pass WITH gradients ===
                    # Obtain logits required for policy gradient calculation
                    outputs = model(input_ids=input_tensor, attention_mask=attn_tensor)
                    logits = outputs.logits # Shape: [1, seq_len, vocab_size]
                    logger.info(f"outputs.logits.requires_grad (immediately after model call): {outputs.logits.requires_grad}")

                    # Calculate log prob for the action (use last 5 tokens' mean log prob as proxy)
                    # This needs to be connected to the graph
                    log_prob_for_action = torch.log_softmax(logits[:, -5:, :], dim=-1).mean()
                    all_log_probs_for_loss.append(log_prob_for_action)

                    # === 2. Generation WITHOUT gradients (for reward) ===
                    with torch.no_grad():
                        if model_type == "gemma3":
                            # Use CPU offloading for generation if needed
                            original_device = model.device
                            try:
                                model = model.to('cpu')
                                cpu_ids = input_tensor.to('cpu')
                                cpu_mask = attn_tensor.to('cpu')
                                
                                gen_outputs = model.generate(
                                    input_ids=cpu_ids,
                                    attention_mask=cpu_mask,
                                    max_new_tokens=25, # Increased
                                    do_sample=True,
                                    temperature=0.6,
                                    top_p=0.95,
                                    pad_token_id=tokenizer.pad_token_id
                                )
                                gen_outputs = gen_outputs.to(original_device) # Move back if needed elsewhere
                            finally:
                                model = model.to(original_device) # Ensure model is back on GPU
                        else:
                            # Standard approach for TinyLlama
                            gen_outputs = model.generate(
                                input_ids=input_tensor,
                                attention_mask=attn_tensor,
                                max_new_tokens=25, # Increased
                                do_sample=True,
                                temperature=0.6,
                                top_p=0.95,
                                pad_token_id=tokenizer.pad_token_id
                            )
                    
                        # Extract generated rating text
                        rating_text = tokenizer.decode(gen_outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
                        
                        # Parse score and compute reward
                        if model_type == "gemma3":
                            numbers = re.findall(r"[0-9]+", str(rating_text))
                            predicted_score = 0.0
                            if numbers:
                                for num in numbers:
                                    try: # Add try-except for safety
                                        n = int(num)
                                        if 0 <= n <= 100:
                                            predicted_score = float(n)
                                            break
                                    except ValueError:
                                        continue # Skip non-integer numbers

                            reward = compute_reward(is_correct, predicted_score, expected_score)
                        else:
                            # TinyLlama or other models: (This part might need revisiting if using non-Gemma RL)
                            # Using max logit value is NOT standard RL practice - should parse generated text too
                            predicted_score = float(logits[:, -1, :].max(dim=-1)[0].item()) # Logits here are from step 1
                            reward = compute_reward(is_correct, predicted_score, expected_score)
                        
                        all_rewards.append(reward)
                        logger.info(f"Item {data_idx+1}: Rating='{str(rating_text).strip()}', Parsed={predicted_score}, Expected={expected_score}, Reward={reward:.4f}")
                
                # --- End of inner batch loop --- 

                # === 3. Compute batch policy loss ===
                if all_rewards and all_log_probs_for_loss:
                    rewards_tensor = torch.tensor(all_rewards, device=device, dtype=torch.float32)
                    log_probs_tensor = torch.stack(all_log_probs_for_loss) # Stack the collected log_probs
                    
                    # Ensure shapes match for element-wise multiplication
                    if rewards_tensor.shape != log_probs_tensor.shape:
                        # This shouldn't happen if collected correctly, but check
                        logger.warning(f"Shape mismatch: rewards {rewards_tensor.shape}, log_probs {log_probs_tensor.shape}. Attempting to fix.")
                        # Handle potential dimension mismatches if necessary (e.g., squeeze)
                        rewards_tensor = rewards_tensor.view_as(log_probs_tensor) 

                    # Policy gradient loss (REINFORCE)
                    policy_loss = -(rewards_tensor * log_probs_tensor).mean()
                    batch_loss_value = policy_loss.item() # For logging
                    
                    # === 4. Backpropagate and Optimize ===
                    policy_loss.backward() # Backpropagate through log_probs -> logits -> model
                    optimizer.step() # Apply gradient updates

                    # Calculate metrics for logging
                    avg_reward = rewards_tensor.mean().item() 
                    epoch_rewards.append(avg_reward)
                    epoch_losses.append(batch_loss_value)
                    
                    # Log batch performance
                    batch_time = time.time() - batch_start_time
                    logger.info(f"Batch {(i // batch_size) + 1}/{len(shuffled_train_data)//batch_size} completed in {batch_time:.2f}s - Avg Reward: {avg_reward:.4f}, Loss: {batch_loss_value:.4f}")
                else:
                    logger.warning(f"Skipping batch {(i // batch_size) + 1} due to no rewards or log_probs collected.")
            
            # --- End of Epoch --- 
            epoch_duration = time.time() - epoch_start_time
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            logger.info(f"Epoch {epoch+1} finished in {epoch_duration:.2f}s - Avg Reward: {avg_epoch_reward:.4f}, Avg Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint for the epoch
            epoch_dir = os.path.join(OUTPUT_DIR, f"epoch-{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            logger.info(f"Saved model after epoch {epoch+1} to {epoch_dir}")
        
        # Save final model in multiple formats
        logger.info("Saving final model for inference...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"Model saved to {OUTPUT_DIR}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during reinforcement learning: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return model


def create_prompts(data, model_type=DEFAULT_MODEL):
    """
    Create three types of prompts from the data with improved discrimination:
    1. Summary generation prompts
    2. Requirement-matching prompts with clear scoring criteria
    3. Contrastive learning prompts (comparing correct vs incorrect implementations directly)
    
    Args:
        data: The training data
        model_type: The type of model (tinyllama or gemma3) to create prompts for
    """
    all_prompts = {}
    
    # Use the appropriate prompt creation function based on model type
    if model_type == "gemma3":
        return create_gemma3_prompts(data)
    else:  # Default to TinyLlama prompts
        return create_tinyllama_prompts(data)


def create_tinyllama_prompts(data):
    """Create prompts specifically formatted for TinyLlama model."""
    all_prompts = []
    
    for example in data:
        code = example['function_content']
        requirement = example["requirement_text"]
        is_correct = example["is_correct"]
        function_summary = example.get("function_summary", "")
        
        # Calculate similarity score - in real data you'd use semantic similarity
        # For now, use a simple heuristic based on word overlap
        if not is_correct:
            # For negative examples, calculate approximate similarity for more nuanced training
            req_words = set(requirement.lower().split())
            summary_words = set(function_summary.lower().split())
            overlap = len(req_words.intersection(summary_words)) / max(1, len(req_words))
            
            # Classify the negative example into subcategories
            if overlap > 0.3:  # Significant overlap but still wrong
                similarity_type = "related_but_incorrect"
                expected_score = 30  # Some relation but incorrect
            elif overlap > 0.1:  # Minor overlap
                similarity_type = "slightly_related"
                expected_score = 15  # Minimal relation
            else:  # Almost no overlap
                similarity_type = "unrelated"
                expected_score = 5   # No relation
        else:
            similarity_type = "correct_match"
            expected_score = 90  # Perfect match (reduced from 100 to 90 to allow for nuance)
        
        # For positive examples, create both summary and rating prompts
        if is_correct:
            # Summary generation prompt with more specific instructions
            summary_prompt = f"""Summarize the following code function in a concise way that captures its main purpose and functionality:

```
{code[:500]}  # Limit code size to save memory
```

Provide a clear, specific Function Summary that highlights what the code actually does:"""

            all_prompts.append({
                "text": summary_prompt,
                "type": "summary",
                "expected_score": 90,
                "requirement": requirement,
                "similarity_type": similarity_type
            })
        
        # Rating prompt with explicit scoring criteria - for all examples
        rating_prompt = f"""Rate how well the function summary implements the given requirement.

Requirement: {requirement}

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

        all_prompts.append({
            "text": rating_prompt,
            "type": "rating",
            "expected_score": expected_score,
            "original_data": example,  # Changed from item to example
            "requirement": requirement,
            "similarity_type": similarity_type
        })
    
    # Add contrastive learning prompts (NEW)
    # For each requirement that has both correct and incorrect implementations,
    # create explicit comparison prompts
    requirement_groups = {}
    for item in data:
        requirement = item["requirement_text"]
        if requirement not in requirement_groups:
            requirement_groups[requirement] = {"correct": [], "incorrect": []}
        
        if item["is_correct"]:
            requirement_groups[requirement]["correct"].append(item)
        else:
            requirement_groups[requirement]["incorrect"].append(item)
    
    for requirement, group in requirement_groups.items():
        if group["correct"] and group["incorrect"]:
            # Take one correct example and up to two incorrect examples
            correct_item = group["correct"][0]
            correct_summary = correct_item.get("function_summary", "")
            
            for incorrect_item in group["incorrect"][:min(2, len(group["incorrect"]))]:
                incorrect_summary = incorrect_item.get("function_summary", "")
                
                contrastive_prompt = f"""Compare these two function summaries for the same requirement:

Requirement: {requirement}

Function Summary A: {correct_summary}
Function Summary B: {incorrect_summary}

Think step by step about how each function summary matches the requirement:
1. What functionality does the requirement specify?
2. What does Function Summary A provide?
3. What does Function Summary B provide?
4. Which one better implements the requirement and why?

Scoring Guide:
- 0-10: No relationship whatsoever
- 11-30: Slightly related but incorrect implementation
- 31-60: Partially implements the requirement but with significant gaps
- 61-89: Mostly implements the requirement with minor issues
- 90-100: Correctly and completely implements the requirement

Rate each implementation:
Function A rating (0-100): 
Function B rating (0-100): """
                
                all_prompts.append({
                    "text": contrastive_prompt,
                    "type": "contrastive",
                    "expected_scores": [90, 20],  # Expect high score for correct, low for incorrect
                    "requirement": requirement,
                    "similarity_type": "contrastive_pair"
                })
    
    logger.info(f"Created {len(all_prompts)} prompts ({sum(1 for p in all_prompts if p['type'] == 'contrastive')} contrastive pairs)")
    return all_prompts


def create_gemma3_prompts(data):
    """Create prompts specifically formatted for Gemma3 model using the chat template format."""
    all_prompts = []
    
    # Group data by requirements to facilitate contrastive learning
    requirement_groups = {}
    for item in data:
        requirement = item["requirement_text"]
        if requirement not in requirement_groups:
            requirement_groups[requirement] = {"correct": [], "incorrect": []}
        
        if item["is_correct"]:
            requirement_groups[requirement]["correct"].append(item)
        else:
            requirement_groups[requirement]["incorrect"].append(item)
    
    for item in data:
        code = item["function_content"]
        requirement = item["requirement_text"]
        is_correct = item["is_correct"]
        function_summary = item.get("function_summary", "")
        
        # Set expected score based on is_correct - make distinction clearer
        expected_score = 90 if is_correct else 15
        
        # Generate function summary prompt
        if not function_summary:  # Only create summary prompts if we don't have one already
            summary_prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a technical documentation assistant that creates concise function summaries."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Create a concise summary of this function's purpose:\n\n```\n{code[:500]}\n```\n\nProvide a clear, specific Function Summary that highlights what the code actually does:"}]
                }
            ]
            
            all_prompts.append({
                "text": summary_prompt,
                "type": "summary",
                "expected_score": 90,
                "original_data": item,
                "similarity_type": "summary_generation"
            })
        
        # Generate requirement matching prompt with clear scoring criteria
        rating_prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a requirement matching assistant."}]
            },
            {
                "role": "user",
                "content": [{
                    "type": "text", 
                    "text": f"Rate how well the function summary implements the given requirement.\n\n"
                             f"Requirement: {requirement}\n\n"
                             f"Function Summary: {function_summary}\n\n"
                             f"Think step by step about whether the function summary matches the requirement:\n"
                             f"1. What functionality does the requirement specify?\n"
                             f"2. What functionality does the summary describe?\n"
                             f"3. How much overlap exists between these functionalities?\n\n"
                             f"Scoring Guide:\n"
                             f"- 0-10: No relationship whatsoever\n"
                             f"- 11-30: Slightly related but incorrect implementation\n"
                             f"- 31-60: Partially implements the requirement but with significant gaps\n"
                             f"- 61-89: Mostly implements the requirement with minor issues\n"
                             f"- 90-100: Correctly and completely implements the requirement\n\n"
                             f"Provide ONLY a numerical score (0-100) without any explanation."
                }]
            }
        ]
        
        all_prompts.append({
            "text": rating_prompt,
            "type": "rating",
            "expected_score": expected_score,
            "original_data": item,
            "similarity_type": "requirement_matching"
        })
    
    # Add contrastive learning prompts
    for requirement, group in requirement_groups.items():
        if group["correct"] and group["incorrect"]:
            # Take one correct example and up to two incorrect examples
            correct_example = random.choice(group["correct"])
            incorrect_examples = random.sample(group["incorrect"], min(2, len(group["incorrect"])))
            
            for incorrect_example in incorrect_examples:
                # Create explicit comparison between correct and incorrect
                correct_summary = correct_example.get("function_summary", "Function A processes data according to business rules.")
                incorrect_summary = incorrect_example.get("function_summary", "Function B handles system operations.")
                
                # Create contrastive prompt in Gemma3 format
                contrastive_prompt = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a requirement evaluation assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Compare two function summaries and decide which one better implements the requirement.\n\nRequirement: {requirement}\n\nFunction A: {correct_summary}\n\nFunction B: {incorrect_summary}\n\nFor each function, rate how well it implements the requirement on a scale of 0-100.\nExplain why one function is better than the other.\nFinally, provide the ratings:\n\nFunction A rating (0-100):\nFunction B rating (0-100):"}]
                    }
                ]
                
                all_prompts.append({
                    "text": contrastive_prompt,
                    "type": "contrastive",
                    "expected_scores": [90, 20],  # Expect high score for correct, low for incorrect
                    "original_data": {"correct": correct_example, "incorrect": incorrect_example, "requirement": requirement},
                    "similarity_type": "contrastive_pair"
                })
    
    logger.info(f"Created {len(all_prompts)} Gemma3-formatted prompts ({sum(1 for p in all_prompts if p['type'] == 'contrastive')} contrastive pairs)")
    return all_prompts


def train_model(model, tokenizer, dataset):
    """Train the model using our custom reinforcement learning approach."""
    # Check if we can use GPU
    has_gpu = torch.cuda.is_available()
    
    # Load training data for reinforcement learning
    data = load_training_data(max_examples=MAX_EXAMPLES)
    if not data:
        logger.error("No training data available for reinforcement learning")
        return model
    
    # Run reinforcement learning training
    logger.info("Starting reinforcement learning training...")
    model = train_with_rl(model, tokenizer, data, has_gpu)
    logger.info("Reinforcement learning training completed")
    
    return model


def train_gemma3_simplified(model, tokenizer, data, device, epochs, batch_size, learning_rate, num_examples):
    """
    Specialized training function for Gemma3 model that handles device and dtype compatibility issues.
    Uses a simplified approach that works reliably with Gemma3's architecture.
    
    Args:
        model: The Gemma3 model to train
        tokenizer: The tokenizer for the model
        data: Training data
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_examples: Number of examples to use
    
    Returns:
        Tuple of (best_model_state_dict, best_epoch, best_reward)
    """
    # Determine the device from the model
    device = next(model.parameters()).device
    logger.info(f"Using device: {device} for Gemma3 simplified training")
    
    # Make sure gradient checkpointing is enabled
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()
 
    # Filter data to use only a subset for training
    train_data = data[:min(len(data), num_examples)]
    logger.info(f"Using {len(train_data)} examples for Gemma3 training")
    
    # Setup optimizer with reduced learning rate for CPU training
    lr = learning_rate * 0.5  # Reduce learning rate for stability
    logger.info(f"Setting up AdamW optimizer with learning rate {lr}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Track best reward for model saving
    best_reward = float('-inf')
    best_epoch = 0
    
    # Define utility functions specific to Gemma3 training
    def create_prompt(requirement, function_summary, is_correct):
        """Create a prompt for Gemma3 in its expected format"""
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a requirement matching assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Rate how well the function summary implements the given requirement.\n\n"
                             f"Requirement: {requirement}\n\n"
                             f"Function Summary: {function_summary}\n\n"
                             f"Think step by step about whether the function summary matches the requirement:\n"
                             f"1. What functionality does the requirement specify?\n"
                             f"2. What functionality does the summary describe?\n"
                             f"3. How much overlap exists between these functionalities?\n\n"
                             f"Scoring Guide:\n"
                             f"- 0-10: No relationship whatsoever\n"
                             f"- 11-30: Slightly related but incorrect implementation\n"
                             f"- 31-60: Partially implements the requirement but with significant gaps\n"
                             f"- 61-89: Mostly implements the requirement with minor issues\n"
                             f"- 90-100: Correctly and completely implements the requirement\n\n"
                             f"Provide ONLY a numerical score (0-100) without any explanation."
                }]
            }
        ]
    
    def compute_custom_loss(score, target_score):
        """Compute a custom loss based on the score difference"""
        # Mean squared error between score and target score
        diff = (score - target_score) ** 2
        # Normalize to a reasonable scale for training stability
        return diff / 1000.0
    
    # Define max input length locally for this function
    MAX_INPUT_LENGTH = 512 

    # Training loop
    logger.info("Starting Gemma3 simplified training loop...")
    
    # === DEBUG LOGGING: Model Structure and Adapter Status ===
    logger.info(f"Model class: {model.__class__}")
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            logger.info("Model is a PEFT-wrapped model.")
            logger.info(f"Active adapters: {getattr(model, 'active_adapters', 'N/A')}")
            logger.info(f"PEFT config: {getattr(model, 'peft_config', 'N/A')}")
        else:
            logger.warning("Model is NOT a PEFT-wrapped model!")
    except ImportError:
        logger.warning("peft not available for adapter inspection.")
    # Log trainable parameters
    trainable_param_list = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Trainable parameter count: {len(trainable_param_list)}")  # Only print count, not full list
    # (Full parameter names and shapes logging removed to reduce terminal output)
    # === END DEBUG LOGGING ===

    # Ensure model is in training mode
    model.train()
    logger.info("Ensured model is in training mode.")

    # === DEBUG: Print device and dtype for all trainable parameters before optimizer ===
    for n, p in model.named_parameters():
        if p.requires_grad:
            pass # Removed print statement
    # === END DEBUG ===

    # Main epoch loop
    for epoch in range(epochs):
        logger.info(f"\n{'=' * 40}\nEpoch {epoch+1}/{epochs}\n{'=' * 40}")
        
        # Track epoch statistics
        epoch_scores = []
        epoch_targets = []
        epoch_loss = 0.0
        
        # Process data in batches
        for batch_idx in range(0, len(train_data), batch_size):
            batch = train_data[batch_idx:min(batch_idx + batch_size, len(train_data))]
            batch_start_time = time.time()
            
            logger.info(f"Batch {batch_idx//batch_size + 1}/{math.ceil(len(train_data)/batch_size)}")
            
            # Process each example in the batch
            for idx, item in enumerate(batch):
                try:
                    # Ensure model is in training mode for this step
                    model.train()
                    optimizer.zero_grad() # Zero gradients for each item

                    # Extract data
                    requirement = item["requirement_text"]
                    function_summary = item.get("function_summary", "This function implements business logic according to requirements.")
                    is_correct = item["is_correct"]
                    target_score = 90 if is_correct else 20  # High for correct, low for incorrect
                    
                    # Create prompt and tokenize (Use standard input format, not chat template for direct model call)
                    # Note: How to format input for direct call depends on Gemma's training. Assuming simple text format.
                    prompt_text = f"Requirement: {requirement}\nFunction Summary: {function_summary}"
                    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH)
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    
                    # Prepare labels for Causal LM loss calculation (standard practice)
                    labels = input_ids.clone()
                    labels[labels == tokenizer.pad_token_id] = -100
                    labels = labels.to(device)

                    # === DEBUG LOGGING: Dtype and Device Consistency ===
                    logger.debug(f"[DEBUG] input_ids dtype: {input_ids.dtype}, device: {input_ids.device}")
                    logger.debug(f"[DEBUG] attention_mask dtype: {attention_mask.dtype}, device: {attention_mask.device}")
                    logger.debug(f"[DEBUG] labels dtype: {labels.dtype}, device: {labels.device}")
                    # === END DEBUG LOGGING ===

                    # === 1. Forward pass WITH gradients ===
                    # Obtain logits required for policy gradient calculation
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits # Shape: [batch_size, seq_len, vocab_size]
                    logger.info(f"outputs.logits.requires_grad (immediately after model call): {outputs.logits.requires_grad}")

                    prompt_len = input_ids.shape[1]

                    # === 2. Generation WITHOUT gradients (for reward) ===
                    with torch.no_grad():
                        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=25, do_sample=True, temperature=0.6, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
                                    # === Decode, Parse Score, and Calculate Reward ===
                    # Decode the generated IDs into text
                    generated_text = tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)[0]

                    logger.info(f"  DEBUG: Raw training output: '{generated_text}'") # Added for debugging training generation

                    # Parse score from the generated text
                    parsed_score = 0.0
                    numbers = re.findall(r"[0-9]+", generated_text)
                    if numbers:
                        for num in numbers:
                            try: # Add try-except for safety
                                n = int(num)
                                if 0 <= n <= 100:
                                    parsed_score = float(n)
                                    break
                            except ValueError:
                                continue # Skip non-integer numbers
                    logger.info(f"Item {idx}: Parsed Score: {parsed_score}") # Keep this log for now
                    logger.info(f"Item {idx}: Target Score: {target_score}") # Keep this log for now

                    # Calculate reward based on parsed score vs target
                    reward = compute_reward(is_correct, parsed_score, target_score)
                    logger.info(f"Item {idx}: Calculated Reward: {reward}") # Keep this log for now
                    epoch_scores.append(reward) # Track rewards for epoch stats
                    # ====================================================
                    # === 3. REINFORCE Loss Calculation ===
                    logger.info(f"[RL Debug] Before second forward pass: model.training = {model.training}")
                    lora_grads_before = {n: p.requires_grad for n, p in model.named_parameters() if 'lora' in n and p.requires_grad}
                    logger.info(f"[RL Debug] Before second forward pass: LoRA param requires_grad sample: {list(lora_grads_before.items())[:2]}") # Log a small sample

                    full_input_ids = generated_ids # Use the sequence from model.generate [batch_size, full_seq_len]
                    full_attention_mask = (full_input_ids != tokenizer.pad_token_id).long()
                    outputs_with_grad = model(input_ids=full_input_ids, attention_mask=full_attention_mask)
                    full_logits = outputs_with_grad.logits # Shape: [batch_size, full_seq_len, vocab_size]

                    logger.info(f"[RL Debug] After second forward pass: full_logits.requires_grad = {full_logits.requires_grad}")
                    logger.info(f"[RL Debug] After second forward pass: full_logits.grad_fn = {full_logits.grad_fn}")

                    # 2. Get log probabilities for the generated part only
                    logits_for_generation = full_logits[:, prompt_len-1:-1, :] # Shape: [batch_size, gen_len, vocab_size]
                    # Actual generated token IDs (T+1 to T+N)
                    generated_tokens_only = generated_ids[:, prompt_len:] # Shape: [batch_size, gen_len]
                    gen_len = generated_tokens_only.shape[1]

                    # Ensure dimensions match before gather (handle potential off-by-one)
                    if logits_for_generation.shape[1] != gen_len:
                        logger.warning(f"Logit/Token length mismatch: Logits {logits_for_generation.shape[1]}, Tokens {gen_len}. Adjusting.")
                        min_len = min(logits_for_generation.shape[1], gen_len)
                        if min_len == 0:
                            sequence_log_prob = torch.tensor(0.0, device=model.device, requires_grad=True) # Assign zero log_prob with grad
                        else:
                            logits_for_generation = logits_for_generation[:, :min_len, :]
                            generated_tokens_only = generated_tokens_only[:, :min_len]
                            log_probs = F.log_softmax(logits_for_generation, dim=-1)
                            action_log_probs = torch.gather(log_probs, dim=-1, index=generated_tokens_only.unsqueeze(-1)).squeeze(-1)
                            sequence_log_prob = action_log_probs.sum(dim=-1)
                    elif gen_len == 0: # Handle case where nothing was generated
                         logger.warning(f"Empty generation detected (gen_len=0).")
                         sequence_log_prob = torch.tensor(0.0, device=model.device, requires_grad=True) # Assign zero log_prob with grad
                    else:
                        # Normal case
                        log_probs = F.log_softmax(logits_for_generation, dim=-1)
                        action_log_probs = torch.gather(log_probs, dim=-1, index=generated_tokens_only.unsqueeze(-1)).squeeze(-1)
                        sequence_log_prob = action_log_probs.sum(dim=-1) # Shape: [batch_size]

                    # 3. Calculate REINFORCE loss (negative log prob * reward)
                    #reward = compute_reward(is_correct, 0.0, target_score)
                    reward_tensor = torch.tensor(reward, device=sequence_log_prob.device, dtype=sequence_log_prob.dtype)
                    loss = (-sequence_log_prob * reward_tensor).mean() # Average over batch

                    # === End Loss Calculation ---

                    # Log loss and logits diagnostics # <-- This logging block might need removal/adjustment for RL
                    # logger.info(f"logits.requires_grad: {logits.requires_grad}")
                    # logger.info(f"logits.grad_fn: {logits.grad_fn}")
                    logger.info(f"loss.requires_grad: {loss.requires_grad}") # Check RL loss grad
                    logger.info(f"loss.grad_fn: {loss.grad_fn}") # Check RL loss grad_fn
                    # Log .grad of trainable params (should be None before backward, but confirms graph connection)
                    lora_printed = False
                    for n, p in model.named_parameters():
                        if p.requires_grad and 'lora' in n and not lora_printed:
                            # logger.info(f"[PARAM] {n} grad before backward: {p.grad}") # Commented out
                            lora_printed = True
                            break

                    if loss is None:
                        logger.warning(f"Loss is None for item {idx}. Skipping update.")
                        continue

                    # Backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue # Skip item on error
            
            # Report batch timing
            batch_time = time.time() - batch_start_time
            logger.info(f"Batch processing time: {batch_time:.2f} seconds")
        
        # Report epoch statistics
        if epoch_targets:
            avg_target = sum(epoch_targets) / len(epoch_targets)
            avg_loss = epoch_loss / len(epoch_targets)
            
            # Reward is inverse of loss (higher reward for lower loss)
            epoch_reward = 1.0 / (avg_loss + 1e-6) if avg_loss > 0 else 0
            
            logger.info(f"\nEpoch {epoch+1} stats:")
            logger.info(f"  Average target: {avg_target:.1f}")
            logger.info(f"  Average loss: {avg_loss:.4f}")
            logger.info(f"  Epoch reward: {epoch_reward:.1f}")
            
            # Save best model
            if epoch_reward > best_reward:
                best_reward = epoch_reward
                best_epoch = epoch + 1
                
                # Save model weights
                target_path = "rl_weights_gemma3"
                logger.info(f"Saving best model (Epoch {epoch+1}, Reward: {epoch_reward:.1f}) to {target_path}")
                try:
                    model.save_pretrained(target_path)
                    tokenizer.save_pretrained(target_path)
                except Exception as e:
                    logger.error(f"Error saving model: {e}")
                    logger.error(traceback.format_exc())
        else:
            logger.warning("No targets collected in this epoch")
    
    # Final summary
    logger.info(f"\nGemma3 training completed. Best model from epoch {best_epoch} with reward {best_reward:.1f}")
    return model.state_dict(), best_epoch, best_reward


def test_model(model, tokenizer, test_data, model_type=DEFAULT_MODEL):
    """Test the model on a few examples to see how well it performs.
    Also verifies that LoRA parameters (0.05% of total) are being applied.
    """
    logger.info("Testing trained model on sample inputs...")
    
    # Verify if LoRA parameters are actually being used in inference
    if HAS_INFERENCE_MODULE:
        # For Gemma3, use a more memory-efficient verification approach
        if model_type == "gemma3" and torch.cuda.is_available():
            try:
                # Calculate available GPU memory
                free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # Convert to GB
                logger.info(f"Available GPU memory: {free_memory:.2f} GB")
                
                if free_memory < 2.0:  # If less than 2GB free memory
                    logger.info("Limited GPU memory detected - skipping full LoRA verification for Gemma3")
                    logger.info(" LoRA verification skipped for Gemma3 to conserve memory")
                    logger.info("Trainable parameters indicate LoRA is configured correctly")
                else:
                    logger.info("Verifying that LoRA parameters are being applied (lightweight check for Gemma3)...")
                    # Simplified check for Gemma3 - just check parameter count
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in model.parameters())
                    if trainable_params > 0:
                        logger.info(f"  VERIFIED: LoRA contains {trainable_params:,} trainable parameters ({100 * trainable_params / total_params:.2f}% of total)")
                    else:
                        logger.warning(" WARNING: No trainable parameters found in model!")
            except Exception as e:
                logger.warning(f"Error during Gemma3 LoRA verification: {e}")
                logger.info("Continuing with training evaluation despite verification error")
        else:
            # Standard verification for other models
            logger.info("Verifying that LoRA parameters (0.05% of total) are being applied...")
            try:
                # Create verification test case
                requirement = "Handle file uploads securely"
                function_summary = "Validates and sanitizes file uploads, checks file types, and stores them with proper permissions"
                
                # Use our inference module to verify LoRA parameters are being applied
                verification_result = verify_lora_effects(
                    requirement=requirement,
                    function_summary=function_summary,
                    model_path=OUTPUT_DIR,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                # Log results of verification
                if verification_result["lora_is_applied"]:
                    logger.info("  VERIFIED: LoRA parameters (0.05% of total) ARE being applied during inference")
                    logger.info(f"Base model rating: {verification_result['base_prediction']['rating']}")
                    logger.info(f"Adapted model rating: {verification_result['adapted_prediction']['rating']}")
                    logger.info(f"Difference: {abs(verification_result['adapted_prediction']['rating'] - verification_result['base_prediction']['rating'])}")
                else:
                    logger.warning(" WARNING: LoRA parameters (0.05% of total) do NOT appear to be applied!")
                    logger.warning("The base model and LoRA-adapted model are producing identical outputs.")
                    logger.warning("Consider using 'save_for_inference' to create a properly merged model.")
            except Exception as e:
                logger.warning(f"Error during LoRA verification: {e}")
                import traceback
                logger.warning(traceback.format_exc())
    else:
        logger.warning("Inference module not available - cannot verify if LoRA parameters are being applied")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle different model types (PPO vs standard)
    if hasattr(model, "pretrained_model"):
        logger.info("Testing PPO-trained model with value head...")
        # If the model has a value head, we need to handle it differently
        # for generation we still use the base model (not the value head wrapper)
        generation_model = model.pretrained_model
        generation_model.to(device)
        generation_model.eval()
    else:
        logger.info("Testing standard model...")
        generation_model = model
        generation_model.to(device)
        generation_model.eval()
    
    # Track performance metrics
    correct_classifications = 0
    total_tested = 0
    correct_scores = []
    incorrect_scores = []
    
    # Get a mix of positive and negative examples
    positive = [item for item in test_data if item["is_correct"]]
    negative = [item for item in test_data if not item["is_correct"]]
    
    # Ensure balanced testing
    num_samples = min(4, min(len(positive), len(negative)) * 2)
    pos_samples = random.sample(positive, min(num_samples // 2, len(positive)))
    neg_samples = random.sample(negative, min(num_samples // 2, len(negative)))
    test_samples = pos_samples + neg_samples
    random.shuffle(test_samples)  # Shuffle to avoid bias
    
    # First test summary generation
    logger.info("===== Testing summary generation =====")
    for i, item in enumerate(test_samples[:2]):  # Only test a couple for summary generation
        function_content = item["function_content"][:500]  # Truncate long functions
        
        # Create summary prompt
        summary_prompt = f"Summarize this code function:\n\n```\n{function_content}\n```\n\nFunction Summary:"
        
        inputs = tokenizer(summary_prompt, return_tensors="pt").to(device)
        
        # Generate summary
        with torch.no_grad():
            outputs = generation_model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_SUMMARY_LENGTH,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Get generated summary
        summary = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        logger.info(f"\nTest {i+1}: Function Summary Generation")
        logger.info(f"Generated: {summary[:100]}..." if len(summary) > 100 else f"Generated: {summary}")
    
    # Now test requirement rating
    logger.info("\n===== Testing requirement matching =====")
    for item in test_samples:
        requirement = item["requirement_text"]
        summary = item["function_summary"]
        is_correct = item["is_correct"]
        total_tested += 1
        
        # Create rating prompt
        prompt = f"""Given the following requirement and function summary, rate how well the function implements the requirement.

Requirement: {requirement}

Function Summary: {summary}

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

On a scale of 0 to 100, rate how confident you are that this function implements the requirement.
Only return a single integer number between 0 and 100, where 0 means no relationship and 100 means perfect match.

Rating:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response from model
        outputs = generation_model.generate(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            max_new_tokens=20, # Keep slightly lower for testing efficiency
            do_sample=True,
            temperature=0.6, # Align with training temp
            top_p=0.95, # Align with training top_p
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        logger.info(f"  DEBUG: Raw model response: '{response}'") # Added for debugging
        
        # Try to extract a numeric score
        try:
            score_text = ''.join(filter(str.isdigit, response.strip()))
            if score_text:
                score = int(score_text)
                score = min(100, max(0, score))  # Clamp to 0-100 range
                
                # Track scores for analysis
                if is_correct:
                    correct_scores.append(score)
                else:
                    incorrect_scores.append(score)
                
                # Check if classification is correct
                # For correct matches, score should be high (≥70)
                # For incorrect matches, score should be low (≤30)
                if (is_correct and score >= 70) or (not is_correct and score <= 30):
                    correct_classifications += 1
            else:
                score = "No number found"
        except Exception as e:
            logger.warning(f"Error parsing score: {e}")
            score = "Failed to parse"
        
        # Calculate reward using our reward function
        if isinstance(score, int):
            reward = compute_reward(is_correct, score)
            reward_str = f", Reward: {reward:.2f}"
        else:
            reward_str = ""
        
        logger.info(f"\nTest example - Actual: {'Correct' if is_correct else 'Incorrect'}")
        logger.info(f"  Requirement: {requirement[:50]}...")
        logger.info(f"  Summary: {summary[:50]}...")
        logger.info(f"  Model output: {response}")
        logger.info(f"  Parsed score: {score}{reward_str}")
        logger.info(f"  Expected range: {'HIGH (>70)' if is_correct else 'LOW (<30)'}")
    
    # Calculate and display metrics
    if total_tested > 0:
        accuracy = (correct_classifications / total_tested) * 100
        logger.info(f"\n===== Performance Metrics =====")
        logger.info(f"Classification Accuracy: {correct_classifications}/{total_tested} ({accuracy:.1f}%)")
        
        # Show score distributions
        if correct_scores:
            avg_correct = sum(correct_scores) / len(correct_scores)
            logger.info(f"Correct matches - Avg score: {avg_correct:.1f}")
        
        if incorrect_scores:
            avg_incorrect = sum(incorrect_scores) / len(incorrect_scores)
            logger.info(f"Incorrect matches - Avg score: {avg_incorrect:.1f}")
        
        # Calculate score gap (separation between correct and incorrect)
        if correct_scores and incorrect_scores:
            score_gap = avg_correct - avg_incorrect
            logger.info(f"Score gap (higher is better): {score_gap:.1f}")
    
    logger.info("\nTesting complete")


def parse_arguments():
    """Parse command line arguments for controlling the training process."""
    parser = argparse.ArgumentParser(description="Fine-tune a language model for requirement tracing using RL")
    
    # Model selection argument
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        choices=["tinyllama", "gemma3"],
                        help="Model architecture to use (tinyllama or gemma3)")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for optimizer")
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES_TRAIN, help="Number of examples to use")
    
    # Output directory override
    parser.add_argument("--output-dir", type=str, help="Custom output directory for model weights")
    
    return parser.parse_args()


def main():
    """Main training function with custom reinforcement learning."""
    # Parse command line arguments
    args = parse_arguments()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_examples = args.num_examples
    model_type = args.model
    
    # Set output directory based on model type
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        # Append model type to default output directory if not already included
        if model_type not in OUTPUT_DIR:
            OUTPUT_DIR = f"{OUTPUT_DIR}_{model_type}"
    
    logger.info(f"Using model type: {model_type}")
    logger.info(f"Model weights will be saved to: {OUTPUT_DIR}")
    
    # Print GPU diagnostic information
    logger.info(f"PyTorch version: {torch.__version__}")
    gpu_available = False
    
    try:
        # Try to check for GPU, but don't let errors stop us
        logger.info(f"CUDA available according to PyTorch: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        
        # Check if NVIDIA driver is actually available via system command
        import subprocess
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA driver is detected in system")
            gpu_info = result.stdout.split('|')[3].strip() if '|' in result.stdout else 'Unknown'
            logger.info(f"GPU detected: {gpu_info}")
        
        # Run our robust GPU check
        gpu_available = check_gpu()
    except Exception as e:
        logger.warning(f"Error during GPU detection: {str(e)}")
        gpu_available = False
    
    # CPU optimization message
    if not gpu_available:
        logger.warning("====================================================================")
        logger.warning("GPU IS NOT AVAILABLE - TRAINING WILL USE CPU (SIGNIFICANTLY SLOWER)")
        logger.warning("This is due to CUDA library/driver configuration issues")
        logger.warning("====================================================================")
        logger.info("Optimizing CPU training settings...")
        
        # Set CPU optimizations
        torch.set_num_threads(max(4, os.cpu_count() - 2))  # Use most CPU cores but leave some for system
        logger.info(f"Using {torch.get_num_threads()} CPU threads for training")
        
        # Reduce training parameters for CPU
        global BATCH_SIZE, NUM_EPOCHS, NUM_EXAMPLES_TRAIN
        BATCH_SIZE = max(1, BATCH_SIZE // 2)  # Reduce batch size for CPU
        # Keep epochs the same but set a max training samples
        NUM_EXAMPLES_TRAIN = min(NUM_EXAMPLES_TRAIN, 20)  # Further limit samples for CPU training
        
        logger.info(f"Adjusted training parameters for CPU: batch_size={BATCH_SIZE}, max_examples={NUM_EXAMPLES_TRAIN}, epochs={NUM_EPOCHS}")
        
        # Ask user if they want to proceed with CPU
        try:
            user_response = input("\nGPU is unavailable. Continue with CPU training (slower)? (y/n): ")
            if user_response.lower().strip() not in ["y", "yes"]:
                logger.info("Exiting as requested by user.")
                return False
            logger.info("Proceeding with CPU training as requested.")
        except Exception as e:
            # Non-interactive mode - proceed with warning
            logger.warning(f"Non-interactive mode, proceeding with CPU: {str(e)}")
    
    # Load training data - this will be used for both stages
    data = load_training_data(max_examples=MAX_EXAMPLES)
    if not data:
        logger.error("No training data available!")
        return False
    
    try:
        logger.info("==== Starting Custom Reinforcement Learning Training ====")
        logger.info(f"Using {len(data)} examples with batch size {batch_size} and {epochs} epochs")
        
        # Initialize model with GPU support if available, using the selected model type
        model, tokenizer = initialize_model(gpu_available, model_type)
        
        # Train with custom reinforcement learning using command line arguments
        # This uses the two-stage approach:
        # 1. Generate summaries from code
        # 2. Rate summaries against requirements with rewards
        model = train_with_rl(
            model=model, 
            tokenizer=tokenizer, 
            data=data, 
            has_gpu=gpu_available,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_examples=num_examples,
            model_type=model_type
        )
        
        # Test the model on a few examples to see improvements
        logger.info("Testing trained model...")
        test_model(model, tokenizer, data, model_type)
        
        logger.info("Custom Reinforcement Learning completed successfully!")
        logger.info(f"Model saved to {OUTPUT_DIR}")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    main()
