import torch
import logging
import os
import re
import json
import random
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Basic Logging Setup
def setup_logging():
    """Sets up basic logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger(__name__)

logger = setup_logging()

# Constants
TRAINING_DATA_PATH = "training_data.jsonl"
MAX_EXAMPLES = 40  # Default limit, can be overridden by args

# GPU Check
def check_gpu():
    """Checks GPU availability and prints details."""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available via torch.cuda.is_available().")
        return False
    
    try:
        logger.info(f"CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        logger.info(f"Detected {device_count} CUDA device(s).")
        
        if device_count == 0:
            logger.warning("No CUDA devices detected by PyTorch.")
            return False
            
        for i in range(device_count):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
            logger.info(f"    Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        
        # Simple allocation test
        try:
            tensor = torch.tensor([1.0, 2.0]).cuda()
            logger.info("Simple CUDA tensor allocation successful.")
            del tensor
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"CUDA allocation test failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"An error occurred during GPU check: {e}")
        return False

# Load Training Data function
def load_training_data(max_examples=MAX_EXAMPLES):
    """Load training data from the JSONL file."""
    if not os.path.exists(TRAINING_DATA_PATH):
        logger.error(f"Training data file {TRAINING_DATA_PATH} not found!")
        return None
    
    data = []
    try:
        with open(TRAINING_DATA_PATH, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line in {TRAINING_DATA_PATH}: {line.strip()} - Error: {e}")
    except Exception as e:
        logger.error(f"Failed to read {TRAINING_DATA_PATH}: {e}")
        return None

    if not data:
        logger.error(f"No valid data loaded from {TRAINING_DATA_PATH}")
        return None
    
    # Limit examples if needed
    if max_examples is not None and len(data) > max_examples:
        # Ensure a balanced dataset if possible (optional, but good practice)
        positive = [item for item in data if item.get("is_correct", False)] # Use .get for safety
        negative = [item for item in data if not item.get("is_correct", False)]
        
        # Determine how many of each to keep
        target_total = min(len(data), max_examples)
        pos_to_keep = min(len(positive), target_total // 2)
        neg_to_keep = min(len(negative), target_total - pos_to_keep)
        
        # Adjust if we don't have enough of one type
        if pos_to_keep < target_total // 2:
            neg_to_keep = min(len(negative), target_total - pos_to_keep)
        elif neg_to_keep < target_total // 2:
             pos_to_keep = min(len(positive), target_total - neg_to_keep)

        # Sample randomly
        random.shuffle(positive)
        random.shuffle(negative)
        
        sampled_data = positive[:pos_to_keep] + negative[:neg_to_keep]
        random.shuffle(sampled_data)
        
        logger.info(f"Using {len(sampled_data)} examples ({pos_to_keep} positive, {neg_to_keep} negative) from {TRAINING_DATA_PATH}")
        return sampled_data
    else:
        logger.info(f"Loaded {len(data)} examples from {TRAINING_DATA_PATH}")
        random.shuffle(data) # Shuffle even if not sampling
        return data

# Prompt Creation Function
def create_prompt(requirement, function_summary):
    """Create the scoring prompt for the base model test (Simplified)."""
    # Using the same structure as the original prompt
    return (
        f"Rate how well the function summary implements the given requirement on a scale of 0 to 100.\n\n"
        f"Requirement: {requirement}\n\n"
        f"Function Summary: {function_summary}\n\n"
        # Removed 'Think step by step' section
        f"Scoring Guide:\n"
        f"- 0-10: No relationship\n"
        f"- 11-30: Slightly related, incorrect implementation\n"
        f"- 31-60: Partially implements, significant gaps\n"
        f"- 61-89: Mostly implements, minor issues\n"
        f"- 90-100: Correctly and completely implements\n\n"
        f"Provide ONLY the numerical score (0-100):" # Adjusted final instruction
    )

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Test Base Gemma Model on Training Data.")
    parser.add_argument(
        "--max_examples", 
        type=int, 
        default=MAX_EXAMPLES, 
        help=f"Maximum number of training examples to load (default: {MAX_EXAMPLES}). Set to 0 or negative for all."
    )
    args = parser.parse_args()
    
    # Use None for max_examples if user wants all data
    max_load_examples = args.max_examples if args.max_examples > 0 else None

    logger.info("==== Starting Base Gemma Model Test on Training Data ====")
    
    # Check GPU
    gpu_available = check_gpu()
    device = torch.device("cuda" if gpu_available else "cpu")
    logger.info(f"Using device: {device}")

    # Load training data FIRST
    logger.info(f"Loading training data from {TRAINING_DATA_PATH} (max examples: {max_load_examples or 'all'})")
    data = load_training_data(max_examples=max_load_examples)
    if data is None or not data:
        logger.error("Failed to load training data. Exiting.")
        return
    logger.info(f"Successfully loaded {len(data)} examples for testing.")

    # Model Configuration
    base_model_name = "google/gemma-3-4b-it" # Changed to 4B model
    
    # Load base model and tokenizer
    try:
        logger.info(f"Loading base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Use bfloat16 if available on GPU for efficiency
        model_dtype = torch.bfloat16 if gpu_available and torch.cuda.is_bf16_supported() else torch.float32
        logger.info(f"Using model dtype: {model_dtype}")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=model_dtype,
            device_map="auto" if gpu_available else None # Automatically map layers to available GPUs
        )
        
        # If not using device_map (i.e., on CPU), ensure model is on the correct device
        if not gpu_available:
            model = model.to(device)
            
        model.eval() # Set to evaluation mode
        logger.info(f"Base model '{base_model_name}' loaded successfully onto device: {model.device}")

    except Exception as e:
        logger.error(f"Failed to load base model '{base_model_name}': {e}")
        logger.error("Ensure you have internet connectivity and the model name is correct.")
        import traceback
        logger.error(traceback.format_exc())
        return

    # --- Test Loop ---
    logger.info("\n--- Generating Outputs for Loaded Data ---")
    total_start_time = time.time()
    
    for i, item in enumerate(data): # Iterate over loaded data
        logger.info(f"\n--- Test Case {i+1}/{len(data)} ---")
        # Ensure keys exist, provide defaults if not
        requirement = item.get("requirement_text", "MISSING REQUIREMENT")
        summary = item.get("function_summary", "MISSING SUMMARY")
        is_correct = item.get("is_correct", None) # Handle potentially missing key
        
        logger.info(f"Requirement: {requirement}")
        logger.info(f"Summary: {summary}")
        if is_correct is not None:
             logger.info(f"Correct Match (from data): {is_correct}")
        else:
             logger.info("Correct Match: Key 'is_correct' missing in data")

        prompt = create_prompt(requirement, summary)
        logger.info(f"--- Input Prompt Start ---\n{prompt}\n--- Input Prompt End ---")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # Ensure input is on model's device

        start_time = time.time()
        try:
            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,  # Reduced length
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, # Ensure pad_token_id is set
                    # Explicitly unset sampling parameters when do_sample=False
                    top_p=None,
                    top_k=None,
                    temperature=None # Temperature is irrelevant if do_sample=False, but set to None for clarity
                )
            
            # Decode the generated part
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            end_time = time.time()
            
            logger.info(f"Raw Model Output: '{response}'")
            logger.info(f"Generation Time: {end_time - start_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error during generation for test case {i+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    total_end_time = time.time()
    logger.info(f"\n--- Test Complete ---")
    logger.info(f"Total testing time: {total_end_time - total_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
