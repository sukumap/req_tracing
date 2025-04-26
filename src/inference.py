#!/usr/bin/env python3
"""
Inference utilities for requirement tracing models.

This module provides functions for:
1. Loading models with LoRA adaptations for inference
2. Predicting requirement-to-function matches
3. Saving models in both LoRA-only and merged formats for different inference scenarios
4. Verifying that LoRA adaptations are being correctly applied

By using these functions, you ensure that the 0.05% of parameters that were trained
via LoRA are properly applied during inference.
"""

import os
import torch
import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUTPUT_DIR = "rl_weights"
MAX_NEW_TOKENS = 5
MAX_SUMMARY_LENGTH = 100


def load_base_model(device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the base model without LoRA adaptations.
    
    Args:
        device: Device to load the model on ('auto', 'cpu', or CUDA device)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model {BASE_MODEL_ID}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map=device,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    
    logger.info(f"Base model loaded on device: {next(model.parameters()).device}")
    return model, tokenizer


def load_adapted_model(model_path: str, device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model with LoRA adaptations for inference.
    
    This ensures the trained 0.05% of parameters are properly applied.
    
    Args:
        model_path: Path to the model with LoRA adapters
        device: Device to load the model on ('auto', 'cpu', or CUDA device)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading adapted model from {model_path}...")
    
    # Check if this is a merged model or LoRA-only
    is_peft_model = os.path.exists(os.path.join(model_path, "adapter_model.bin"))
    
    if is_peft_model:
        # Load the base model first
        logger.info("Loading as PEFT (LoRA) model...")
        base_model, tokenizer = load_base_model(device)
        
        # Then load LoRA adapters
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            is_trainable=False  # Set to False for inference
        )
    else:
        # Load as a merged model
        logger.info("Loading as merged model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Set to evaluation mode
    model.eval()
    
    logger.info(f"Adapted model loaded on device: {next(model.parameters()).device}")
    return model, tokenizer


def save_for_inference(model: PeftModel, tokenizer: AutoTokenizer, output_dir: str) -> None:
    """
    Save the model in both LoRA-only and merged formats for inference.
    
    The LoRA-only format is smaller but requires loading the base model.
    The merged format is larger but offers faster inference.
    
    Args:
        model: The trained model with LoRA adapters
        tokenizer: The tokenizer
        output_dir: Directory to save the models
    """
    # Create output directories
    lora_dir = os.path.join(output_dir, "lora_adapters")
    merged_dir = os.path.join(output_dir, "merged_model")
    
    os.makedirs(lora_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)
    
    # Save LoRA adapters (small but requires loading base model)
    logger.info(f"Saving LoRA adapters to {lora_dir}...")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    
    # Save merged model (larger but faster inference)
    logger.info(f"Saving merged model to {merged_dir}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    
    logger.info("Models saved successfully for inference")


def predict_requirement_match(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    requirement: str,
    function_summary: str,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Predict how well a function implements a requirement using the trained model.
    
    Args:
        model: The model (base or adapted)
        tokenizer: The tokenizer
        requirement: The requirement text
        function_summary: The function summary text
        temperature: Temperature for generation (lower = more deterministic)
        
    Returns:
        Dictionary with prediction results
    """
    prompt = f"""Rate how well this function implements the requirement:

Requirement: {requirement}

Function Summary: {function_summary}

Think step by step about how well the function summary matches the requirement:
1. What functionality does the requirement specify?
2. What functionality does the function summary describe?
3. How much overlap exists between these functionalities?

Rating scale:
- 0-10: No match - completely different functionality
- 11-30: Poor match - minimal overlap, missing core functionality
- 31-50: Partial match - some overlap but significant gaps
- 51-70: Good match - substantial overlap with minor differences
- 71-90: Very good match - implements most requirements with minimal gaps
- 91-100: Perfect match - completely implements the requirement

Rate from 0-100:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract the prediction
    prediction_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    logger.info(f"Raw prediction: {prediction_text}")
    # Parse the rating
    try:
        # Extract just the digits
        rating_text = ''.join(filter(str.isdigit, prediction_text.strip()))
        rating = int(rating_text) if rating_text else 50
        rating = min(100, max(0, rating))  # Clamp between 0 and 100
    except Exception as e:
        logger.warning(f"Error parsing rating '{prediction_text}': {e}")
        rating = 50  # Default
    
    # Determine if it's a match (over 70 is considered a match)
    is_match = rating > 70
    
    return {
        "requirement": requirement,
        "function_summary": function_summary,
        "rating": rating,
        "is_match": is_match,
        "raw_prediction": prediction_text.strip()
    }


def verify_lora_effects(
    requirement: str,
    function_summary: str,
    model_path: str = None,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Verify that LoRA adaptations are being applied by comparing 
    base model and adapted model predictions.
    
    Args:
        requirement: Requirement text
        function_summary: Function summary text
        model_path: Path to the adapted model
        device: Device to load models on
        
    Returns:
        Dictionary with comparison results
    """
    if model_path is None:
        model_path = DEFAULT_OUTPUT_DIR
    
    # Load base model
    base_model, base_tokenizer = load_base_model(device)
    
    # Load adapted model
    adapted_model, adapted_tokenizer = load_adapted_model(model_path, device)
    
    # Get predictions from both models
    base_prediction = predict_requirement_match(base_model, base_tokenizer, requirement, function_summary)
    adapted_prediction = predict_requirement_match(adapted_model, adapted_tokenizer, requirement, function_summary)
    
    # Check if predictions differ
    prediction_differs = base_prediction["raw_prediction"] != adapted_prediction["raw_prediction"]
    rating_differs = base_prediction["rating"] != adapted_prediction["rating"]
    
    # Log results
    logger.info(f"Base model rating: {base_prediction['rating']} ({base_prediction['raw_prediction']})")
    logger.info(f"Adapted model rating: {adapted_prediction['rating']} ({adapted_prediction['raw_prediction']})")
    logger.info(f"LoRA adaptations {'are' if prediction_differs else 'are NOT'} being applied")
    
    return {
        "base_prediction": base_prediction,
        "adapted_prediction": adapted_prediction,
        "prediction_differs": prediction_differs,
        "rating_differs": rating_differs,
        "lora_is_applied": prediction_differs or rating_differs
    }


def generate_function_summary(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    function_code: str,
    temperature: float = 0.7
) -> str:
    """
    Generate a summary for a function using the model.
    
    Args:
        model: The model (base or adapted)
        tokenizer: The tokenizer
        function_code: The function code to summarize
        temperature: Temperature for generation
        
    Returns:
        Generated function summary
    """
    prompt = f"""Summarize this code function:

```
{function_code[:500]}  # Limit code size
```

Function Summary:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=MAX_SUMMARY_LENGTH,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract the summary
    summary = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return summary.strip()


def batch_verify_lora(test_cases: List[Dict[str, str]], model_path: str = None, device: str = "auto") -> Dict[str, Any]:
    """
    Run batch verification to confirm LoRA adaptations are correctly applied.
    
    Args:
        test_cases: List of dictionaries with 'requirement' and 'function_summary' keys
        model_path: Path to the adapted model
        device: Device to load models on
        
    Returns:
        Dictionary with verification statistics
    """
    if model_path is None:
        model_path = DEFAULT_OUTPUT_DIR
    
    # Load models once
    base_model, base_tokenizer = load_base_model(device)
    adapted_model, adapted_tokenizer = load_adapted_model(model_path, device)
    
    results = []
    differs_count = 0
    
    for i, test_case in enumerate(test_cases):
        requirement = test_case["requirement"]
        function_summary = test_case["function_summary"]
        
        # Get predictions
        base_pred = predict_requirement_match(base_model, base_tokenizer, requirement, function_summary)
        adapted_pred = predict_requirement_match(adapted_model, adapted_tokenizer, requirement, function_summary)
        
        # Check differences
        differs = base_pred["rating"] != adapted_pred["rating"]
        if differs:
            differs_count += 1
        
        results.append({
            "requirement": requirement,
            "function_summary": function_summary,
            "base_rating": base_pred["rating"],
            "adapted_rating": adapted_pred["rating"],
            "rating_difference": adapted_pred["rating"] - base_pred["rating"],
            "differs": differs
        })
        
        logger.info(f"Test case {i+1}/{len(test_cases)}: Difference: {adapted_pred['rating'] - base_pred['rating']}")
    
    # Calculate statistics
    differences = [r["adapted_rating"] - r["base_rating"] for r in results]
    avg_difference = sum(differences) / len(differences) if differences else 0
    abs_differences = [abs(d) for d in differences]
    avg_abs_difference = sum(abs_differences) / len(abs_differences) if abs_differences else 0
    
    logger.info(f"LoRA verification complete: {differs_count}/{len(test_cases)} ({differs_count/len(test_cases)*100:.1f}%) cases differ")
    logger.info(f"Average rating difference: {avg_difference:.2f}")
    logger.info(f"Average absolute difference: {avg_abs_difference:.2f}")
    
    return {
        "results": results,
        "differs_count": differs_count,
        "total_cases": len(test_cases),
        "differs_percentage": differs_count/len(test_cases)*100 if test_cases else 0,
        "avg_difference": avg_difference,
        "avg_abs_difference": avg_abs_difference,
        "lora_is_active": avg_abs_difference > 1.0  # Consider LoRA active if average difference > 1
    }


if __name__ == "__main__":
    """
    Simple command-line interface to verify LoRA is being applied.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify LoRA adaptations for requirement tracing")
    parser.add_argument("--model-path", type=str, default=DEFAULT_OUTPUT_DIR, help="Path to the adapted model")
    parser.add_argument("--device", type=str, default="auto", help="Device to load models on ('auto', 'cpu', or CUDA device)")
    args = parser.parse_args()
    
    # Example test cases
    test_cases = [
        {
            "requirement": "Encrypt data using AES algorithm",
            "function_summary": "AES encryption implementation that processes data blocks"
        },
        {
            "requirement": "Create TCP socket connection",
            "function_summary": "Establishes HTTP REST endpoint for user authentication"
        },
        {
            "requirement": "Add two numbers",
            "function_summary": "Function that sums two integer values and returns the result"
        },
        {
            "requirement": "Process command line arguments",
            "function_summary": "Parses string inputs for file path pattern matching"
        }
    ]
    
    # Run verification
    verification_results = batch_verify_lora(test_cases, args.model_path, args.device)
    
    # Display detailed results
    print("\n===== LoRA Verification Results =====")
    print(f"Model path: {args.model_path}")
    print(f"Cases that differ: {verification_results['differs_count']}/{verification_results['total_cases']} ({verification_results['differs_percentage']:.1f}%)")
    print(f"Average rating difference: {verification_results['avg_difference']:.2f}")
    print(f"Average absolute difference: {verification_results['avg_abs_difference']:.2f}")
    print(f"LoRA is {'ACTIVE' if verification_results['lora_is_active'] else 'NOT ACTIVE'}")
    
    # Individual case results
    print("\n===== Individual Test Cases =====")
    for i, result in enumerate(verification_results["results"]):
        print(f"\nCase {i+1}:")
        print(f"  Requirement: {result['requirement']}")
        print(f"  Function: {result['function_summary']}")
        print(f"  Base rating: {result['base_rating']}")
        print(f"  Adapted rating: {result['adapted_rating']}")
        print(f"  Difference: {result['rating_difference']} ({'differs' if result['differs'] else 'same'})")
