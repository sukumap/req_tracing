#!/usr/bin/env python3
"""
Requirement Matcher - Inference tool for requirement tracing

This script uses your trained TinyLlama model with the most recent LoRA weights
to perform requirement tracing tasks:

1. Generate summaries of functions from code
2. Rate how well functions match requirements

Usage:
    # Generate a function summary
    python requirement_matcher.py --function-code "def add(a, b): return a + b"
    
    # Rate a function against a requirement
    python requirement_matcher.py --requirement "Add two numbers" --function-summary "This function adds two integers"
    
    # Do both (summarize then rate)
    python requirement_matcher.py --requirement "Add two numbers" --function-code "def add(a, b): return a + b"
    
    # Use specific model weights (otherwise uses latest epoch)
    python requirement_matcher.py --model-path rl_weights/epoch-100 --requirement "Add two numbers" --function-code "def add(a, b): return a + b"
    
    # Read function code from a file
    python requirement_matcher.py --requirement "Add two numbers" --input-file my_function.py
"""

import os
import sys
import torch
import logging
import argparse
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "rl_weights"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def run_inference(model_path=None, requirement=None, function_code=None, function_summary=None):
    """
    Run inference with the trained model on new inputs.
    
    This function handles two main inference tasks:
    1. Generate a summary for a function (if function_code is provided)
    2. Rate how well a function summary matches a requirement (if both are provided)
    
    Args:
        model_path: Path to the model with LoRA adapters (defaults to latest in OUTPUT_DIR)
        requirement: Requirement text for matching (optional)
        function_code: Function code to summarize (optional)
        function_summary: Pre-existing function summary for matching (optional)
        
    Returns:
        Dictionary with inference results
    """
    results = {}
    
    # Import inference module
    try:
        from inference import load_adapted_model, generate_function_summary, predict_requirement_match
    except ImportError:
        logger.error("Inference module not available. Please ensure inference.py is in the same directory.")
        return None
    
    # Use the most recent model if not specified
    if model_path is None:
        # Find most recent epoch directory
        try:
            epoch_dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("epoch-")]
            if epoch_dirs:
                epoch_nums = [int(d.split("-")[1]) for d in epoch_dirs]
                latest_epoch = max(epoch_nums)
                model_path = os.path.join(OUTPUT_DIR, f"epoch-{latest_epoch}")
                logger.info(f"Using most recent model: {model_path}")
            else:
                # No epoch directories found, use the base OUTPUT_DIR
                model_path = OUTPUT_DIR
                logger.info(f"Using base model directory: {model_path}")
        except Exception as e:
            logger.warning(f"Error finding latest model, falling back to base directory: {str(e)}")
            model_path = OUTPUT_DIR
    
    # Determine device for inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running inference on device: {device}")
    
    try:
        logger.info(f"Loading adapted model from {model_path}...")
        model, tokenizer = load_adapted_model(model_path, device=device)
        
        # Task 1: Generate function summary if code is provided
        if function_code:
            logger.info("Generating function summary...")
            summary = generate_function_summary(model, tokenizer, function_code)
            results["function_summary"] = summary
            
            # If requirement is also provided, use the generated summary for matching
            if requirement:
                function_summary = summary
        
        # Task 2: Predict requirement match if both requirement and summary are provided
        if requirement and function_summary:
            logger.info("Predicting requirement match...")
            prediction = predict_requirement_match(model, tokenizer, requirement, function_summary)
            results["match_prediction"] = prediction
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    return results

def main():
    """Main entry point for the inference tool."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Requirement Tracing Inference Tool")
    parser.add_argument("--model-path", type=str, help="Path to the model with LoRA adapters for inference")
    parser.add_argument("--requirement", type=str, help="Requirement text for matching")
    parser.add_argument("--function-code", type=str, help="Function code to summarize")
    parser.add_argument("--function-summary", type=str, help="Function summary for matching")
    parser.add_argument("--input-file", type=str, help="Path to a file containing function code")
    
    args = parser.parse_args()
    
    # If an input file is specified, read function code from it
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                args.function_code = f.read()
            logger.info(f"Loaded function code from {args.input_file}")
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            return False
    
    # Validate inputs
    if not args.function_code and not (args.requirement and args.function_summary):
        logger.error("You must provide either:")
        logger.error("1. Function code (--function-code or --input-file) for summarization, or")
        logger.error("2. Both requirement (--requirement) and function summary (--function-summary) for matching")
        return False
    
    # Run inference
    results = run_inference(
        model_path=args.model_path,
        requirement=args.requirement,
        function_code=args.function_code,
        function_summary=args.function_summary
    )
    
    if not results:
        logger.error("Inference failed.")
        return False
        
    # Display results in a nice format
    print("\n===== Inference Results =====")
    
    if "function_summary" in results:
        print("\nGenerated Function Summary:")
        print(f"{results['function_summary']}")
    
    if "match_prediction" in results:
        print(f"\nRequirement Match Rating: {results['match_prediction']['rating']}/100")
        # Check if reasoning key exists to avoid KeyError
        if 'reasoning' in results['match_prediction']:
            print(f"Reasoning: {results['match_prediction']['reasoning']}")
        else:
            # Try to get generated text if available
            if 'generated_text' in results['match_prediction']:
                print(f"Generated Text: {results['match_prediction']['generated_text']}")
        
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
