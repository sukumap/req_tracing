#!/usr/bin/env python3
"""
Simplified Gemma Fine-Tuning with LoRA - Basic version for debugging.

This script implements a simplified version of LoRA fine-tuning for the Gemma model
to help diagnose and fix training issues.
"""

import os
import re
import json
import random
import logging
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

# Basic Logging Setup
def setup_logging():
    """Sets up basic logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logging.getLogger(__name__)

logger = setup_logging()

# Constants
TRAINING_DATA_PATH = "training_data.jsonl"
MAX_EXAMPLES = 500  # Increased for better training
SAVE_DIR = "gemma_lora_weights"
MATCH_THRESHOLD = 70  # Scores >= 70 are considered correct matches

# GPU Check
def check_gpu():
    """Checks GPU availability and prints details."""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available.")
        return False
    
    logger.info(f"CUDA Version: {torch.version.cuda}")
    logger.info(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Test tensor allocation
    try:
        tensor = torch.tensor([1.0, 2.0]).cuda()
        logger.info(f"Test tensor on {tensor.device}: {tensor}")
        del tensor
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logger.error(f"CUDA test failed: {e}")
        return False

# Load Training Data
def analyze_training_data(data):
    """Analyze training data to understand score distributions."""
    confidence_scores = []
    expected_scores = []
    is_correct_counts = {True: 0, False: 0}
    
    for item in data:
        if "confidence_score" in item:
            try:
                confidence_scores.append(float(item["confidence_score"]))
            except (ValueError, TypeError):
                pass
                
        if "expected_score" in item:
            try:
                expected_scores.append(float(item["expected_score"]))
            except (ValueError, TypeError):
                pass
                
        is_correct = item.get("is_correct", None)
        if is_correct is not None:
            is_correct_counts[bool(is_correct)] += 1
    
    logger.info("=== Training Data Analysis ===")
    
    if confidence_scores:
        logger.info(f"Confidence scores: {len(confidence_scores)} values")
        logger.info(f"  Range: {min(confidence_scores):.2f} to {max(confidence_scores):.2f}")
        logger.info(f"  Mean: {sum(confidence_scores)/len(confidence_scores):.2f}")
    
    if expected_scores:
        logger.info(f"Expected scores: {len(expected_scores)} values")
        logger.info(f"  Range: {min(expected_scores):.2f} to {max(expected_scores):.2f}")
        logger.info(f"  Mean: {sum(expected_scores)/len(expected_scores):.2f}")
    
    logger.info(f"Is correct: {is_correct_counts[True]} positive, {is_correct_counts[False]} negative")
    logger.info("===========================")

def load_training_data(max_examples=MAX_EXAMPLES):
    """Load training data from the JSONL file."""
    # Try different data files
    data_files = [
        TRAINING_DATA_PATH,
        "training_data_balanced.jsonl",
        "training_data_500_balanced_5050.jsonl"
    ]
    
    data_file = None
    for file in data_files:
        if os.path.exists(file):
            data_file = file
            logger.info(f"Using data file: {file}")
            break
    
    if data_file is None:
        logger.error(f"No training data file found!")
        return None
    
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(data)} examples")
    
    # Analyze the training data
    analyze_training_data(data)
    
    # Limit examples for debugging
    if max_examples and len(data) > max_examples:
        random.shuffle(data)
        data = data[:max_examples]
        logger.info(f"Using {len(data)} examples for debugging")
    
    return data

# Create prompt
def create_prompt(requirement, function_summary):
    """Create the scoring prompt."""
    return (
        f"Rate how well the function summary implements the given requirement on a scale of 0 to 100.\n\n"
        f"Requirement: {requirement}\n\n"
        f"Function Summary: {function_summary}\n\n"
        f"Scoring Guide:\n"
        f"- 0-10: No relationship\n"
        f"- 11-30: Slightly related, incorrect implementation\n"
        f"- 31-60: Partially implements, significant gaps\n"
        f"- 61-89: Mostly implements, minor issues\n"
        f"- 90-100: Correctly and completely implements\n\n"
        f"Provide ONLY the numerical score (0-100):"
    )

# Parse score from text
def parse_score(text):
    """Parse numerical score from text with improved accuracy."""
    # First try to find a clean number at the start of the text
    match = re.search(r'^\s*(\d{1,3})\s*$', text.strip())
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score
    
    # Next, try to find any number in the text
    numbers = re.findall(r'\b(\d{1,3})\b', text)
    for num in numbers:
        score = int(num)
        if 0 <= score <= 100:
            return score
    
    # If no valid score found, return a default value
    return 50  # Default to middle value instead of 0

# Simple Dataset
class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for item in data:
            # Get requirement and summary
            requirement = item.get("requirement_text", "")
            if not requirement and "requirement" in item:
                requirement = item["requirement"]
                
            summary = item.get("function_summary", "")
            if not summary and "summary" in item:
                summary = item["summary"]
            
            # Get target score
            if "confidence_score" in item:
                # Convert confidence score (0-1) to a score (0-100)
                # Make sure to convert to float first in case it's stored as string
                confidence = float(item["confidence_score"])
                score = int(confidence * 100)
                # Ensure score is within 0-100 range
                score = max(0, min(100, score))
            elif "expected_score" in item:
                # Ensure expected_score is within 0-100 range
                try:
                    score = float(item["expected_score"])
                    # If score is too high, it might be on a different scale
                    if score > 100:
                        # Normalize to 0-100 range assuming it might be on a 0-10000 scale
                        score = min(100, score / 100)
                    score = int(score)
                except (ValueError, TypeError):
                    # Default if conversion fails
                    score = 90 if item.get("is_correct", False) else 20
            else:
                # Default based on is_correct
                score = 90 if item.get("is_correct", False) else 20
                
            # Final sanity check to ensure score is in range 0-100
            score = max(0, min(100, score))
            
            # Create prompt
            prompt = create_prompt(requirement, summary)
            
            # Create example
            self.examples.append({
                "prompt": prompt,
                "score": score,
                "is_correct": item.get("is_correct", score >= MATCH_THRESHOLD)
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        inputs = self.tokenizer(
            example["prompt"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Add score as a label
        inputs["score"] = torch.tensor(example["score"], dtype=torch.float)
        inputs["is_correct"] = example["is_correct"]
        
        return inputs

def train(args):
    """Train the model with LoRA."""
    # Set random seed
    set_seed(args.seed)
    
    # Check GPU
    has_gpu = check_gpu()
    device = torch.device("cuda" if has_gpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    data = load_training_data(args.max_examples)
    if not data:
        logger.error("No data loaded. Exiting.")
        return
    
    # Split data
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=args.seed
    )
    logger.info(f"Training set: {len(train_data)} examples")
    logger.info(f"Validation set: {len(val_data)} examples")
    
    # Load model and tokenizer
    model_id = f"google/gemma-3-{args.model_size}-it"
    logger.info(f"Loading model: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with standard settings
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if has_gpu else torch.float32,
            device_map="auto" if has_gpu else None
        )
        
        # Enable gradient checkpointing to save memory
        if has_gpu:
            model.gradient_checkpointing_enable()
        
        # Disable KV cache for training
        model.config.use_cache = False
        
        logger.info(f"Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Configure LoRA
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Create datasets
    train_dataset = SimpleDataset(train_data, tokenizer)
    val_dataset = SimpleDataset(val_data, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Check if any parameters are trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.error("No trainable parameters found!")
        return
    
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Create scheduler
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"]  # Use input_ids as labels for causal LM
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Log progress
            if (batch_idx + 1) % 5 == 0:
                logger.info(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            # Clear cache
            if has_gpu and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"]
                )
                
                val_loss += outputs.loss.item()
                
                # Generate predictions
                model.config.use_cache = True  # Enable for generation
                
                for i in range(len(batch["input_ids"])):
                    input_ids = batch["input_ids"][i].unsqueeze(0)
                    attn_mask = batch["attention_mask"][i].unsqueeze(0)
                    
                    gen_output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=1.0,  # Deterministic generation
                        num_beams=1,      # Greedy decoding
                        top_p=None,       # Disable top-p filtering
                        top_k=None        # Disable top-k filtering
                    )
                    
                    # Extract generated text
                    gen_text = tokenizer.decode(gen_output[0][input_ids.shape[1]:], skip_special_tokens=True)
                    
                    # Parse score
                    pred_score = parse_score(gen_text)
                    target_score = batch["score"][i].item()
                    
                    all_preds.append(pred_score)
                    all_targets.append(target_score)
                
                model.config.use_cache = False  # Disable for training
        
        # Calculate metrics
        val_loss /= len(val_loader)
        mae = mean_absolute_error(all_targets, all_preds)
        
        # Binary classification metrics
        pred_binary = [score >= MATCH_THRESHOLD for score in all_preds]
        target_binary = [score >= MATCH_THRESHOLD for score in all_targets]
        accuracy = accuracy_score(target_binary, pred_binary)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean([(p - t) ** 2 for p, t in zip(all_preds, all_targets)]))
        
        # Calculate correlation
        correlation = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0
        
        # Log detailed metrics
        logger.info(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        logger.info(f"Accuracy: {accuracy:.4f} | Correlation: {correlation:.4f}")
        
        # Log some example predictions
        logger.info("Example predictions:")
        for i in range(min(5, len(all_preds))):
            logger.info(f"  Predicted: {all_preds[i]}, Target: {all_targets[i]}")
        
        # Save model
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"epoch_{epoch+1}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simplified Gemma Fine-Tuning")
    
    # Model configuration
    parser.add_argument("--model-size", type=str, choices=["1b", "4b"], default="4b", 
                        help="Gemma model size (1b or 4b)")
    
    # Training configuration
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES, 
                        help=f"Maximum number of examples to use (default: {MAX_EXAMPLES})")
    parser.add_argument("--batch-size", type=int, default=4, 
                        help="Batch size (default: 4)")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of epochs (default: 5)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, 
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--lora-rank", type=int, default=16, 
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, 
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, 
                        help=f"Directory to save model (default: {SAVE_DIR})")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    logger.info("==== Starting Simplified Gemma LoRA Fine-Tuning ====")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    
    train(args)

if __name__ == "__main__":
    main()
