"""
Model Trainer module for fine-tuning the Gemma model with reinforcement learning
to better adapt to specific requirements-to-code mapping tasks.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import ollama
import subprocess
import tempfile
import shutil
from pathlib import Path

from src.utils.logger import get_logger
from src.cpp_parser import CppFunction

logger = get_logger()

class ModelTrainer:
    """
    Trains the Gemma model using reinforcement learning from human feedback (RLHF)
    to improve its performance on domain-specific requirements-to-code mapping.
    
    This class handles:
    1. Collecting feedback data (correct/incorrect mappings)
    2. Preparing training datasets
    3. Fine-tuning the model using LoRA (Low-Rank Adaptation)
    4. Exporting and deploying the fine-tuned model to Ollama
    """
    
    def __init__(self, 
                base_model: str = "gemma3:1b", 
                fine_tuned_model: str = "gemma3-req-tracer:latest",
                training_data_path: str = "training_data.jsonl",
                lora_weights_dir: str = "lora_weights",
                epochs: int = 3,
                learning_rate: float = 2e-5,
                batch_size: int = 4):
        """
        Initialize the model trainer.
        
        Args:
            base_model (str): Name of the base model in Ollama
            fine_tuned_model (str): Name for the fine-tuned model
            training_data_path (str): Path to store training data
            lora_weights_dir (str): Directory to store LoRA weights
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for training
            batch_size (int): Batch size for training
        """
        self.base_model = base_model
        self.fine_tuned_model = fine_tuned_model
        self.training_data_path = training_data_path
        self.lora_weights_dir = lora_weights_dir
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Create directories if they don't exist
        os.makedirs(self.lora_weights_dir, exist_ok=True)
        
        # Training data storage
        self.feedback_data = []
        self._load_feedback_data()
        
        logger.info(f"Initialized ModelTrainer for fine-tuning {base_model}")
    
    def _load_feedback_data(self):
        """Load existing feedback data if available."""
        if os.path.exists(self.training_data_path):
            try:
                with open(self.training_data_path, 'r') as f:
                    self.feedback_data = [json.loads(line) for line in f]
                logger.info(f"Loaded {len(self.feedback_data)} feedback examples")
            except Exception as e:
                logger.error(f"Error loading feedback data: {str(e)}")
                self.feedback_data = []
    
    def _save_feedback_data(self):
        """Save feedback data to disk."""
        try:
            with open(self.training_data_path, 'w') as f:
                for item in self.feedback_data:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Saved {len(self.feedback_data)} feedback examples")
        except Exception as e:
            logger.error(f"Error saving feedback data: {str(e)}")
    
    def add_mapping_feedback(self, 
                           requirement_id: str,
                           requirement_text: str,
                           function_name: str,
                           function_summary: str,
                           function_content: str,
                           is_correct: bool,
                           confidence_score: Optional[float] = None):
        """
        Add feedback for a requirement-to-function mapping.
        
        Args:
            requirement_id (str): The requirement ID
            requirement_text (str): The requirement text
            function_name (str): The function's qualified name
            function_summary (str): The function summary
            function_content (str): The function content
            is_correct (bool): Whether this mapping is correct
            confidence_score (float, optional): The model's original confidence score
        """
        feedback_item = {
            "requirement_id": requirement_id,
            "requirement_text": requirement_text,
            "function_name": function_name,
            "function_summary": function_summary,
            "function_content": function_content,
            "is_correct": is_correct,
            "confidence_score": confidence_score,
            "prompt": self._create_training_prompt(requirement_text, function_summary)
        }
        
        # Add reward value based on correctness
        # Higher reward for correctly identified mappings (positive examples)
        # Lower/negative reward for incorrect mappings (negative examples)
        feedback_item["reward"] = 1.0 if is_correct else -1.0
        
        self.feedback_data.append(feedback_item)
        self._save_feedback_data()
        logger.info(f"Added feedback for requirement {requirement_id} and function {function_name}")
    
    def _create_training_prompt(self, requirement_text: str, function_summary: str) -> str:
        """
        Create a prompt for training the model.
        
        Args:
            requirement_text (str): The requirement text
            function_summary (str): The function summary
            
        Returns:
            str: Formatted prompt for training
        """
        return f"""Given the following requirement:

Requirement: {requirement_text}

And this function summary:

Function Summary: {function_summary}

On a scale of 0 to 100, rate how confident you are that this function implements the requirement.
Only return a single integer number between 0 and 100, where 0 means no relationship and 100 means perfect match.
"""
    
    def prepare_training_data(self) -> str:
        """
        Prepare training data in the format required for fine-tuning.
        
        Returns:
            str: Path to the prepared training data file
        """
        if not self.feedback_data:
            logger.warning("No feedback data available for training")
            return None
        
        logger.info(f"Preparing training data from {len(self.feedback_data)} feedback examples")
        
        # Create a temporary directory for formatted training data
        temp_dir = tempfile.mkdtemp()
        train_data_path = os.path.join(temp_dir, "train.jsonl")
        
        try:
            # Format data for RLHF training
            formatted_data = []
            for item in self.feedback_data:
                # For reinforcement learning, we need prompts and their associated rewards
                formatted_item = {
                    "prompt": item["prompt"],
                    "completion": str(int(item["reward"] * 100)) if item["is_correct"] else str(0),
                    "reward": float(item["reward"])
                }
                formatted_data.append(formatted_item)
            
            # Write to JSONL file
            with open(train_data_path, 'w') as f:
                for item in formatted_data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Prepared training data saved to {train_data_path}")
            return train_data_path
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            shutil.rmtree(temp_dir)
            return None
    
    def train_model(self) -> bool:
        """
        Fine-tune the model using LoRA and reinforcement learning.
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        # Prepare training data
        train_data_path = self.prepare_training_data()
        if not train_data_path:
            logger.error("Cannot train model: No training data available")
            return False
        
        logger.info(f"Starting model fine-tuning with {self.epochs} epochs")
        
        try:
            # Get the HuggingFace model ID corresponding to the Ollama model
            model_id = self._extract_base_model()
            if not model_id:
                return False
            
            # Set up fine-tuning environment
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU if available
            
            # Check for GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                    logger.info(f"CUDA Version: {torch.version.cuda}")
                else:
                    logger.warning("No GPU detected. Training will be slow on CPU.")
            except ImportError:
                logger.warning("Could not check GPU availability. Make sure PyTorch is installed.")
            
            # Run fine-tuning with HuggingFace Transformers
            trained_model_dir = self._run_fine_tuning(model_id, train_data_path)
            if not trained_model_dir:
                logger.error("Fine-tuning failed")
                return False
            
            # Create and register the fine-tuned model with Ollama
            success = self._register_fine_tuned_model(trained_model_dir)
            
            if success:
                logger.info(f"Model {self.fine_tuned_model} is now available for use with Ollama")
                logger.info(f"You can use it in the MappingEngine by setting model='{self.fine_tuned_model}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _extract_base_model(self) -> Optional[str]:
        """
        Get the HuggingFace model ID corresponding to the Ollama model.
        Gemma is available on HuggingFace, so we can use it directly.
        
        Returns:
            Optional[str]: HuggingFace model ID corresponding to the Ollama model
        """
        try:
            # Map from Ollama model names to open-source HuggingFace model IDs
            model_mapping = {
                "gemma3:1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Alternative to Gemma 1B
                "gemma3:2b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Same alternative
                "gemma3:7b": "mistralai/Mistral-7B-v0.1",           # Larger alternative
                "gemma2:2b": "microsoft/phi-2",                      # Good 2B alternative
                "gemma2:9b": "mistralai/Mistral-7B-v0.1",           # Medium alternative
                "gemma2:27b": "mistralai/Mistral-7B-v0.1"            # Smaller but open
            }
            
            # Check if model is in our mapping
            if self.base_model in model_mapping:
                hf_model_id = model_mapping[self.base_model]
                logger.info(f"Mapped Ollama model {self.base_model} to HuggingFace ID {hf_model_id}")
                return hf_model_id
            else:
                # For other models, try to use the same name
                logger.warning(f"No direct mapping for {self.base_model}, using as-is")
                return self.base_model
            
        except Exception as e:
            logger.error(f"Error mapping model: {str(e)}")
            return None
    
    def _run_fine_tuning(self, model_id: str, train_data_path: str):
        """
        Run the LoRA fine-tuning process with RLHF using the transformers library.
        
        Args:
            model_id (str): HuggingFace model ID
            train_data_path (str): Path to the training data
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from datasets import load_dataset
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from trl import SFTTrainer, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
            from trl import RewardTrainer, RewardConfig, TextEnvironment

            logger.info(f"Starting LoRA fine-tuning process for model {model_id}")
            
            # 1. Load the model and tokenizer
            logger.info("Loading base model and tokenizer")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.padding_side = "right"  # For better batch processing
            
            # 2. Configure LoRA
            logger.info("Configuring LoRA adapters")
            lora_config = LoraConfig(
                r=16,                # Rank of the LoRA matrices
                lora_alpha=32,       # LoRA alpha parameter
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Prepare model for LoRA training
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            
            # 3. Load and format the dataset
            logger.info("Loading training dataset")
            raw_dataset = load_dataset("json", data_files=train_data_path, split="train")
            
            # 4. First phase: Supervised Fine-Tuning (SFT)
            logger.info("Starting Supervised Fine-Tuning phase")
            training_args = TrainingArguments(
                output_dir=os.path.join(self.lora_weights_dir, "sft"),
                num_train_epochs=1,             # Limited SFT before RLHF
                per_device_train_batch_size=self.batch_size,
                gradient_accumulation_steps=4,  # Effective batch size = batch_size * grad_accum
                gradient_checkpointing=True,     # Save memory
                learning_rate=self.learning_rate,
                logging_steps=10,
                save_strategy="epoch",
                optim="paged_adamw_8bit",        # Memory-efficient optimizer
                fp16=True,                      # Mixed precision training
            )
                        
            # In newer TRL versions, we need to pass lora_config as peft_config
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=raw_dataset,
                peft_config=lora_config,
                dataset_text_field="prompt",
                max_seq_length=512
            )
            
            logger.info("Running SFT training phase")
            trainer.train()
            sft_model_dir = os.path.join(self.lora_weights_dir, "sft")
            trainer.save_model(sft_model_dir)
            logger.info(f"SFT model saved to {sft_model_dir}")
            
            # 5. Second phase: Train a reward model
            logger.info("Starting Reward Model training phase")
            reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_dir)
            
            reward_config = RewardConfig(
                output_dir=os.path.join(self.lora_weights_dir, "reward_model"),
                per_device_train_batch_size=self.batch_size // 2,  # Reward training is more memory intensive
                num_train_epochs=1,
                learning_rate=self.learning_rate / 2,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
            )
            
            # Prepare reward training dataset
            def process_reward_dataset(example):
                example["input"] = example["prompt"]
                example["value"] = float(example["reward"])
                return example
                
            reward_dataset = raw_dataset.map(process_reward_dataset)
            
            reward_trainer = RewardTrainer(
                model=reward_model,
                tokenizer=tokenizer,
                args=reward_config,
                train_dataset=reward_dataset,
            )
            
            logger.info("Running Reward Model training")
            reward_trainer.train()
            reward_model_dir = os.path.join(self.lora_weights_dir, "reward_model")
            reward_trainer.save_model(reward_model_dir)
            logger.info(f"Reward model saved to {reward_model_dir}")
            
            # 6. Third phase: PPO training (actual RLHF)
            logger.info("Starting PPO training phase (RLHF)")
            # Load saved SFT model as the starting policy model
            policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_dir)
            # Load the reward model
            reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(reward_model_dir)
            
            ppo_config = PPOConfig(
                model_name=sft_model_dir,
                learning_rate=self.learning_rate / 4,
                ppo_epochs=self.epochs,
                mini_batch_size=1,  # Process examples one by one for memory efficiency
                batch_size=self.batch_size // 4,
                gradient_accumulation_steps=8,
                optimize_device_cache=True,
                target_kl=0.1,  # KL penalty to prevent drift from original model
            )
            
            # Create a PPO trainer
            ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=policy_model,
                ref_model=None,  # We use KL penalty instead of a separate reference model
                tokenizer=tokenizer,
                reward_model=reward_model,
            )
            
            # Set up a simple text environment
            env = TextEnvironment(raw_dataset, tokenizer)
            
            # PPO Training loop
            logger.info("Running PPO training loop")
            for epoch in range(self.epochs):
                logger.info(f"Starting PPO epoch {epoch+1}/{self.epochs}")
                for batch_idx, batch in enumerate(env.get_batch()):
                    # Generate responses from the policy model
                    queries = batch["prompt"]
                    responses = ppo_trainer.generate(queries, max_new_tokens=20)
                    
                    # Compute rewards using the reward model
                    rewards = []
                    for query, response in zip(queries, responses):
                        # Combine prompt and response
                        full_text = query + response
                        # Get reward prediction from the reward model
                        reward = ppo_trainer.compute_reward(full_text)
                        rewards.append(reward)
                    
                    # Update the policy with PPO
                    stats = ppo_trainer.step(queries, responses, rewards)
                    logger.info(f"Batch {batch_idx+1}: mean reward = {stats['ppo/mean_reward']}")
                
                # Save checkpoint after each epoch
                ppo_checkpoint_dir = os.path.join(self.lora_weights_dir, f"ppo_epoch_{epoch+1}")
                ppo_trainer.save_pretrained(ppo_checkpoint_dir)
                logger.info(f"Saved PPO checkpoint to {ppo_checkpoint_dir}")
            
            # Final model path is the last PPO checkpoint
            final_model_dir = os.path.join(self.lora_weights_dir, f"ppo_epoch_{self.epochs}")
            logger.info(f"Fine-tuning complete! Final model saved to {final_model_dir}")
            
            return final_model_dir
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            # Print full exception traceback for debugging
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _register_fine_tuned_model(self, trained_model_dir: str) -> bool:
        """
        Register the fine-tuned model with Ollama by creating a Modelfile.
        
        Args:
            trained_model_dir (str): Path to the directory containing the trained LoRA weights
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        try:
            logger.info(f"Registering fine-tuned model as {self.fine_tuned_model}")
            
            # Create a directory for the Ollama model
            ollama_model_dir = os.path.join(os.path.expanduser("~"), ".ollama", "models", self.fine_tuned_model)
            os.makedirs(ollama_model_dir, exist_ok=True)
            
            # Copy the LoRA weights to a subdirectory within the Ollama model directory
            lora_adapter_dir = os.path.join(ollama_model_dir, "adapter")
            if os.path.exists(lora_adapter_dir):
                shutil.rmtree(lora_adapter_dir)
            shutil.copytree(trained_model_dir, lora_adapter_dir)
            
            # Create a modelfile for Ollama
            modelfile_content = f"""
FROM {self.base_model}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER stop "\n"

# Path to the LoRA adapter weights relative to the model directory
ADAPTER adapter

# Add a system message about the fine-tuning
SYSTEM You are a fine-tuned version of {self.base_model} that has been trained specifically for requirement-to-code mapping. You analyze requirements and determine their relationship to code implementations.
"""
            
            # Write the modelfile
            modelfile_path = os.path.join(ollama_model_dir, "Modelfile")
            with open(modelfile_path, 'w') as f:
                f.write(modelfile_content)
            
            # Register with Ollama using the CLI
            logger.info("Calling Ollama CLI to register the model")
            result = subprocess.run(
                ["ollama", "create", self.fine_tuned_model, "-f", modelfile_path],
                capture_output=True,
                text=True,
                check=False  # Don't raise exception, we'll handle errors
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully registered fine-tuned model: {self.fine_tuned_model}")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"Error registering model with Ollama: {result.stderr}")
                return False
            
        except Exception as e:
            logger.error(f"Error registering fine-tuned model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def evaluate_model(self, test_requirements, test_functions, function_summaries):
        """
        Evaluate the fine-tuned model on a test set.
        
        Args:
            test_requirements: List of requirements for testing
            test_functions: List of functions for testing
            function_summaries: Function summaries for testing
            
        Returns:
            dict: Evaluation metrics
        """
        # This is a simplified evaluation - in a real implementation, 
        # we would perform a comprehensive evaluation
        
        logger.info("Evaluating fine-tuned model performance")
        
        # Example metrics that would be calculated
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "accuracy": 0.0
        }
        
        return metrics
    
    def get_feedback_statistics(self) -> Dict:
        """
        Get statistics about the collected feedback data.
        
        Returns:
            Dict: Statistics about the feedback data
        """
        if not self.feedback_data:
            return {"total": 0, "positive": 0, "negative": 0}
        
        positive = sum(1 for item in self.feedback_data if item["is_correct"])
        negative = len(self.feedback_data) - positive
        
        return {
            "total": len(self.feedback_data),
            "positive": positive,
            "negative": negative,
            "positive_percentage": (positive / len(self.feedback_data)) * 100,
            "negative_percentage": (negative / len(self.feedback_data)) * 100,
        }