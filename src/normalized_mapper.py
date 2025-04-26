"""
Normalized TinyLlama Mapper that extends the TinyLlamaMapper with score normalization.

This implementation addresses the overconfidence bias in the fine-tuned TinyLlama model
by normalizing scores across multiple functions for the same requirement.
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from src.cpp_parser import CppFunction
from src.tinyllama_mapper import TinyLlamaMapper
from src.utils.logger import get_logger

logger = get_logger()

class NormalizedMapper(TinyLlamaMapper):
    """
    Extension of TinyLlamaMapper that applies score normalization to address
    the model's tendency to assign high confidence scores to all functions.
    """
    
    def __init__(self, similarity_threshold=0.5, model_path=None, strict_mode=False):
        """
        Initialize the normalized mapper.
        
        Args:
            similarity_threshold (float): Threshold for considering a match
            model_path (str): Path to the trained TinyLlama model with LoRA weights
            strict_mode (bool): If True, never fall back to Ollama even if model fails to load
        """
        super().__init__(similarity_threshold, model_path, strict_mode)
        
        # Used to store raw scores for normalization
        self.raw_scores = defaultdict(dict)  # {requirement_id: {function_id: score}}
        
        logger.info("Using normalized mapper with score calibration")
    
    def get_normalized_confidence(self, raw_score: float, requirement_id: str, function_id: str) -> float:
        """
        Store raw score for later normalization and return a temporary value.
        
        Args:
            raw_score (float): Raw confidence score from model
            requirement_id (str): Identifier for the requirement
            function_id (str): Identifier for the function
            
        Returns:
            float: Temporary score that will be updated during finalization
        """
        # Store raw score for normalization
        self.raw_scores[requirement_id][function_id] = raw_score
        
        # Return the raw score for now - it will be normalized later
        return raw_score
    
    def normalize_scores(self):
        """Normalize scores for all requirements to address overconfidence bias."""
        logger.info(f"Normalizing confidence scores for {len(self.raw_scores)} requirements")
        
        for req_id, function_scores in self.raw_scores.items():
            if not function_scores:
                continue
                
            scores = list(function_scores.values())
            
            # Skip normalization if all scores are identical
            if len(set(scores)) <= 1:
                logger.warning(f"All scores for requirement {req_id} are identical: {scores[0]}")
                continue
                
            # Get statistics
            min_score = min(scores)
            max_score = max(scores)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            logger.info(f"Requirement {req_id} scores - Min: {min_score:.2f}, Max: {max_score:.2f}, "
                      f"Mean: {mean_score:.2f}, StdDev: {std_score:.2f}")
            
            # Apply z-score normalization and sigmoid scaling
            if std_score > 0:
                for func_id in function_scores:
                    # Z-score normalization: (x - mean) / std
                    z_score = (function_scores[func_id] - mean_score) / std_score
                    
                    # Apply sigmoid to map from (-inf, inf) to (0, 1)
                    # This gives a well-distributed range of values
                    normalized_score = 1.0 / (1.0 + np.exp(-z_score))
                    
                    # Store back the normalized score
                    function_scores[func_id] = normalized_score
            
            # Log the normalization results
            new_scores = list(function_scores.values())
            logger.info(f"After normalization - Min: {min(new_scores):.2f}, "
                      f"Max: {max(new_scores):.2f}, Mean: {np.mean(new_scores):.2f}")
    
    def get_llm_confidence(self, requirement_desc: str, module_id: str, function_name: str, 
                         function_summary: str, function_content: str) -> float:
        """
        Override the parent method to add score normalization.
        
        Args:
            requirement_desc (str): The requirement description
            module_id (str): The module identifier
            function_name (str): The name of the function
            function_summary (str): The summary of the function
            function_content (str): The content of the function
            
        Returns:
            float: Normalized confidence score between 0.0 and 1.0
        """
        # Get the raw confidence score from the parent method
        raw_confidence = super().get_llm_confidence(
            requirement_desc, module_id, function_name, function_summary, function_content
        )
        
        # This will be used as the function identifier
        function_id = f"{module_id}::{function_name}"
        
        # Store for later normalization
        # We're using the requirement description as the requirement ID since we don't have the ID here
        normalized_confidence = self.get_normalized_confidence(
            raw_confidence, requirement_desc, function_id
        )
        
        return normalized_confidence
    
    def map_requirement(self, requirement_id: str, requirement_desc: str, functions: List[CppFunction]) -> Dict[str, float]:
        """
        Map a requirement to matching functions with normalized confidence scores.
        
        Args:
            requirement_id (str): The requirement ID
            requirement_desc (str): The requirement description
            functions (List[CppFunction]): List of functions to check for matches
            
        Returns:
            Dict[str, float]: Dictionary mapping function IDs to confidence scores
        """
        # Clear any previous scores for this requirement
        self.raw_scores[requirement_id] = {}
        
        # Run the standard mapping to get raw scores
        matches = super().map_requirement(requirement_id, requirement_desc, functions)
        
        # Now normalize the scores for this requirement
        self.normalize_scores()
        
        # Return the normalized matches
        normalized_matches = {}
        for func_id, _ in matches.items():
            # Use normalized score instead of raw score
            function_key = func_id.split(":")[-1]  # Extract function name from ID
            for stored_func_id in self.raw_scores[requirement_id]:
                if function_key in stored_func_id:
                    normalized_matches[func_id] = self.raw_scores[requirement_id][stored_func_id]
                    break
        
        # Sort by confidence (descending)
        sorted_matches = {k: v for k, v in sorted(
            normalized_matches.items(), key=lambda x: x[1], reverse=True
        )}
        
        logger.info(f"Normalized {len(matches)} raw matches to {len(sorted_matches)} normalized matches")
        return sorted_matches
