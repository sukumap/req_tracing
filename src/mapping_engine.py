"""
Mapping engine for comparing requirements with function summaries
and determining the best matches.
"""
import numpy as np
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

from src.cpp_parser import CppFunction
from src.utils.logger import get_logger

logger = get_logger()

class MappingEngine:
    """
    Maps requirements to functions by comparing their descriptions
    with function summaries using semantic similarity and LLM judgment.
    """
    
    def __init__(self, similarity_threshold=0.5, model="gemma3:1b"):
        """
        Initialize the mapping engine.
        
        Args:
            similarity_threshold (float): Threshold for considering a match
            model (str): The Ollama model to use for LLM judgment
        """
        self.similarity_threshold = similarity_threshold
        self.model = model
        self.vectorizer = TfidfVectorizer(stop_words='english')
        logger.info(f"Initialized mapping engine with threshold: {similarity_threshold}")
    
    def compute_similarity(self, requirement_desc: str, function_summaries: Dict[str, str]) -> Dict[str, float]:
        """
        Compute semantic similarity between a requirement and multiple function summaries.
        
        Args:
            requirement_desc (str): The requirement description
            function_summaries (Dict[str, str]): Map of function names to summaries
            
        Returns:
            Dict[str, float]: Dictionary mapping function names to similarity scores
        """
        if not function_summaries:
            return {}
        
        # Combine requirement and all summaries
        texts = [requirement_desc] + list(function_summaries.values())
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Compute cosine similarity between requirement and each summary
            req_vector = tfidf_matrix[0:1]
            summary_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(req_vector, summary_vectors)[0]
            
            # Map similarity scores to function names
            result = {}
            for i, func_name in enumerate(function_summaries.keys()):
                result[func_name] = float(similarities[i])
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            # Return empty dict on error
            return {}
    
    def get_llm_confidence(self, requirement_desc: str, module_id: str, function_name: str, 
                          function_summary: str, function_content: str) -> float:
        """
        Use the LLM to determine confidence in matching a requirement to a function.
        Only uses requirement description and function summary for comparison.
        
        Args:
            requirement_desc (str): The requirement description
            module_id (str): Not used in simplified implementation
            function_name (str): Not used in simplified implementation 
            function_summary (str): The summary of the function
            function_content (str): Not used in simplified implementation
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        # Construct a simplified prompt with just requirement and function summary
        prompt = f"""Given the following requirement:

Requirement: {requirement_desc}

And this function summary:

Function Summary: {function_summary}

On a scale of 0 to 100, rate how confident you are that this function implements the requirement.
Only return a single integer number between 0 and 100, where 0 means no relationship and 100 means perfect match.
"""
        
        try:
            # Call the Ollama API
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            
            confidence_text = response["message"]["content"].strip()
            
            # Extract numeric value
            confidence = 0.0
            for word in confidence_text.split():
                try:
                    value = float(word)
                    if 0 <= value <= 100:
                        confidence = value / 100.0  # Normalize to 0-1
                        break
                except ValueError:
                    continue
            
            logger.debug(f"LLM confidence for '{function_name}': {confidence:.2f}")
            return confidence
            
        except Exception as e:
            logger.error(f"Error getting LLM confidence: {str(e)}")
            return 0.0
    
    def map_requirement(self, req_id: str, req_desc: str, module_id: str,
                       functions: List[CppFunction],
                       function_summaries: Dict[str, str],
                       mapping_method: str = "all") -> List[Tuple[str, float, float]]:
        """
        Map a single requirement to functions using the specified mapping method.
        
        Args:
            req_id (str): The requirement identifier
            req_desc (str): The requirement description
            module_id (str): The module identifier
            functions (List[CppFunction]): List of all available functions
            function_summaries (Dict[str, str]): Map of function names to summaries
            mapping_method (str): Method to use for mapping: "explicit", "semantic", "llm", 
                                 "semantic+llm", or "all"
            
        Returns:
            List[Tuple[str, float, float]]: List of (function_name, similarity, llm_confidence) tuples
        """
        logger.info(f"Mapping requirement {req_id} using method: {mapping_method}")
        
        # Check for explicit requirement annotations if method is "explicit" or "all"
        if mapping_method in ["explicit", "all"]:
            explicit_matches = []
            for function in functions:
                if req_id in function.requirements:
                    # This function explicitly declares it implements this requirement
                    # Add it with high confidence scores
                    explicit_matches.append((function.qualified_name, 1.0, 1.0))
            
            # If we found explicit matches and method is "explicit" or "all", return them
            if explicit_matches:
                logger.info(f"Found {len(explicit_matches)} explicit matches for requirement {req_id}")
                return explicit_matches
            elif mapping_method == "explicit":
                # If no explicit matches found and method is "explicit", return empty list
                logger.info(f"No explicit matches found for requirement {req_id}")
                return []
        
        # If method is "explicit" only and we got here, return empty list (no matches)
        if mapping_method == "explicit":
            return []
        
        # Process semantic similarity if method includes "semantic"
        semantic_scores = {}
        if mapping_method in ["semantic", "semantic+llm", "all"]:
            # Calculate similarity scores
            semantic_scores = self.compute_similarity(req_desc, function_summaries)
            
            # Filter functions by similarity threshold
            semantic_scores = {
                func_name: score for func_name, score in semantic_scores.items()
                if score >= self.similarity_threshold
            }
            
            logger.info(f"Found {len(semantic_scores)} candidate functions using semantic similarity for requirement {req_id}")
        
        # If method is "semantic" only, convert scores to results format and return
        if mapping_method == "semantic":
            results = [(func_name, score, 0.0) for func_name, score in semantic_scores.items()]
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        
        # Build a function name to function object mapping for quick lookups
        function_map = {f.qualified_name: f for f in functions}
        
        # Process LLM confidence if method includes "llm"
        if mapping_method in ["llm", "semantic+llm", "all"]:
            results = []
            
            # For "llm" only, we need to consider all functions, not just those from semantic matching
            if mapping_method == "llm":
                candidate_functions = {func.qualified_name: 0.0 for func in functions}
            else:
                candidate_functions = semantic_scores
            
            # For each candidate, get LLM confidence
            for func_name, similarity in candidate_functions.items():
                if func_name in function_map:
                    function = function_map[func_name]
                    function_content = function.content
                    function_summary = function_summaries.get(func_name, "")
                    
                    llm_confidence = self.get_llm_confidence(
                        req_desc, module_id, func_name, function_summary, function_content
                    )
                    
                    # Only include functions with sufficient LLM confidence
                    if llm_confidence >= self.similarity_threshold:
                        results.append((func_name, similarity, llm_confidence))
                        logger.debug(f"Function {func_name} matched requirement {req_id} with LLM confidence: {llm_confidence:.2f}")
            
            logger.info(f"Found {len(results)} functions with sufficient LLM confidence for requirement {req_id}")
            
            # Sort results by the appropriate criteria based on method
            if mapping_method == "llm":
                # For "llm" only, sort solely by LLM confidence
                results.sort(key=lambda x: x[2], reverse=True)
            else:
                # For "semantic+llm" or "all", use weighted combination
                results.sort(key=lambda x: (0.4 * x[1] + 0.6 * x[2]), reverse=True)
            
            return results
        
        # This should not be reached if the mapping_method is valid
        logger.warning(f"Unknown mapping method: {mapping_method}")
        return []
    
    def map_all_requirements(self, requirements: List[Tuple[str, str, str]],
                           functions: List[CppFunction],
                           function_summaries: Dict[str, str],
                           mapping_method: str = "all") -> Dict[str, List[Tuple[str, float, float]]]:
        """
        Map all requirements to functions using the specified mapping method.
        
        Args:
            requirements (List[Tuple[str, str, str]]): List of (req_id, req_desc, module_id) tuples
            functions (List[CppFunction]): All available functions
            function_summaries (Dict[str, str]): Map of function names to summaries
            mapping_method (str): Method to use for mapping: "explicit", "semantic", "llm", 
                                 "semantic+llm", or "all"
            
        Returns:
            Dict[str, List[Tuple[str, float, float]]]: 
                Maps requirement IDs to lists of (function_name, similarity, llm_confidence) tuples
        """
        results = {}
        
        logger.info(f"Mapping {len(requirements)} requirements to functions")
        
        # Process each requirement
        for i, (req_id, req_desc, module_id) in enumerate(requirements):
            matches = self.map_requirement(
                req_id, req_desc, module_id, functions, function_summaries, mapping_method
            )
            results[req_id] = matches
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(requirements)} requirements")
                
        return results
