"""
Main application script for the C++ Codebase Requirement Tracing Tool.
This module ties together all components and provides a command-line interface.
"""
import os
import sys
import argparse
import logging
import pandas as pd
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# Now import relative to the src directory
from utils.logger import get_logger
from excel_processor import ExcelProcessor
from cpp_parser import CppParser, CppFunction
from llm_summarizer import LLMSummarizer
from mapping_engine import MappingEngine
from report_generator import ReportGenerator

# Import our custom model extensions if available
try:
    from tinyllama_summarizer import TinyLlamaSummarizer
    from tinyllama_mapper import TinyLlamaMapper
    TINYLLAMA_AVAILABLE = True
except ImportError:
    TINYLLAMA_AVAILABLE = False

# Import our custom Gemma fine-tuned extensions if available
try:
    from gemma_finetuned_mapper import GemmaFineTunedMapper
    GEMMA_FINETUNED_AVAILABLE = True
except ImportError:
    GEMMA_FINETUNED_AVAILABLE = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Trace C++ code to requirements using semantic analysis and LLM"
    )
    
    parser.add_argument(
        "--codebase", 
        "-c", 
        required=True,
        help="Path to the C++ codebase directory"
    )
    
    parser.add_argument(
        "--requirements", 
        "-r", 
        required=True,
        help="Path to the requirements Excel file"
    )
    
    parser.add_argument(
        "--output", 
        "-o", 
        required=True,
        help="Path to the output Excel report"
    )
    
    parser.add_argument(
        "--threshold", 
        "-t", 
        type=float, 
        default=0.5,
        help="Similarity threshold for matching (0.0 to 1.0, default: 0.5)"
    )
    
    parser.add_argument(
        "--model",
        "-m",
        default="gemma3:1b",
        help="Ollama model name (default: gemma3:1b)"
    )
    
    # Options for using trained TinyLlama model with LoRA weights
    parser.add_argument(
        "--use-tinyllama",
        action="store_true",
        help="Use trained TinyLlama model with LoRA weights instead of Ollama model"
    )
    
    parser.add_argument(
        "--tinyllama-path",
        default=None,
        help="Path to trained TinyLlama model weights (default: use latest epoch from rl_weights)"
    )
    
    # Options for using fine-tuned Gemma model with LoRA weights
    parser.add_argument(
        "--use-gemma-finetuned",
        action="store_true",
        help="Use fine-tuned Gemma model with LoRA weights instead of Ollama model"
    )
    
    parser.add_argument(
        "--gemma-path",
        default=None,
        help="Path to fine-tuned Gemma model weights (default: use latest epoch from models/gemma_lora_weights)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--max-functions",
        type=int,
        default=30,
        help="Maximum functions to list directly before creating a separate tab (default: 30)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=True,  # Disabled by default to ensure debug output is visible
        help="Disable caching for LLM summarizations (default: True)"
    )
    
    # Replace the simple --no-llm-confidence flag with a more flexible mapping method selection
    mapping_group = parser.add_argument_group('Mapping Method Options')
    mapping_group.add_argument(
        "--mapping-method",
        choices=["explicit", "semantic", "llm", "semantic+llm", "all"],
        default="all",
        help="Method to use for mapping requirements to functions: "
             "explicit (only use explicit @requirement tags), "
             "semantic (only use semantic similarity), "
             "llm (only use LLM confidence), "
             "semantic+llm (combine semantic similarity with LLM confidence), "
             "all (use explicit tags first, then semantic+llm) (default: all)"
    )
    
    return parser.parse_args()

def main():
    """Main function that runs the requirement tracing tool."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logger
    log_level = getattr(logging, args.log_level)
    logger = get_logger(log_level=log_level)
    logger.info("Starting C++ Codebase Requirement Tracing Tool")
    
    # Validate paths
    if not os.path.exists(args.requirements):
        logger.error(f"Requirements file not found: {args.requirements}")
        sys.exit(1)
    
    if not os.path.exists(args.codebase) or not os.path.isdir(args.codebase):
        logger.error(f"Codebase directory not found: {args.codebase}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Print model selection info
        if args.use_gemma_finetuned:
            if GEMMA_FINETUNED_AVAILABLE:
                logger.info("Using fine-tuned Gemma model with LoRA weights")
                if args.gemma_path:
                    logger.info(f"Fine-tuned Gemma model path: {args.gemma_path}")
                else:
                    logger.info("Using latest fine-tuned Gemma epoch from models/gemma_lora_weights directory")
            else:
                logger.warning("Fine-tuned Gemma extensions not available, falling back to Ollama model")
                logger.info(f"Using Ollama model: {args.model}")
        elif args.use_tinyllama:
            if TINYLLAMA_AVAILABLE:
                logger.info("Using trained TinyLlama model with LoRA weights")
                if args.tinyllama_path:
                    logger.info(f"TinyLlama model path: {args.tinyllama_path}")
                else:
                    logger.info("Using latest TinyLlama epoch from rl_weights directory")
            else:
                logger.warning("TinyLlama extensions not available, falling back to Ollama model")
                logger.info(f"Using Ollama model: {args.model}")
        else:
            logger.info(f"Using Ollama model: {args.model}")
        
        # 1. Read requirements from Excel
        logger.info("Reading requirements from Excel...")
        excel_processor = ExcelProcessor()
        requirements_df = excel_processor.read_requirements(args.requirements)
        
        # 2. Parse C++ code to extract functions
        logger.info("Parsing C++ codebase...")
        cpp_parser = CppParser()
        functions = cpp_parser.parse_directory(args.codebase)
        logger.info(f"Extracted {len(functions)} functions from the codebase")
        
        # 3. Generate function summaries using selected model
        if args.use_tinyllama and TINYLLAMA_AVAILABLE:
            logger.info("Generating function summaries using trained TinyLlama model with LoRA weights...")
            logger.info(f"Cache {'disabled' if args.no_cache else 'enabled'} for function summarization")
            llm_summarizer = TinyLlamaSummarizer(model_path=args.tinyllama_path, use_cache=not args.no_cache)
        else:
            model_name = "Gemma 3:1b" if "gemma" in args.model.lower() else args.model
            logger.info(f"Generating function summaries using {model_name} via Ollama...")
            logger.info(f"Cache {'disabled' if args.no_cache else 'enabled'} for function summarization")
            llm_summarizer = LLMSummarizer(model=args.model, use_cache=not args.no_cache)
        
        function_summaries = llm_summarizer.summarize_functions(functions)
        logger.info(f"Generated {len(function_summaries)} function summaries")
        
        # 4. Map requirements to functions
        logger.info("Mapping requirements to functions...")
        if args.use_gemma_finetuned and GEMMA_FINETUNED_AVAILABLE:
            logger.info("Using fine-tuned Gemma model for requirement mapping...")
            mapping_engine = GemmaFineTunedMapper(similarity_threshold=args.threshold, model_path=args.gemma_path)
        elif args.use_tinyllama and TINYLLAMA_AVAILABLE:
            logger.info("Using trained TinyLlama model for requirement mapping...")
            mapping_engine = TinyLlamaMapper(similarity_threshold=args.threshold, model_path=args.tinyllama_path)
        else:
            mapping_engine = MappingEngine(similarity_threshold=args.threshold, model=args.model)
        
        # Prepare requirements list
        requirements_list = []
        for _, row in requirements_df.iterrows():
            req_id = row["Requirement ID"]
            req_desc = row["Requirement Description"]
            module_id = row["Module Id"]
            requirements_list.append((req_id, req_desc, module_id))
        
        # Perform mapping
        mappings = mapping_engine.map_all_requirements(
            requirements_list,
            functions,
            function_summaries,
            mapping_method=args.mapping_method
        )
        
        # 5. Generate the report
        logger.info("Generating Excel report...")
        report_generator = ReportGenerator(max_functions_per_req=args.max_functions)
        report_generator.generate_report(
            args.output,
            requirements_df,
            mappings,
            functions
        )
        
        logger.info(f"Successfully generated report at: {args.output}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
