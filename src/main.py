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

from src.utils.logger import get_logger
from src.excel_processor import ExcelProcessor
from src.cpp_parser import CppParser, CppFunction
from src.llm_summarizer import LLMSummarizer
from src.mapping_engine import MappingEngine
from src.report_generator import ReportGenerator

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
        # 1. Read requirements from Excel
        logger.info("Reading requirements from Excel...")
        excel_processor = ExcelProcessor()
        requirements_df = excel_processor.read_requirements(args.requirements)
        
        # 2. Parse C++ code to extract functions
        logger.info("Parsing C++ codebase...")
        cpp_parser = CppParser()
        functions = cpp_parser.parse_directory(args.codebase)
        logger.info(f"Extracted {len(functions)} functions from the codebase")
        
        # 3. Generate function summaries using Gemma 3:1b via Ollama
        logger.info("Generating function summaries using Gemma 3:1b...")
        llm_summarizer = LLMSummarizer(model=args.model)
        function_summaries = llm_summarizer.summarize_functions(functions)
        logger.info(f"Generated {len(function_summaries)} function summaries")
        
        # 4. Map requirements to functions
        logger.info("Mapping requirements to functions...")
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
