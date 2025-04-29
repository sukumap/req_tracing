#!/usr/bin/env python3
"""
Script to filter an Excel file to only include functions with a confidence score >= 0.95
"""

import pandas as pd
import re
import os

# File paths
input_file = "/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing/tests/data/output_gemma_finetuned_strict.xlsx"
output_file = "/home/praveen/sw_projects/requirement_tracer/req_tracing/req_tracing/tests/data/output_gemma_finetuned_strict_1.xlsx"

# Confidence threshold
THRESHOLD = 0.95

def filter_high_confidence(input_file, output_file, threshold):
    """
    Filter Excel file to only include functions with confidence >= threshold
    """
    print(f"Reading input file: {input_file}")
    
    # Read the Excel file
    try:
        # Try reading with pandas
        df = pd.read_excel(input_file)
        print(f"Successfully read Excel file with {len(df)} rows")
        
        # Create a new DataFrame to store the filtered results
        filtered_df = pd.DataFrame(columns=df.columns)
        
        # Regular expression to extract confidence scores
        confidence_pattern = re.compile(r'Conf:\s+([\d.]+)')
        
        # Process each row
        for index, row in df.iterrows():
            function_details = row['Function Details']
            if not isinstance(function_details, str):
                continue
                
            # Split the function details into individual functions
            functions = function_details.strip().split('\n')
            
            # Filter functions with confidence >= threshold
            high_conf_functions = []
            for func in functions:
                match = confidence_pattern.search(func)
                if match:
                    confidence = float(match.group(1))
                    if confidence >= threshold:
                        high_conf_functions.append(func)
            
            # If there are high-confidence functions, add to the filtered DataFrame
            if high_conf_functions:
                new_row = row.copy()
                new_row['Function Details'] = '\n'.join(high_conf_functions)
                new_row['Number of Functions'] = len(high_conf_functions)
                filtered_df = pd.concat([filtered_df, pd.DataFrame([new_row])], ignore_index=True)
        
        print(f"Filtered to {len(filtered_df)} rows with at least one function having confidence >= {threshold}")
        
        # Save to new Excel file
        filtered_df.to_excel(output_file, index=False)
        print(f"Saved filtered data to: {output_file}")
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")

if __name__ == "__main__":
    # Verify the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
    else:
        filter_high_confidence(input_file, output_file, THRESHOLD)
