"""
Report generator module for creating Excel reports that map requirements to functions.
"""
import os
from typing import Dict, List, Tuple
import pandas as pd
import xlsxwriter

from src.cpp_parser import CppFunction
from src.utils.logger import get_logger

logger = get_logger()

class ReportGenerator:
    """
    Generates Excel reports mapping requirements to implementing functions.
    Handles special formatting for requirements with large numbers of matches.
    """
    
    def __init__(self, max_functions_per_req=30):
        """
        Initialize the report generator.
        
        Args:
            max_functions_per_req (int): Maximum number of functions to list directly
                                        before creating a separate tab
        """
        self.max_functions_per_req = max_functions_per_req
        logger.info(f"Initialized report generator with max functions per requirement: {max_functions_per_req}")
    
    def generate_report(self, 
                        output_file: str,
                        requirements: pd.DataFrame,
                        mappings: Dict[str, List[Tuple[str, float, float]]],
                        functions: List[CppFunction]) -> None:
        """
        Generate an Excel report with requirement to function mappings.
        
        Args:
            output_file (str): Path to the output Excel file
            requirements (pd.DataFrame): DataFrame with requirements information
            mappings (Dict[str, List[Tuple[str, float, float]]]): 
                Map of requirement IDs to function matches with scores
            functions (List[CppFunction]): All functions, used for reference
        """
        logger.info(f"Generating report at: {output_file}")
        
        # Create a mapping from qualified function names to function objects
        function_map = {func.qualified_name: func for func in functions}
        
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Create formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D0E0FF',
                'border': 1
            })
            
            hyperlink_format = workbook.add_format({
                'underline': True,
                'font_color': 'blue'
            })
            
            # Create the main summary sheet
            self._create_summary_sheet(
                writer, requirements, mappings, function_map, 
                header_format, hyperlink_format
            )
            
            # Create separate sheets for requirements with many functions
            for req_id, matches in mappings.items():
                if len(matches) > self.max_functions_per_req:
                    self._create_requirement_detail_sheet(
                        writer, req_id, matches, function_map, 
                        header_format
                    )
            
            logger.info(f"Successfully generated report at: {output_file}")
    
    def _create_summary_sheet(self, 
                             writer: pd.ExcelWriter,
                             requirements: pd.DataFrame,
                             mappings: Dict[str, List[Tuple[str, float, float]]],
                             function_map: Dict[str, CppFunction],
                             header_format, 
                             hyperlink_format) -> None:
        """
        Create the main summary sheet in the Excel report.
        
        Args:
            writer (pd.ExcelWriter): The Excel writer
            requirements (pd.DataFrame): DataFrame with requirements information
            mappings (Dict[str, List[Tuple[str, float, float]]]): 
                Map of requirement IDs to function matches with scores
            function_map (Dict[str, CppFunction]): Map of function names to function objects
            header_format: Excel format for headers
            hyperlink_format: Excel format for hyperlinks
        """
        # Prepare data for the summary sheet
        summary_data = []
        
        for _, row in requirements.iterrows():
            req_id = row["Requirement ID"]
            req_desc = row["Requirement Description"]
            module_id = row["Module Id"]
            
            matches = mappings.get(req_id, [])
            num_matches = len(matches)
            
            # For requirements with few functions, list them directly
            # For requirements with many functions, add a hyperlink
            if num_matches <= self.max_functions_per_req:
                if num_matches > 0:
                    # List functions with scores
                    function_list = []
                    for func_name, similarity, confidence in matches:
                        if func_name in function_map:
                            func = function_map[func_name]
                            location = f"{os.path.basename(func.file_path)}:{func.start_line}"
                            score_info = f" (Sim: {similarity:.2f}, Conf: {confidence:.2f})"
                            function_list.append(f"{func_name} @ {location}{score_info}")
                        else:
                            function_list.append(f"{func_name} (Unknown location)")
                    
                    functions_text = "\n".join(function_list)
                else:
                    functions_text = "No matching functions found"
                
                summary_data.append([
                    req_id, req_desc, module_id, num_matches, functions_text, "", ""
                ])
            else:
                # For requirements with many functions, prepare for hyperlink
                sheet_name = f"REQ_{req_id.replace('-', '_')}"
                top_match = matches[0][0] if matches else "N/A"
                summary_data.append([
                    req_id, req_desc, module_id, num_matches, 
                    f"See details", sheet_name, top_match
                ])
        
        # Create DataFrame for the summary sheet
        summary_df = pd.DataFrame(summary_data, columns=[
            "Requirement ID", "Requirement Description", "Module Id",
            "Number of Functions", "Function Details", "_sheet_name", "_top_match"
        ])
        
        # Write the summary sheet
        summary_df.drop(columns=["_sheet_name", "_top_match"]).to_excel(
            writer, sheet_name="Summary", index=False
        )
        
        # Get worksheet object
        worksheet = writer.sheets["Summary"]
        
        # Apply header formatting
        for col_num, value in enumerate(summary_df.columns[:-2]):
            worksheet.write(0, col_num, value, header_format)
        
        # Add hyperlinks for requirements with many functions
        for row_num, row in enumerate(summary_data):
            if row[5]:  # If _sheet_name exists
                worksheet.write_url(
                    row_num + 1, 4,  # +1 for header row
                    f"internal:'{row[5]}'!A1",
                    hyperlink_format,
                    "Click for details"
                )
        
        # Adjust column widths
        worksheet.set_column('A:A', 15)  # Requirement ID
        worksheet.set_column('B:B', 50)  # Requirement Description
        worksheet.set_column('C:C', 15)  # Module Id
        worksheet.set_column('D:D', 10)  # Number of Functions
        worksheet.set_column('E:E', 50)  # Function Details
    
    def _create_requirement_detail_sheet(self, 
                                        writer: pd.ExcelWriter,
                                        req_id: str,
                                        matches: List[Tuple[str, float, float]],
                                        function_map: Dict[str, CppFunction],
                                        header_format) -> None:
        """
        Create a detailed sheet for a requirement with many functions.
        
        Args:
            writer (pd.ExcelWriter): The Excel writer
            req_id (str): The requirement ID
            matches (List[Tuple[str, float, float]]): List of function matches
            function_map (Dict[str, CppFunction]): Map of function names to function objects
            header_format: Excel format for headers
        """
        sheet_name = f"REQ_{req_id.replace('-', '_')}"
        
        # Prepare data for the detail sheet
        detail_data = []
        
        for func_name, similarity, confidence in matches:
            if func_name in function_map:
                func = function_map[func_name]
                file_path = func.file_path
                file_name = os.path.basename(file_path)
                start_line = func.start_line
                
                detail_data.append([
                    func_name,
                    file_name,
                    start_line,
                    file_path,  # Full path for reference
                    similarity,
                    confidence,
                    (0.4 * similarity + 0.6 * confidence)  # Combined score
                ])
            else:
                detail_data.append([
                    func_name, "Unknown", 0, "Unknown", similarity, confidence,
                    (0.4 * similarity + 0.6 * confidence)
                ])
        
        # Create DataFrame for the detail sheet
        detail_df = pd.DataFrame(detail_data, columns=[
            "Function Name", "File Name", "Line Number", "Full Path",
            "Similarity Score", "LLM Confidence", "Combined Score"
        ])
        
        # Write the detail sheet
        detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Get worksheet object
        worksheet = writer.sheets[sheet_name]
        
        # Apply header formatting
        for col_num, value in enumerate(detail_df.columns):
            worksheet.write(0, col_num, value, header_format)
        
        # Add a title with the requirement ID
        worksheet.merge_range('A1:G1', f"Detailed Function Mappings for Requirement: {req_id}", header_format)
        worksheet.write_string(1, 0, "Function Name", header_format)
        
        # Shift everything down by 1 row to make space for the title
        for row_num in range(len(detail_data)):
            for col_num in range(len(detail_df.columns)):
                worksheet.write(row_num + 2, col_num, detail_df.iloc[row_num, col_num])
        
        # Adjust column widths
        worksheet.set_column('A:A', 40)  # Function Name
        worksheet.set_column('B:B', 20)  # File Name
        worksheet.set_column('C:C', 10)  # Line Number
        worksheet.set_column('D:D', 50)  # Full Path
        worksheet.set_column('E:E', 15)  # Similarity Score
        worksheet.set_column('F:F', 15)  # LLM Confidence
        worksheet.set_column('G:G', 15)  # Combined Score
