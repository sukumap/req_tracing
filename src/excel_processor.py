"""
Excel processing module for handling requirement input and validation.
"""
import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger()

class ExcelProcessor:
    """
    Handles reading and validating requirements from Excel files.
    """
    # Required columns in the input Excel file
    REQUIRED_COLUMNS = ["Requirement ID", "Requirement Description", "Module Id"]
    
    def __init__(self):
        """Initialize the Excel processor."""
        self.requirements_df = None
    
    def read_requirements(self, file_path):
        """
        Read requirements from an Excel file.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            pd.DataFrame: DataFrame with requirement data
            
        Raises:
            FileNotFoundError: If the Excel file doesn't exist
            ValueError: If the Excel file is missing required columns
        """
        logger.info(f"Reading requirements from: {file_path}")
        
        if not os.path.exists(file_path):
            error_msg = f"Requirements file not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            self.requirements_df = pd.read_excel(file_path)
            logger.info(f"Successfully read {len(self.requirements_df)} requirements")
            
            # Validate required columns
            self._validate_columns()
            
            # Preprocess data
            self._preprocess_data()
            
            return self.requirements_df
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise
    
    def _validate_columns(self):
        """
        Validate that the Excel file contains all required columns.
        
        Raises:
            ValueError: If any required column is missing
        """
        missing_columns = [col for col in self.REQUIRED_COLUMNS 
                          if col not in self.requirements_df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("All required columns present in the requirements file")
    
    def _preprocess_data(self):
        """Clean and standardize input data."""
        # Trim whitespace from string columns
        for col in self.requirements_df.select_dtypes(include='object').columns:
            self.requirements_df[col] = self.requirements_df[col].str.strip()
        
        # Drop rows with missing values in required columns
        self.requirements_df.dropna(subset=self.REQUIRED_COLUMNS, inplace=True)
        
        # Reset index after dropping rows
        self.requirements_df.reset_index(drop=True, inplace=True)
        
        logger.info(f"After preprocessing: {len(self.requirements_df)} valid requirements")
        
    def get_requirements(self):
        """
        Get the processed requirements.
        
        Returns:
            pd.DataFrame: DataFrame with requirement data
            
        Raises:
            RuntimeError: If no requirements have been loaded yet
        """
        if self.requirements_df is None:
            error_msg = "No requirements have been loaded yet"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return self.requirements_df
