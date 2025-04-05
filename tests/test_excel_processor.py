"""
Tests for the ExcelProcessor module.
"""
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.excel_processor import ExcelProcessor

class TestExcelProcessor:
    """Tests for the ExcelProcessor class."""
    
    def test_read_requirements_success(self, sample_requirements_file):
        """Test successfully reading requirements from an Excel file."""
        processor = ExcelProcessor()
        df = processor.read_requirements(sample_requirements_file)
        
        # Check that we got a DataFrame with the expected columns
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Verify required columns are present
        for col in processor.REQUIRED_COLUMNS:
            assert col in df.columns
            
        # Verify content
        assert "REQ-MATH-001" in df["Requirement ID"].values
        
    def test_read_requirements_file_not_found(self):
        """Test handling of file not found error."""
        processor = ExcelProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.read_requirements("nonexistent_file.xlsx")
    
    def test_validate_columns_missing_columns(self):
        """Test validation when required columns are missing."""
        processor = ExcelProcessor()
        
        # Create a DataFrame with missing columns
        processor.requirements_df = pd.DataFrame({
            "Requirement ID": ["REQ-001"],
            # Missing "Requirement Description" column
            "Module Id": ["Test"]
        })
        
        with pytest.raises(ValueError) as excinfo:
            processor._validate_columns()
        
        assert "Missing required columns" in str(excinfo.value)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        processor = ExcelProcessor()
        
        # Create test data with whitespace and missing values
        processor.requirements_df = pd.DataFrame({
            "Requirement ID": ["REQ-001", "REQ-002", "REQ-003"],
            "Requirement Description": ["Test 1 ", " Test 2", None],
            "Module Id": ["Mod1", "Mod2", "Mod3"]
        })
        
        processor._preprocess_data()
        
        # Check that whitespace was trimmed
        assert processor.requirements_df["Requirement Description"][0] == "Test 1"
        assert processor.requirements_df["Requirement Description"][1] == "Test 2"
        
        # Check that row with None was dropped
        assert len(processor.requirements_df) == 2
    
    def test_get_requirements_before_loading(self):
        """Test getting requirements before loading."""
        processor = ExcelProcessor()
        
        with pytest.raises(RuntimeError):
            processor.get_requirements()
    
    def test_get_requirements_after_loading(self, sample_requirements_file):
        """Test getting requirements after loading."""
        processor = ExcelProcessor()
        processor.read_requirements(sample_requirements_file)
        
        df = processor.get_requirements()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
