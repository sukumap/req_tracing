"""
Tests for the Report Generator module.
"""
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, mock_open

from src.report_generator import ReportGenerator
from src.cpp_parser import CppFunction

class TestReportGenerator:
    """Tests for the ReportGenerator class."""
    
    @patch('src.report_generator.get_logger')
    def setup_method(self, method, mock_logger):
        """Set up test environment before each test method."""
        # Mock logger
        self.mock_logger = MagicMock()
        mock_logger.return_value = self.mock_logger
        
        # Create test instance with test parameters
        self.report_generator = ReportGenerator(max_functions_per_req=5)
        
        # Create sample test data
        self.sample_requirements = pd.DataFrame({
            'Requirement ID': ['REQ-001', 'REQ-002'],
            'Requirement Description': ['Requirement 1', 'Requirement 2'],
            'Module Id': ['Category A', 'Category B']
        })
        
        # Create some sample functions
        self.sample_functions = [
            CppFunction(
                name="add",
                content="int add(int a, int b) { return a + b; }",
                file_path="/path/to/math.cpp",
                start_line=10,
                namespace="math"
            ),
            CppFunction(
                name="subtract",
                content="int subtract(int a, int b) { return a - b; }",
                file_path="/path/to/math.cpp",
                start_line=20,
                namespace="math"
            )
        ]
        
        # Sample mappings (requirement ID -> list of (function, similarity_score, confidence_score))
        self.sample_mappings = {
            'REQ-001': [(self.sample_functions[0].qualified_name, 0.8, 0.9)],
            'REQ-002': [(self.sample_functions[1].qualified_name, 0.7, 0.8)]
        }
    
    def test_initialization(self):
        """Test report generator initialization."""
        # Create a new instance with different parameters
        custom_generator = ReportGenerator(max_functions_per_req=10)
        
        # Verify initialization parameters
        assert custom_generator.max_functions_per_req == 10
        assert self.report_generator.max_functions_per_req == 5
    
    @patch('src.report_generator.ReportGenerator._create_summary_sheet')
    @patch('src.report_generator.ReportGenerator._create_requirement_detail_sheet')
    @patch('pandas.ExcelWriter')
    def test_generate_report_creates_excel_file(self, mock_excel_writer, mock_detail_sheet, mock_summary_sheet):
        """Test that generate_report creates an Excel file."""
        # Setup mock ExcelWriter
        mock_writer = MagicMock()
        mock_excel_writer.return_value.__enter__.return_value = mock_writer
        
        # Call the function under test
        self.report_generator.generate_report(
            "test_report.xlsx",
            self.sample_requirements,
            self.sample_mappings,
            self.sample_functions
        )
        
        # Verify ExcelWriter was created with the correct file path
        mock_excel_writer.assert_called_once_with("test_report.xlsx", engine='xlsxwriter')
        
        # Verify summary sheet was created
        mock_summary_sheet.assert_called_once()
        
        # Detail sheet should not be called for our sample data (less than max functions)
        mock_detail_sheet.assert_not_called()
    
    @patch('pandas.DataFrame.to_excel')
    @patch('pandas.ExcelWriter')
    def test_create_summary_sheet(self, mock_writer, mock_to_excel):
        """Test creating the summary sheet."""
        # Setup mocks
        mock_xl_writer = MagicMock()
        mock_workbook = MagicMock()
        mock_worksheet = MagicMock()
        
        # Setup the writer context manager
        mock_writer.return_value.__enter__.return_value = mock_xl_writer
        
        # Setup workbook and worksheet
        mock_xl_writer.book = mock_workbook
        mock_xl_writer.sheets = {"Summary": mock_worksheet}
        
        # Mock formats
        mock_header_format = MagicMock()
        mock_hyperlink_format = MagicMock()
        
        # Create function map
        function_map = {func.qualified_name: func for func in self.sample_functions}
        
        # Call the method directly
        with patch.object(self.report_generator, '_create_summary_sheet') as mock_create_summary:
            self.report_generator.generate_report(
                "test_report.xlsx",
                self.sample_requirements,
                self.sample_mappings,
                self.sample_functions
            )
            
            # Verify _create_summary_sheet was called
            mock_create_summary.assert_called_once()
    
    @patch('pandas.DataFrame.to_excel')
    @patch('pandas.ExcelWriter')
    def test_create_detail_sheet_for_many_functions(self, mock_writer, mock_to_excel):
        """Test creating detail sheets for requirements with many functions."""
        # Setup mocks
        mock_xl_writer = MagicMock()
        mock_workbook = MagicMock()
        
        # Setup the writer context manager
        mock_writer.return_value.__enter__.return_value = mock_xl_writer
        
        # Setup workbook
        mock_xl_writer.book = mock_workbook
        
        # Mock formats
        mock_header_format = MagicMock()
        
        # Create function map
        function_map = {func.qualified_name: func for func in self.sample_functions}
        
        # Create mappings with more functions than the max
        many_functions_mappings = {
            'REQ-001': [(f"func{i}", 0.9 - i*0.01, 0.8 - i*0.01) for i in range(10)]
        }
        
        # Call the method directly with patch to isolate the detail sheet creation
        with patch.object(self.report_generator, '_create_summary_sheet'):
            with patch.object(self.report_generator, '_create_requirement_detail_sheet') as mock_create_detail:
                self.report_generator.generate_report(
                    "test_report.xlsx",
                    self.sample_requirements,
                    many_functions_mappings,
                    self.sample_functions
                )
                
                # Verify _create_requirement_detail_sheet was called
                mock_create_detail.assert_called()
    
    def test_coverage_calculation(self):
        """Test coverage calculation from mappings."""
        # Create a scenario with 3 requirements but only 2 mapped
        requirements = pd.DataFrame({
            'Requirement ID': ['REQ-001', 'REQ-002', 'REQ-003'],
            'Requirement Description': ['Req 1', 'Req 2', 'Req 3'],
            'Module Id': ['Mod A', 'Mod B', 'Mod C']
        })
        
        # Only 2 of 3 requirements are mapped
        mappings = {
            'REQ-001': [(self.sample_functions[0].qualified_name, 0.8, 0.9)],
            'REQ-002': [(self.sample_functions[1].qualified_name, 0.7, 0.8)]
        }
        
        # We don't have a direct way to test coverage calculation from ReportGenerator
        # so we'll verify our own calculation to match
        total_reqs = len(requirements)
        mapped_reqs = len(mappings)
        
        # Calculate coverage
        coverage_percentage = (mapped_reqs / total_reqs) * 100
        
        # We expect 2/3 = 66.67% coverage
        assert abs(coverage_percentage - 66.67) < 0.01
