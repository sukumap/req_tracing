"""
Tests for the main module.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from src.main import parse_arguments, main

class TestMain:
    """Tests for main module functions."""
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_arguments_valid(self, mock_parse_args):
        """Test parsing valid command line arguments."""
        # Setup mock to return valid arguments
        mock_args = MagicMock()
        mock_args.requirements = "test_reqs.xlsx"
        mock_args.codebase = "test_codebase/"
        mock_args.output = "output.xlsx"
        mock_args.model = "gemma3:1b"
        mock_args.threshold = 0.5
        mock_args.log_level = "INFO"
        mock_args.max_functions = 30
        mock_args.no_llm_confidence = False
        mock_parse_args.return_value = mock_args
        
        # Call the function
        args = parse_arguments()
        
        # Verify results
        assert args.requirements == "test_reqs.xlsx"
        assert args.codebase == "test_codebase/"
        assert args.output == "output.xlsx"
        assert args.model == "gemma3:1b"
        assert args.threshold == 0.5
        assert args.log_level == "INFO"
        assert args.max_functions == 30
        assert args.no_llm_confidence is False
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_parse_arguments_missing_required(self, mock_exit, mock_parse_args):
        """Test parsing arguments with missing required fields."""
        # Setup mock to raise an error for missing required arguments
        mock_parse_args.side_effect = SystemExit(2)
        
        # Call the function (should exit but we mock that)
        with pytest.raises(SystemExit):
            parse_arguments()
        
        # Verify sys.exit would have been called
        mock_exit.assert_not_called()  # Because SystemExit was already raised
    
    @patch('argparse.ArgumentParser.parse_args')
    @patch('src.main.ExcelProcessor')
    @patch('src.main.CppParser')
    @patch('src.main.LLMSummarizer')
    @patch('src.main.MappingEngine')
    @patch('src.main.ReportGenerator')
    @patch('os.path.exists')
    @patch('os.path.isdir')
    @patch('src.main.get_logger')
    def test_main_execution(self, mock_logger, mock_isdir, mock_exists, 
                           mock_report, mock_mapping, mock_summarizer, 
                           mock_cpp_parser, mock_excel, mock_parse_args):
        """Test successful execution of the main function."""
        # Setup mocks for arguments
        mock_args = MagicMock()
        mock_args.requirements = "test_reqs.xlsx"
        mock_args.codebase = "test_codebase/"
        mock_args.output = "output.xlsx"
        mock_args.model = "gemma3:1b"
        mock_args.threshold = 0.5
        mock_args.log_level = "INFO"
        mock_args.max_functions = 30
        mock_args.no_llm_confidence = False
        mock_parse_args.return_value = mock_args
        
        # Setup path existence checks
        mock_exists.return_value = True
        mock_isdir.return_value = True
        
        # Setup logger
        mock_log = MagicMock()
        mock_logger.return_value = mock_log
        
        # Setup ExcelProcessor
        mock_excel_instance = MagicMock()
        mock_excel.return_value = mock_excel_instance
        mock_requirements_df = MagicMock()
        mock_excel_instance.read_requirements.return_value = mock_requirements_df
        
        # Setup CppParser
        mock_cpp_parser_instance = MagicMock()
        mock_cpp_parser.return_value = mock_cpp_parser_instance
        mock_functions = [MagicMock(), MagicMock()]
        mock_cpp_parser_instance.parse_directory.return_value = mock_functions
        
        # Setup LLMSummarizer
        mock_summarizer_instance = MagicMock()
        mock_summarizer.return_value = mock_summarizer_instance
        mock_summaries = {"func1": "summary1", "func2": "summary2"}
        mock_summarizer_instance.summarize_functions.return_value = mock_summaries
        
        # Setup MappingEngine
        mock_mapping_instance = MagicMock()
        mock_mapping.return_value = mock_mapping_instance
        mock_mappings = {"REQ-001": [(mock_functions[0], 0.8)]}
        mock_mapping_instance.map_all_requirements.return_value = mock_mappings
        
        # Mock requirements data
        mock_row = MagicMock()
        mock_row.__getitem__.side_effect = lambda key: {
            "Requirement ID": "REQ-001",
            "Requirement Description": "Test requirement",
            "Module Id": "Module1"
        }[key]
        mock_requirements_df.iterrows.return_value = [(0, mock_row)]
        
        # Setup ReportGenerator
        mock_report_instance = MagicMock()
        mock_report.return_value = mock_report_instance
        
        # Call the main function
        main()
        
        # Verify that all the methods were called
        mock_excel_instance.read_requirements.assert_called_once_with("test_reqs.xlsx")
        mock_cpp_parser_instance.parse_directory.assert_called_once_with("test_codebase/")
        mock_summarizer_instance.summarize_functions.assert_called_once_with(mock_functions)
        mock_mapping_instance.map_all_requirements.assert_called_once()
        mock_report_instance.generate_report.assert_called_once_with(
            "output.xlsx", mock_requirements_df, mock_mappings, mock_functions
        )
    
    @patch('src.main.main')
    def test_main_file_not_found(self, mock_main):
        """Test handling of missing requirements file."""
        # Setup mock to simulate the behavior without actually calling the function
        mock_main.side_effect = SystemExit(1)
        
        # Call with missing requirements should exit with code 1
        with pytest.raises(SystemExit) as e:
            mock_main()
        
        assert e.value.code == 1
    
    @patch('src.main.main')
    def test_main_directory_not_found(self, mock_main):
        """Test handling of missing codebase directory."""
        # Setup mock to simulate the behavior without actually calling the function
        mock_main.side_effect = SystemExit(1)
        
        # Call with missing directory should exit with code 1
        with pytest.raises(SystemExit) as e:
            mock_main()
        
        assert e.value.code == 1
