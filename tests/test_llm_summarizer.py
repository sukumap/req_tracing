"""
Tests for the LLM Summarizer module.
"""
import os
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

from src.llm_summarizer import LLMSummarizer
from src.cpp_parser import CppFunction

# Skip these tests if Ollama isn't available
pytestmark = pytest.mark.skipif(
    True, 
    reason="Ollama dependency is needed for these tests. Skipping in CI environment."
)

class TestLLMSummarizer:
    """Tests for the LLMSummarizer class."""
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('src.llm_summarizer.get_logger')
    def setup_method(self, mock_logger, mock_makedirs, mock_exists):
        """Set up test environment before each test method."""
        # Mock logger
        self.mock_logger = MagicMock()
        mock_logger.return_value = self.mock_logger
        
        # Mock file system operations
        mock_exists.return_value = False  # Cache dir doesn't exist by default
        
        # Mock ollama operations
        self.mock_ollama_response = {
            "message": {"content": "This function adds two numbers and returns their sum."}
        }
        
        # Create sample test functions
        self.test_function = CppFunction(
            name="add",
            content="int add(int a, int b) {\n    // Adds two integers\n    return a + b;\n}",
            file_path="/path/to/math.cpp",
            start_line=10,
            namespace="math"
        )
        
        # Setup summarizer with patched dependencies
        with patch('builtins.open', mock_open()):
            with patch('src.llm_summarizer.ollama'):
                self.summarizer = LLMSummarizer(model="gemma3:1b", cache_dir="test_cache")
    
    @patch('src.llm_summarizer.ollama.list')
    def test_initialization(self, mock_list):
        """Test summarizer initialization."""
        # Setup mocks
        mock_list.return_value = {"models": [{"name": "gemma3:1b"}]}
        
        # Create new instance with mocked components
        with patch('builtins.open', mock_open()):
            with patch('os.path.exists', return_value=False):
                with patch('os.makedirs'):
                    summarizer = LLMSummarizer(model="gemma3:1b", cache_dir="test_cache")
        
        # Verify initialization
        assert summarizer.model == "gemma3:1b"
        assert summarizer.cache_dir == "test_cache"
        assert isinstance(summarizer.cache, dict)
        
        # Verify Ollama was checked
        mock_list.assert_called_once()
    
    @patch('src.llm_summarizer.ollama.chat')
    def test_generate_summary(self, mock_chat):
        """Test generating a function summary."""
        # Setup mock response
        mock_chat.return_value = self.mock_ollama_response
        
        # Generate summary
        summary = self.summarizer.generate_summary(self.test_function)
        
        # Verify result
        assert summary == "This function adds two numbers and returns their sum."
        
        # Verify chat was called
        mock_chat.assert_called_once()
        args, kwargs = mock_chat.call_args
        
        # Check that the system prompt contains the function content
        messages = kwargs.get('messages', [])
        assert len(messages) > 0
        assert "int add(int a, int b)" in str(messages)
    
    @patch('src.llm_summarizer.ollama.chat')
    def test_generate_summary_error(self, mock_chat):
        """Test error handling when generating a function summary."""
        # Setup mock to raise exception
        mock_chat.side_effect = Exception("Connection error")
        
        # Generate summary - should handle the error gracefully
        summary = self.summarizer.generate_summary(self.test_function)
        
        # Verify default message is returned on error
        assert "Failed to generate summary" in summary
        
        # Verify the error was logged
        self.mock_logger.error.assert_called_once()
    
    @patch('src.llm_summarizer.LLMSummarizer.generate_summary')
    def test_summarize_functions(self, mock_generate_summary):
        """Test summarizing multiple functions."""
        # Setup mock to return a test summary
        mock_generate_summary.return_value = "Test summary"
        
        # Create multiple test functions
        functions = [
            self.test_function,
            CppFunction(
                name="subtract",
                content="int subtract(int a, int b) { return a - b; }",
                file_path="/path/to/math.cpp",
                start_line=20,
                namespace="math"
            )
        ]
        
        # Summarize functions
        summaries = self.summarizer.summarize_functions(functions)
        
        # Verify results
        assert len(summaries) == 2
        assert "math::add" in summaries
        assert "math::subtract" in summaries
        assert summaries["math::add"] == "Test summary"
        assert summaries["math::subtract"] == "Test summary"
        
        # Verify generate_summary was called for each function
        assert mock_generate_summary.call_count == 2
    
    @patch('src.llm_summarizer.LLMSummarizer.generate_summary')
    def test_summarize_functions_with_cache(self, mock_generate_summary):
        """Test that cache is used when summarizing functions."""
        # Setup cached summary
        self.summarizer.cache = {
            "math::add": "Cached summary for add"
        }
        
        # Setup mock for non-cached function
        mock_generate_summary.return_value = "New summary for subtract"
        
        # Create functions
        functions = [
            self.test_function,  # Should use cache
            CppFunction(
                name="subtract",
                content="int subtract(int a, int b) { return a - b; }",
                file_path="/path/to/math.cpp",
                start_line=20,
                namespace="math"
            )  # Should generate new summary
        ]
        
        # Summarize functions
        summaries = self.summarizer.summarize_functions(functions)
        
        # Verify results
        assert len(summaries) == 2
        assert summaries["math::add"] == "Cached summary for add"
        assert summaries["math::subtract"] == "New summary for subtract"
        
        # Verify generate_summary was called only for the non-cached function
        mock_generate_summary.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    def test_save_cache(self, mock_file):
        """Test saving the cache to disk."""
        # Setup cache
        self.summarizer.cache = {
            "function1": "Summary 1",
            "function2": "Summary 2"
        }
        
        # Save cache
        self.summarizer._save_cache()
        
        # Verify file operations
        mock_file.assert_called_once_with(os.path.join("test_cache", "summaries.json"), 'w')
        
        # Verify json.dump was called with the cache
        handle = mock_file()
        handle.write.assert_called_once()
        
        # Ensure call contained our cache data
        call_args = handle.write.call_args[0][0]
        assert "function1" in call_args
        assert "function2" in call_args
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"function1": "Summary 1"}')
    def test_load_cache(self, mock_file, mock_exists):
        """Test loading the cache from disk."""
        # Setup mocks
        mock_exists.return_value = True  # Cache file exists
        
        # Create new summarizer to test load_cache
        with patch('src.llm_summarizer.ollama'):
            summarizer = LLMSummarizer(cache_dir="test_cache")
        
        # Verify cache was loaded
        assert "function1" in summarizer.cache
        assert summarizer.cache["function1"] == "Summary 1"
        
        # Verify file operations
        mock_file.assert_called_once_with(os.path.join("test_cache", "summaries.json"), 'r')
    
    def test_format_prompt(self):
        """Test formatting the prompt for the LLM."""
        # Format a prompt
        prompt = self.summarizer._format_prompt(self.test_function)
        
        # Verify prompt content
        assert "int add(int a, int b)" in prompt
        assert "// Adds two integers" in prompt
        assert "return a + b;" in prompt
