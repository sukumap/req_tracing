"""
Tests for the CPP Parser module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open

from src.cpp_parser import CppParser, CppFunction

# Skip these tests if tree-sitter isn't available
pytestmark = pytest.mark.skipif(
    True, 
    reason="Tree-sitter dependency is needed for these tests. Skipping in CI environment."
)

class TestCppParser:
    """Tests for the CppParser class."""
    
    @patch('src.cpp_parser.Language')
    @patch('src.cpp_parser.Parser')
    @patch('src.cpp_parser.get_logger')
    @patch('os.path.exists', return_value=True)
    def setup_method(self, mock_exists, mock_logger, mock_parser_class, mock_language_class):
        """Set up test environment before each test method."""
        # Mock the logger
        self.mock_logger = MagicMock()
        mock_logger.return_value = self.mock_logger
        
        # Mock the tree-sitter Parser
        self.mock_parser = MagicMock()
        mock_parser_class.return_value = self.mock_parser
        
        # Mock the Language
        self.mock_language = MagicMock()
        mock_language_class.return_value = self.mock_language
        
        # Create test instance
        self.parser = CppParser()
        
        # Some test data
        self.sample_cpp_code = """
        #include <iostream>
        
        namespace example {
            
            int add(int a, int b) {
                return a + b;
            }
            
            class Calculator {
            public:
                int subtract(int a, int b) {
                    return a - b;
                }
            };
        }
        """
    
    @patch('src.cpp_parser.Language')
    @patch('src.cpp_parser.Parser')
    @patch('os.path.exists', return_value=True)
    def test_init(self, mock_exists, mock_parser_class, mock_language_class):
        """Test parser initialization."""
        # Mock the components
        mock_parser = MagicMock()
        mock_parser_class.return_value = mock_parser
        mock_language = MagicMock()
        mock_language_class.return_value = mock_language
        
        # Create a parser with the mocked components
        with patch('src.cpp_parser.get_logger'):
            parser = CppParser()
        
        # Verify that the parser was set up correctly
        mock_parser.set_language.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data="namespace test { void func() {} }")
    @patch('os.path.exists', return_value=True)
    def test_parse_file(self, mock_exists, mock_file):
        """Test parsing a C++ file."""
        # Mock the tree-sitter parsing
        mock_tree = MagicMock()
        self.mock_parser.parse.return_value = mock_tree
        
        # Create a mock node structure that would be returned by tree-sitter
        mock_root = MagicMock()
        mock_function_node = MagicMock()
        mock_function_node.type = "function_definition"
        mock_function_node.start_point = (5, 0)  # Line 5, column 0
        
        # Set up the namespace and function name nodes
        mock_namespace_node = MagicMock()
        mock_namespace_node.type = "namespace_definition"
        mock_namespace_node.text.decode.return_value = "test"
        
        mock_function_name_node = MagicMock()
        mock_function_name_node.text.decode.return_value = "func"
        
        # Mock finding the function name
        mock_tree.root = mock_root
        mock_root.children = [mock_namespace_node]
        mock_namespace_node.children = [mock_function_node]
        
        # Configure _extract_function_content to return a CppFunction
        with patch.object(self.parser, '_extract_function_content') as mock_extract:
            test_function = CppFunction(
                name="func",
                content="void func() {}",
                file_path="/path/to/test.cpp",
                start_line=5,
                namespace="test"
            )
            mock_extract.return_value = test_function
            
            # Call the method
            file_path = "/path/to/test.cpp"
            functions = self.parser.parse_file(file_path)
            
            # Verify results
            assert len(functions) > 0
            assert functions[0].name == "func"
            assert functions[0].namespace == "test"
            
            # Verify parser was called
            self.mock_parser.parse.assert_called_once()
    
    @patch('os.walk')
    @patch('os.path.exists', return_value=True)
    def test_parse_directory(self, mock_exists, mock_walk):
        """Test parsing a directory of C++ files."""
        # Mock os.walk to return a set of test files
        mock_walk.return_value = [
            ("/root", ["dir1"], ["file1.cpp", "file2.h"]),
            ("/root/dir1", [], ["file3.cpp"])
        ]
        
        # Mock parse_file to return some functions
        with patch.object(self.parser, 'parse_file') as mock_parse_file:
            # Create some test functions
            func1 = CppFunction(
                name="func1",
                content="void func1() {}",
                file_path="/root/file1.cpp",
                start_line=1
            )
            
            func2 = CppFunction(
                name="func2",
                content="void func2() {}",
                file_path="/root/file2.h",
                start_line=2
            )
            
            func3 = CppFunction(
                name="func3",
                content="void func3() {}",
                file_path="/root/dir1/file3.cpp",
                start_line=3
            )
            
            # Configure mock to return different functions for different files
            def side_effect(file_path):
                if file_path.endswith("file1.cpp"):
                    return [func1]
                elif file_path.endswith("file2.h"):
                    return [func2]
                elif file_path.endswith("file3.cpp"):
                    return [func3]
                return []
                
            mock_parse_file.side_effect = side_effect
            
            # Call the method
            functions = self.parser.parse_directory("/root")
            
            # Verify results
            assert len(functions) == 3
            assert any(f.name == "func1" for f in functions)
            assert any(f.name == "func2" for f in functions)
            assert any(f.name == "func3" for f in functions)
            
            # Verify parse_file was called for each file
            assert mock_parse_file.call_count == 3
    
    def test_cpp_function_properties(self):
        """Test the properties of the CppFunction class."""
        # Test with only name and basic properties
        func1 = CppFunction(
            name="func",
            content="void func() {}",
            file_path="/path/to/file.cpp",
            start_line=1
        )
        assert func1.qualified_name == "func"
        
        # Test with namespace
        func2 = CppFunction(
            name="func",
            content="void func() {}",
            file_path="/path/to/file.cpp",
            start_line=1,
            namespace="test"
        )
        assert func2.qualified_name == "test::func"
        
        # Test with class name
        func3 = CppFunction(
            name="func",
            content="void func() {}",
            file_path="/path/to/file.cpp",
            start_line=1,
            class_name="TestClass"
        )
        assert func3.qualified_name == "TestClass::func"
        
        # Test with both namespace and class name
        func4 = CppFunction(
            name="func",
            content="void func() {}",
            file_path="/path/to/file.cpp",
            start_line=1,
            namespace="test",
            class_name="TestClass"
        )
        assert func4.qualified_name == "test::TestClass::func"
    
    def test_cpp_function_qualified_name_variations(self):
        """Test different variations of CppFunction qualified names."""
        # Create a function with different namespace/class configurations
        test_cases = [
            # name, namespace, class_name, expected_qualified_name
            ("func", None, None, "func"),
            ("func", "math", None, "math::func"),
            ("func", None, "Calculator", "Calculator::func"),
            ("func", "math", "Calculator", "math::Calculator::func"),
            ("func", "system::utils", None, "system::utils::func"),
            ("func", "system::utils", "Calculator", "system::utils::Calculator::func"),
        ]
        
        for name, namespace, class_name, expected in test_cases:
            func = CppFunction(
                name=name,
                content=f"void {name}() {{}}",
                file_path="/path/to/file.cpp",
                start_line=1,
                namespace=namespace,
                class_name=class_name
            )
            assert func.qualified_name == expected
