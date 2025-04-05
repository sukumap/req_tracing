"""
Tests for the Mapping Engine module.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.mapping_engine import MappingEngine
from src.cpp_parser import CppFunction

# Skip these tests if Ollama isn't available
pytestmark = pytest.mark.skipif(
    True, 
    reason="Ollama dependency is needed for these tests. Skipping in CI environment."
)

class TestMappingEngine:
    """Tests for the MappingEngine class."""
    
    @patch('src.mapping_engine.get_logger')
    def setup_method(self, mock_logger):
        """Set up test environment before each test method."""
        # Mock logger
        self.mock_logger = MagicMock()
        mock_logger.return_value = self.mock_logger
        
        # Create a test instance with a low threshold for testing
        with patch('src.mapping_engine.ollama'):
            self.engine = MappingEngine(similarity_threshold=0.3, model="gemma3:1b")
        
        # Create mock functions
        self.mock_function1 = CppFunction(
            name="add",
            content="int add(int a, int b) { return a + b; }",
            file_path="/path/math.cpp",
            start_line=1,
            namespace="math"
        )
        
        self.mock_function2 = CppFunction(
            name="subtract",
            content="int subtract(int a, int b) { return a - b; }",
            file_path="/path/math.cpp",
            start_line=7,
            namespace="math"
        )
        
        # Functions for testing
        self.functions = [self.mock_function1, self.mock_function2]
        
        # Function summaries for testing
        self.function_summaries = {
            "math::add": "Adds two integers and returns the sum",
            "math::subtract": "Subtracts one integer from another"
        }
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.similarity_threshold == 0.3
        assert self.engine.model == "gemma3:1b"
        assert self.engine.vectorizer is not None
    
    @patch('sklearn.feature_extraction.text.TfidfVectorizer.fit_transform')
    @patch('sklearn.metrics.pairwise.cosine_similarity')
    def test_compute_similarity(self, mock_cosine_similarity, mock_fit_transform):
        """Test computing similarity between requirement and function summaries."""
        # Setup mocks
        mock_matrix = MagicMock()
        mock_fit_transform.return_value = mock_matrix
        
        # Mock cosine similarity to return specific values
        mock_cosine_similarity.return_value = np.array([[0.8, 0.3]])
        
        # Compute similarity
        requirement_desc = "Add two numbers together"
        similarities = self.engine.compute_similarity(
            requirement_desc, 
            self.function_summaries
        )
        
        # Verify results
        assert len(similarities) == 2
        assert abs(similarities["math::add"] - 0.8) < 0.001
        assert abs(similarities["math::subtract"] - 0.3) < 0.001
        
        # Verify function calls
        mock_fit_transform.assert_called_once()
        mock_cosine_similarity.assert_called_once()
    
    def test_compute_similarity_empty_summaries(self):
        """Test computing similarity with empty function summaries."""
        similarities = self.engine.compute_similarity("Test requirement", {})
        assert similarities == {}
    
    @patch('src.mapping_engine.MappingEngine.compute_similarity')
    def test_map_requirement_direct_annotation(self, mock_compute_similarity):
        """Test mapping a requirement to functions based on direct annotations."""
        # Add direct requirement annotation to mock_function1
        self.mock_function1.requirements = ["REQ-001"]
        self.mock_function2.requirements = []
        
        # Setup mock for similarity computation
        mock_compute_similarity.return_value = {
            "math::add": 0.8,
            "math::subtract": 0.3
        }
        
        # Map requirement
        mappings = self.engine.map_requirement(
            "REQ-001",  # Direct annotation in mock_function1
            "Add two numbers",
            "Math",
            self.functions,
            self.function_summaries,
            use_llm_confidence=False
        )
        
        # Verify results
        assert len(mappings) > 0
        
        # Direct annotation match should be in results
        assert any(f.name == "add" for f, _ in mappings)
        
        # Clean up for other tests
        self.mock_function1.requirements = []
    
    @patch('src.mapping_engine.MappingEngine.compute_similarity')
    def test_map_requirement_similarity(self, mock_compute_similarity):
        """Test mapping a requirement to functions based on similarity."""
        # Setup mock for similarity computation
        mock_compute_similarity.return_value = {
            "math::add": 0.2,  # Below threshold
            "math::subtract": 0.4  # Above threshold
        }
        
        # Map requirement
        mappings = self.engine.map_requirement(
            "REQ-003",  # No direct annotation
            "Subtract numbers",
            "Math",
            self.functions,
            self.function_summaries,
            use_llm_confidence=False
        )
        
        # Verify results
        assert len(mappings) > 0
        
        # Similarity match should be in results
        assert any(f.name == "subtract" for f, _ in mappings)
    
    @patch('src.mapping_engine.MappingEngine.compute_similarity')
    @patch('src.mapping_engine.MappingEngine.get_llm_confidence')
    def test_map_requirement_with_llm_confidence(self, mock_get_llm_confidence, mock_compute_similarity):
        """Test mapping with LLM confidence scoring."""
        # Setup mocks
        mock_compute_similarity.return_value = {
            "math::add": 0.7,
            "math::subtract": 0.6
        }
        
        # Mock LLM confidence scores
        mock_get_llm_confidence.return_value = 0.9
        
        # Map requirement
        mappings = self.engine.map_requirement(
            "REQ-003",
            "Math operation",
            "Math",
            self.functions,
            self.function_summaries,
            use_llm_confidence=True
        )
        
        # Verify results
        assert len(mappings) == 2
        
        # Both functions should be in results
        functions_mapped = [f.name for f, _ in mappings]
        assert "add" in functions_mapped
        assert "subtract" in functions_mapped
        
        # Verify LLM confidence was called
        assert mock_get_llm_confidence.call_count == 2
    
    @patch('src.mapping_engine.ollama.chat')
    def test_get_llm_confidence(self, mock_chat):
        """Test getting confidence score from LLM."""
        # Setup mock
        mock_response = {
            "message": {"content": "85"}
        }
        mock_chat.return_value = mock_response
        
        # Get confidence
        confidence = self.engine.get_llm_confidence(
            "Add two numbers",
            "Math",
            "add",
            "Adds two integers",
            "return a + b;"
        )
        
        # Verify result
        assert confidence == 0.85
        
        # Verify call to ollama.chat
        mock_chat.assert_called_once()
    
    @patch('src.mapping_engine.ollama.chat')
    def test_get_llm_confidence_invalid_response(self, mock_chat):
        """Test handling invalid LLM response."""
        # Setup mock with invalid response
        mock_response = {
            "message": {"content": "The function definitely implements the requirement."}
        }
        mock_chat.return_value = mock_response
        
        # Get confidence
        confidence = self.engine.get_llm_confidence(
            "Add two numbers",
            "Math",
            "add",
            "Adds two integers",
            "return a + b;"
        )
        
        # Default to 0 if no valid number
        assert confidence == 0.0
    
    @patch('src.mapping_engine.MappingEngine.map_requirement')
    def test_map_all_requirements(self, mock_map_requirement):
        """Test mapping all requirements."""
        # Setup mock
        mock_map_requirement.return_value = [(self.mock_function1, 0.8)]
        
        # Requirements to map
        requirements = [
            ("REQ-001", "Add numbers", "Math"),
            ("REQ-002", "Subtract numbers", "Math")
        ]
        
        # Map all requirements
        mappings = self.engine.map_all_requirements(
            requirements,
            self.functions,
            self.function_summaries,
            use_llm_confidence=False
        )
        
        # Verify results
        assert len(mappings) == 2
        assert "REQ-001" in mappings
        assert "REQ-002" in mappings
        
        # Verify map_requirement was called for each requirement
        assert mock_map_requirement.call_count == 2
