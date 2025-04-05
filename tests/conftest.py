"""
Pytest configuration file for the requirement tracer tests.
"""
import os
import sys
import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def sample_requirements_file():
    """Path to a sample requirements file for testing."""
    return os.path.join(os.path.dirname(__file__), 'data', 'sample_requirements.xlsx')

@pytest.fixture
def sample_cpp_file():
    """Path to a sample C++ file for testing."""
    return os.path.join(os.path.dirname(__file__), 'data', 'sample.cpp')

@pytest.fixture
def sample_cpp_directory():
    """Path to a directory with sample C++ files for testing."""
    return os.path.join(os.path.dirname(__file__), 'data', 'cpp_samples')

@pytest.fixture
def output_directory():
    """Path to a directory for test outputs."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
