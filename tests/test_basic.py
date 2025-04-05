"""
Basic tests for the requirement tracer project.
This file demonstrates how to use the fixtures defined in conftest.py.
"""
import os

def test_sample_requirements_file_exists(sample_requirements_file):
    """Test that the sample requirements file exists."""
    assert os.path.exists(sample_requirements_file)
    
def test_sample_cpp_file_exists(sample_cpp_file):
    """Test that the sample C++ file exists."""
    assert os.path.exists(sample_cpp_file)
    
def test_sample_cpp_file_contains_requirements(sample_cpp_file):
    """Test that the sample C++ file contains requirement annotations."""
    with open(sample_cpp_file, 'r') as f:
        content = f.read()
    assert '@requirement REQ-001' in content
    assert '@requirement REQ-002' in content
