# C++ Codebase Requirement Tracing Tool

A Python-based tool that traces requirements from an Excel file to functions in a C++ codebase using semantic mapping powered by a locally deployed Gemma 3:1b model via Ollama.

## Overview

This tool helps developers and project managers map requirements to their actual implementations in C++ code by:

1. Reading requirements from an Excel file
2. Parsing C++ code to extract functions
3. Using Gemma 3:1b (via Ollama) to generate function summaries
4. Computing semantic similarity between requirements and function summaries
5. Generating an Excel report that links requirements to implementing functions

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd requirement-tracer

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python -m src.main --requirements path/to/requirements.xlsx --codebase path/to/cpp/code --output path/to/output.xlsx
```

### Parameters

- `--requirements`: Path to Excel file containing requirements
- `--codebase`: Path to directory containing C++ code
- `--output`: Path to save the output Excel report
- `--threshold` (optional): Similarity threshold for matching (default: 0.6)
- `--model` (optional): Ollama model name (default: "gemma3:1b")

## Input Format

The requirements Excel file must contain the following columns:
- **Requirement ID**
- **Requirement Description**
- **Module Id**

## Output Format

The output Excel file contains:
- A summary sheet with all requirements and their linked functions
- For requirements with more than 30 linked functions, separate tabs with detailed listings
- Hyperlinks from the summary sheet to the detailed tabs

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/
```

## License

[License Information]
