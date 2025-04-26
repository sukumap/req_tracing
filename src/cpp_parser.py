"""
C++ parsing module for extracting functions from C++ source files.
"""
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Union

from src.utils.logger import get_logger

logger = get_logger()

# Try to import tree-sitter, but handle case when it's not available
try:
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    logger.warning("tree-sitter package not found. Using fallback regex parser instead.")
    TREE_SITTER_AVAILABLE = False

@dataclass
class CppFunction:
    """Class to store C++ function information."""
    name: str
    content: str
    file_path: str
    start_line: int
    class_name: Optional[str] = None
    namespace: Optional[str] = None
    requirements: List[str] = None
    
    def __post_init__(self):
        """Initialize requirements list if not provided."""
        if self.requirements is None:
            self.requirements = []
            
            # Extract requirements from comments
            req_pattern = r'@requirement\s+(REQ-[A-Z]+-\d+)'
            for match in re.finditer(req_pattern, self.content, re.IGNORECASE):
                self.requirements.append(match.group(1))
    
    @property
    def qualified_name(self):
        """Get the fully qualified function name."""
        parts = []
        if self.namespace:
            parts.append(self.namespace)
        if self.class_name:
            parts.append(self.class_name)
        parts.append(self.name)
        return "::".join(parts)
    
    def __str__(self):
        return f"{self.qualified_name} ({os.path.basename(self.file_path)}:{self.start_line})"


class CppParser:
    """
    Parser for C++ code that extracts functions and their content.
    Uses tree-sitter for accurate parsing if available, falls back to regex otherwise.
    """
    
    def __init__(self, use_fallback=False):
        """
        Initialize the C++ parser.
        
        Args:
            use_fallback (bool): Force using fallback regex parser instead of tree-sitter
        """
        self.use_fallback = use_fallback or not TREE_SITTER_AVAILABLE
        
        if not self.use_fallback:
            try:
                self._initialize_tree_sitter()
            except Exception as e:
                logger.warning(f"Failed to initialize tree-sitter, falling back to regex parser: {str(e)}")
                self.use_fallback = True
        
        if self.use_fallback:
            logger.info("Using fallback regex-based C++ parser")
    
    def _initialize_tree_sitter(self):
        """Initialize tree-sitter parser if available."""
        try:
            # Load the C++ language for tree-sitter
            cpp_language_path = os.path.join(os.path.expanduser('~'), '.tree-sitter', 'tree-sitter-cpp.so')
            
            if not os.path.exists(cpp_language_path):
                # Try to build the language if not found
                logger.info("C++ tree-sitter grammar not found, attempting to build...")
                self._build_cpp_language()
            
            self.CPP_LANGUAGE = Language(cpp_language_path, 'cpp')
            self.parser = Parser()
            self.parser.set_language(self.CPP_LANGUAGE)
            logger.info("Successfully initialized C++ parser with tree-sitter")
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter parser: {str(e)}")
            raise
    
    def _build_cpp_language(self):
        """Build the C++ language for tree-sitter if not available."""
        import subprocess
        
        try:
            # Create directory if it doesn't exist
            tree_sitter_dir = os.path.join(os.path.expanduser('~'), '.tree-sitter')
            os.makedirs(tree_sitter_dir, exist_ok=True)
            
            # Clone the repository if needed
            cpp_repo_path = os.path.join(tree_sitter_dir, 'tree-sitter-cpp')
            if not os.path.exists(cpp_repo_path):
                logger.info("Cloning tree-sitter-cpp repository...")
                subprocess.run(
                    ['git', 'clone', 'https://github.com/tree-sitter/tree-sitter-cpp.git', cpp_repo_path],
                    check=True
                )
            
            # Try different build methods depending on tree-sitter version
            logger.info("Building tree-sitter-cpp language...")
            
            try:
                # Try newer API first
                from tree_sitter import Parser
                parser = Parser()
                parser.set_language(Language.build_library(
                    os.path.join(tree_sitter_dir, 'tree-sitter-cpp.so'),
                    [cpp_repo_path]
                ))
            except AttributeError:
                # Try alternative method for newer versions
                try:
                    Language.build_library(
                        os.path.join(tree_sitter_dir, 'tree-sitter-cpp.so'),
                        [cpp_repo_path]
                    )
                except AttributeError:
                    # For even newer versions that use compiled_path
                    try:
                        Language.compiled_path(
                            os.path.join(tree_sitter_dir, 'tree-sitter-cpp.so'),
                            [cpp_repo_path]
                        )
                    except (AttributeError, TypeError):
                        # Last resort - build using py-tree-sitter-languages
                        try:
                            import tree_sitter_languages
                            tree_sitter_languages.get('cpp')
                            logger.info("Using pre-built tree-sitter language from tree_sitter_languages package")
                        except ImportError:
                            logger.error("Could not build or find tree-sitter C++ language parser")
                            raise RuntimeError("Failed to build or find tree-sitter C++ language")
            
            logger.info("Successfully built tree-sitter-cpp language")
        except Exception as e:
            logger.error(f"Failed to build tree-sitter-cpp language: {str(e)}")
            raise

    def _extract_function_content(self, source_code, node, file_path):
        """
        Extract the complete function content from the source code.
        
        Args:
            source_code (bytes): The source code of the C++ file
            node (Node): The tree-sitter node representing the function
            file_path (str): The path to the C++ file
            
        Returns:
            CppFunction: Object containing function details
        """
        start_byte = node.start_byte
        end_byte = node.end_byte
        content = source_code[start_byte:end_byte].decode('utf-8')
        
        # Count the number of newlines up to the start_byte to determine the line number
        start_line = source_code[:start_byte].decode('utf-8').count('\n') + 1
        
        # Get function name
        function_name = "unknown"
        for child in node.children:
            if child.type == 'function_declarator':
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        function_name = source_code[grandchild.start_byte:grandchild.end_byte].decode('utf-8')
                        break
                break
        
        # Try to find parent class if any
        class_name = None
        namespace = None
        
        # Search for parent class or namespace
        current = node.parent
        while current:
            if current.type == 'class_specifier':
                for child in current.children:
                    if child.type == 'name':
                        class_name = source_code[child.start_byte:child.end_byte].decode('utf-8')
                        break
            elif current.type == 'namespace_definition':
                for child in current.children:
                    if child.type == 'identifier':
                        namespace_part = source_code[child.start_byte:child.end_byte].decode('utf-8')
                        namespace = namespace_part if namespace is None else f"{namespace_part}::{namespace}"
                        break
            current = current.parent
        
        return CppFunction(
            name=function_name,
            content=content,
            file_path=file_path,
            start_line=start_line,
            class_name=class_name,
            namespace=namespace
        )
    
    def _parse_file_with_tree_sitter(self, file_path):
        """
        Parse a C++ file using tree-sitter.
        
        Args:
            file_path (str): Path to the C++ file
            
        Returns:
            List[CppFunction]: List of functions found in the file
        """
        with open(file_path, 'rb') as f:
            source_code = f.read()
        
        tree = self.parser.parse(source_code)
        
        functions = []
        
        # Query for function definitions
        query_string = """
        (function_definition) @function
        (method_definition) @method
        """
        
        query = self.CPP_LANGUAGE.query(query_string)
        captures = query.captures(tree.root_node)
        
        for capture in captures:
            node, _ = capture
            function = self._extract_function_content(source_code, node, file_path)
            functions.append(function)
        
        logger.info(f"Extracted {len(functions)} functions from {file_path} using tree-sitter")
        return functions
    
    def _parse_file_with_regex(self, file_path):
        """
        Parse a C++ file using regex (fallback method).
        
        Args:
            file_path (str): Path to the C++ file
            
        Returns:
            List[CppFunction]: List of functions found in the file
        """
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        functions = []
        
        # Extract function declarations (in header files)
        declaration_pattern = r'(?:(?:virtual|static|inline|extern)\s+)?(?:[\w:]+\s+)+(\w+)\s*\([^)]*\)\s*(?:const|override|final|noexcept)?\s*(?:=\s*0)?;'
        
        # Extract function definitions (in cpp files)
        definition_pattern = r'(?:(?:virtual|static|inline|extern)\s+)?(?:[\w:]+\s+)+(\w+)\s*\([^)]*\)\s*(?:const|override|final|noexcept)?\s*(?:=\s*0)?\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}'
        
        # First, scan the file for comment blocks with requirement tags
        comment_req_map = {}
        comment_pattern = r'/\*\*(.*?)\*/'
        for comment_match in re.finditer(comment_pattern, content, re.DOTALL):
            comment_end_pos = comment_match.end()
            
            # Look for requirement tag in this comment
            req_pattern = r'@requirement\s+(REQ-[A-Z]+-\d+)'
            req_matches = list(set(re.findall(req_pattern, comment_match.group(1))))  # Use set to remove duplicates
            
            if req_matches:
                # Store the requirement IDs along with the position after the comment
                comment_req_map[comment_end_pos] = req_matches
        
        # Keep track of functions we've already processed to avoid duplicates
        processed_funcs = set()
        
        # Process both declarations and definitions
        for pattern, is_definition in [(declaration_pattern, False), (definition_pattern, True)]:
            for match in re.finditer(pattern, content):
                try:
                    func_content = match.group(0)
                    func_name = match.group(1)
                    start_pos = match.start()
                    
                    # Skip duplicate functions (e.g., definition after declaration)
                    if func_name in processed_funcs:
                        continue
                    
                    processed_funcs.add(func_name)
                    
                    # Calculate line number
                    start_line = content[:start_pos].count('\n') + 1
                    
                    # Look for the comment block above the function
                    comment_pos = content[:start_pos].rfind('/**')
                    comment_requirements = []
                    
                    if comment_pos >= 0:
                        comment_end = content[:start_pos].rfind('*/')
                        if comment_end > comment_pos:
                            comment_block = content[comment_pos:comment_end + 2]
                            
                            # Include comment block in function content
                            func_content = comment_block + '\n' + func_content
                            
                            # Check if this comment has requirements
                            for end_pos, req_ids in comment_req_map.items():
                                if abs(end_pos - comment_end - 2) < 5:  # Small tolerance for position differences
                                    comment_requirements.extend(req_ids)
                    
                    # Extract namespace (basic check)
                    namespace = None
                    namespace_pattern = r'namespace\s+(\w+)'
                    namespace_matches = re.findall(namespace_pattern, content[:start_pos])
                    if namespace_matches:
                        namespace = "::".join(namespace_matches)
                    
                    # Extract class (basic check)
                    class_name = None
                    class_pattern = r'class\s+(\w+)'
                    class_matches = re.findall(class_pattern, content[:start_pos])
                    if class_matches:
                        class_name = class_matches[-1]  # Take the last one as it's likely the closest
                    
                    # Create the function object
                    function = CppFunction(
                        name=func_name,
                        content=func_content,
                        file_path=file_path,
                        start_line=start_line,
                        class_name=class_name,
                        namespace=namespace
                    )
                    
                    # Add requirements found in the comment (ensuring no duplicates)
                    function.requirements = list(set(comment_requirements))
                    
                    functions.append(function)
                
                except Exception as e:
                    logger.error(f"Error parsing function at line {content[:start_pos].count('\n') + 1}: {str(e)}")
        
        logger.info(f"Extracted {len(functions)} functions from {file_path} using regex fallback")
        return functions

    def parse_file(self, file_path):
        """
        Parse a C++ file and extract all functions.
        
        Args:
            file_path (str): Path to the C++ file
            
        Returns:
            List[CppFunction]: List of functions found in the file
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return []
            
            if self.use_fallback:
                return self._parse_file_with_regex(file_path)
            else:
                return self._parse_file_with_tree_sitter(file_path)
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            
            # If tree-sitter failed, try fallback
            if not self.use_fallback:
                logger.info(f"Attempting fallback parsing for {file_path}")
                try:
                    return self._parse_file_with_regex(file_path)
                except Exception as e2:
                    logger.error(f"Fallback parsing also failed for {file_path}: {str(e2)}")
            
            return []

    def parse_directory(self, directory, extensions=None):
        """
        Recursively parse all C++ files in a directory.
        
        Args:
            directory (str): Path to the directory containing C++ files
            extensions (List[str], optional): List of file extensions to consider.
                                             Defaults to ['.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx']
                                             
        Returns:
            List[CppFunction]: List of all functions found in the directory with
            implementations prioritized over declarations
        """
        if extensions is None:
            extensions = ['.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx']
        
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Divide extensions into implementation and header files
        impl_extensions = ['.cpp', '.cc', '.cxx']
        header_extensions = ['.h', '.hpp', '.hxx']
        
        # First collect all functions
        all_functions = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    all_functions.extend(self.parse_file(file_path))
        
        # Group functions by qualified name
        function_map = {}
        for func in all_functions:
            # Determine if this is an implementation or declaration
            is_implementation = any(func.file_path.endswith(ext) for ext in impl_extensions)
            
            if func.qualified_name in function_map:
                existing_func = function_map[func.qualified_name]
                existing_is_impl = any(existing_func.file_path.endswith(ext) for ext in impl_extensions)
                
                # Replace only if current is implementation and existing is header
                if is_implementation and not existing_is_impl:
                    function_map[func.qualified_name] = func
                    logger.debug(f"Prioritizing implementation of {func.qualified_name} from {func.file_path} over declaration")
            else:
                function_map[func.qualified_name] = func
        
        # Get the prioritized functions
        prioritized_functions = list(function_map.values())
        
        # Count implementations vs declarations for logging
        impl_count = sum(1 for f in prioritized_functions if any(f.file_path.endswith(ext) for ext in impl_extensions))
        header_count = len(prioritized_functions) - impl_count
        
        logger.info(f"Total of {len(all_functions)} functions found in {directory}")
        logger.info(f"After prioritization: {impl_count} implementations, {header_count} declarations")
        
        return prioritized_functions
