"""
Logging utility for the requirement tracing tool.
"""
import logging
import os
import sys
from datetime import datetime

class Logger:
    """
    Centralized logging for the application.
    Handles both console and file-based logging.
    """
    def __init__(self, log_dir="logs", log_level=logging.INFO):
        """
        Initialize the logger.
        
        Args:
            log_dir (str): Directory to store log files
            log_level (int): Logging level (default: INFO)
        """
        self.logger = logging.getLogger("requirement_tracer")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create formatters
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"requirement_tracer_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """Return the configured logger."""
        return self.logger

# Global logger instance
_logger_instance = None

def get_logger(log_dir="logs", log_level=logging.INFO):
    """
    Get or create the global logger instance.
    
    Args:
        log_dir (str): Directory to store log files
        log_level (int): Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger(log_dir, log_level)
    return _logger_instance.get_logger()
