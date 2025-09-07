"""Logging configuration with colorful output."""

import logging
import colorlog
import sys
from typing import Optional


def setup_logging(level: str = "INFO", logger_name: Optional[str] = None, 
                 show_timestamp: bool = True, show_logger_name: bool = True) -> logging.Logger:
    """
    Set up colorful logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Name of the logger (None for root logger)
        show_timestamp: Whether to show timestamp in logs
        show_logger_name: Whether to show logger name in logs
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with colorlog
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create format string based on options
    format_parts = []
    if show_timestamp:
        format_parts.append('%(asctime)s')
    if show_logger_name:
        format_parts.append('%(name)s')
    format_parts.append('%(levelname)s')
    
    format_string = '%(log_color)s' + ' '.join(format_parts) + '%(reset)s: %(message)s'
    
    # Create colorful formatter
    formatter = colorlog.ColoredFormatter(
        format_string,
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Also configure root logger to handle propagated messages
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_handler = colorlog.StreamHandler(sys.stdout)
        root_handler.setFormatter(formatter)
        root_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(root_handler)
        root_logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    logger = logging.getLogger(name)
    # Set level to DEBUG to ensure debug messages are processed
    logger.setLevel(logging.DEBUG)
    # Ensure propagation is enabled so messages go to root logger
    logger.propagate = True
    
    # Also ensure root logger and its handlers are set to DEBUG to handle propagated messages
    root_logger = logging.getLogger()
    if root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)
    
    # Update all root handlers to DEBUG level
    for handler in root_logger.handlers:
        if handler.level > logging.DEBUG:
            handler.setLevel(logging.DEBUG)
    
    return logger


# Create a default logger for the application
app_logger = setup_logging("INFO", "topic_evolver")
