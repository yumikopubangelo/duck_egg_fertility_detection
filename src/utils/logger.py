"""
Logging utilities for egg fertility detection.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """Custom logger class with improved formatting."""
    
    def __init__(self, name: str = 'egg-detector', level: int = logging.INFO):
        """Initialize logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # File handler
            file_handler = None
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            console_handler.setFormatter(formatter)
            if file_handler:
                file_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            if file_handler:
                self.logger.addHandler(file_handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)


def setup_logger(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('egg-detector')
    logger.setLevel(level)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        handlers = [console_handler]
        
        # File handler if specified
        if log_file:
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            handlers.append(file_handler)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    return logger


def log_exception(e: Exception, logger: Optional[logging.Logger] = None) -> None:
    """Log exception with stack trace."""
    if logger:
        logger.error(f"Exception: {e}", exc_info=True)
    else:
        logging.error(f"Exception: {e}", exc_info=True)


def log_metrics(
    logger: logging.Logger,
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_iou: float,
    val_iou: float,
    train_dice: float,
    val_dice: float
) -> None:
    """Log training metrics in a structured format."""
    logger.info(
        f"Epoch {epoch:03d} - "
        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
        f"Train IoU: {train_iou:.3f}, Val IoU: {val_iou:.3f} - "
        f"Train Dice: {train_dice:.3f}, Val Dice: {val_dice:.3f}"
    )


def log_model_summary(
    logger: logging.Logger,
    model_name: str,
    num_parameters: int,
    input_shape: tuple,
    output_shape: tuple
) -> None:
    """Log model architecture summary."""
    logger.info(
        f"Model Summary - {model_name}\n"
        f"Parameters: {num_parameters:,}\n"
        f"Input Shape: {input_shape}\n"
        f"Output Shape: {output_shape}"
    )


def log_data_stats(
    logger: logging.Logger,
    train_size: int,
    val_size: int,
    test_size: int,
    image_size: tuple
) -> None:
    """Log data statistics."""
    logger.info(
        f"Data Statistics\n"
        f"Train: {train_size}, Val: {val_size}, Test: {test_size}\n"
        f"Image Size: {image_size}"
    )
