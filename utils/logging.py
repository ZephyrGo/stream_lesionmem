"""
Logging utilities for STREAM-LesionMem.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


# Global logger registry
_LOGGERS: dict = {}


def setup_logger(
    name: str = "stream_lesionmem",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_str: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (e.g., logging.INFO, "DEBUG")
        log_file: Optional file path for logging
        format_str: Custom format string
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    global _LOGGERS
    
    # Return existing logger if already configured
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_str is None:
        format_str = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = "stream_lesionmem") -> logging.Logger:
    """
    Get or create a logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _LOGGERS:
        return _LOGGERS[name]
    return setup_logger(name)


class TrainingLogger:
    """
    Training progress logger with metrics tracking.
    """
    
    def __init__(
        self,
        name: str = "training",
        log_file: Optional[Union[str, Path]] = None,
        log_interval: int = 10,
    ):
        """
        Args:
            name: Logger name
            log_file: Optional log file path
            log_interval: Steps between logging
        """
        self.logger = setup_logger(name, log_file=log_file)
        self.log_interval = log_interval
        
        self.step = 0
        self.epoch = 0
        self.metrics_history: list = []
        self._running_metrics: dict = {}
    
    def log_step(
        self,
        step: int,
        metrics: dict,
        prefix: str = "train",
    ) -> None:
        """
        Log training step metrics.
        
        Args:
            step: Current step
            metrics: Dict of metric values
            prefix: Prefix for metric names
        """
        self.step = step
        
        # Accumulate running metrics
        for key, value in metrics.items():
            if key not in self._running_metrics:
                self._running_metrics[key] = []
            self._running_metrics[key].append(value)
        
        # Log at interval
        if step % self.log_interval == 0:
            # Average running metrics
            avg_metrics = {}
            for key, values in self._running_metrics.items():
                avg_metrics[key] = sum(values) / len(values)
            
            # Format message
            metrics_str = " | ".join(
                f"{prefix}/{k}: {v:.4f}" for k, v in avg_metrics.items()
            )
            self.logger.info(f"Step {step} | {metrics_str}")
            
            # Save to history
            self.metrics_history.append({
                "step": step,
                "epoch": self.epoch,
                **{f"{prefix}/{k}": v for k, v in avg_metrics.items()},
            })
            
            # Reset running metrics
            self._running_metrics.clear()
    
    def log_epoch(
        self,
        epoch: int,
        metrics: dict,
        prefix: str = "val",
    ) -> None:
        """
        Log epoch-level metrics.
        
        Args:
            epoch: Current epoch
            metrics: Dict of metric values
            prefix: Prefix for metric names
        """
        self.epoch = epoch
        
        # Format message
        metrics_str = " | ".join(
            f"{prefix}/{k}: {v:.4f}" for k, v in metrics.items()
        )
        self.logger.info(f"Epoch {epoch} | {metrics_str}")
        
        # Save to history
        self.metrics_history.append({
            "step": self.step,
            "epoch": epoch,
            "is_epoch_end": True,
            **{f"{prefix}/{k}": v for k, v in metrics.items()},
        })
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def get_history(self) -> list:
        """Get metrics history."""
        return self.metrics_history


class ProgressTracker:
    """
    Simple progress tracker for inference.
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        log_interval: int = 10,
    ):
        """
        Args:
            total: Total number of items
            desc: Description
            log_interval: Items between logging
        """
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        
        self.current = 0
        self.start_time: Optional[datetime] = None
        self.logger = get_logger("progress")
    
    def start(self) -> None:
        """Start tracking."""
        self.start_time = datetime.now()
        self.logger.info(f"{self.desc}: Starting ({self.total} items)")
    
    def update(self, n: int = 1) -> None:
        """Update progress."""
        self.current += n
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / max(elapsed, 1e-6)
            eta = (self.total - self.current) / max(rate, 1e-6)
            
            self.logger.info(
                f"{self.desc}: {self.current}/{self.total} "
                f"({100 * self.current / self.total:.1f}%) | "
                f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s"
            )
    
    def finish(self) -> None:
        """Finish tracking."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.desc}: Completed {self.total} items in {elapsed:.1f}s"
        )
    
    def __enter__(self) -> "ProgressTracker":
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.finish()


def log_model_info(
    model,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log model parameter information.
    
    Args:
        model: PyTorch model
        logger: Logger to use
    """
    if logger is None:
        logger = get_logger()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"Model Parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Frozen: {frozen_params:,}")
    
    # Log per-module breakdown
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"  {name}: {params:,} params ({trainable:,} trainable)")


def log_gpu_memory(
    logger: Optional[logging.Logger] = None,
    prefix: str = "",
) -> None:
    """
    Log GPU memory usage.
    
    Args:
        logger: Logger to use
        prefix: Prefix for log message
    """
    if logger is None:
        logger = get_logger()
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
                
                logger.info(
                    f"{prefix}GPU {i}: "
                    f"Allocated {allocated:.2f}GB | "
                    f"Reserved {reserved:.2f}GB | "
                    f"Max {max_allocated:.2f}GB"
                )
    except Exception as e:
        logger.warning(f"Could not log GPU memory: {e}")
