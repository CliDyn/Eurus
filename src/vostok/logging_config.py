"""
Vostok Logging Configuration
============================
Centralized logging setup for both web and CLI modes.
Logs are saved to PROJECT_ROOT/logs/ with timestamps.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(mode: str = "web", level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure logging for Vostok.
    
    Args:
        mode: 'web' or 'cli' - determines log file prefix
        level: logging level (default: DEBUG for full logs)
    
    Returns:
        Root logger configured with file and console handlers
    """
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"vostok_{mode}_{timestamp}.log"
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler - FULL DEBUG logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Log startup info
    logger = logging.getLogger("vostok.logging")
    logger.info(f"=" * 80)
    logger.info(f"VOSTOK {mode.upper()} STARTING")
    logger.info(f"Log file: {log_file}")
    logger.info(f"=" * 80)
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


# Cleanup old logs (keep last 20)
def cleanup_old_logs(keep: int = 20):
    """Remove old log files, keeping the most recent ones."""
    try:
        log_files = sorted(LOGS_DIR.glob("vostok_*.log"), key=os.path.getmtime)
        if len(log_files) > keep:
            for old_file in log_files[:-keep]:
                old_file.unlink()
    except Exception:
        pass  # Don't fail on cleanup
