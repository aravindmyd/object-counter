import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


def setup_logger(
    name=None,
    log_level=logging.INFO,
    log_dir="logs",
    console_output=True,
    file_output=True,
    file_size_limit=10 * 1024 * 1024,  # 10 MB
    backup_count=5,
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    """
    Set up and configure a logger instance.

    Args:
        name (str, optional): Logger name. Defaults to None (root logger).
        log_level (int, optional): Logging level. Defaults to logging.INFO.
        log_dir (str, optional): Directory for log files. Defaults to "logs".
        console_output (bool, optional): Enable console output. Defaults to True.
        file_output (bool, optional): Enable file output. Defaults to True.
        file_size_limit (int, optional): Max log file size in bytes. Defaults to 10MB.
        backup_count (int, optional): Number of backup files to keep. Defaults to 5.
        log_format (str, optional): Format for log messages.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Add console handler if enabled
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if enabled
    if file_output:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Determine log file path
        log_file = os.path.join(log_dir, f"{name or 'app'}.log")

        # Set up rotating file handler (rotates by size)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=file_size_limit, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optional: Add a time-based rotating handler (rotates daily)
        daily_file = os.path.join(log_dir, f"{name or 'app'}_daily.log")
        daily_handler = TimedRotatingFileHandler(
            daily_file, when="midnight", backupCount=30  # Keep a month of daily logs
        )
        daily_handler.setFormatter(formatter)
        logger.addHandler(daily_handler)

    return logger


def get_api_logger():
    """Get logger configured for API modules"""
    return setup_logger(
        name="api",
        log_level=logging.INFO,
        log_format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s - (%(filename)s:%(lineno)d)",
    )


def get_db_logger():
    """Get logger configured for database operations"""
    return setup_logger(
        name="db",
        log_level=logging.INFO,
        log_format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s - (%(filename)s:%(lineno)d)",
    )


def get_model_logger():
    """Get logger configured for ML model operations"""
    return setup_logger(
        name="model",
        log_level=logging.INFO,
        log_format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s - (%(filename)s:%(lineno)d)",
    )
