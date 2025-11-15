"""
Logger for the application.
"""

import os
import logging
import sys
import datetime
import yaml
from pathlib import Path

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.INFO)

def get_shared_logger(name: str = "", dirname: str = "logs", filename: str = "app.log") -> logging.Logger:
    """
    Returns a logger with the specified name. If no name is provided, returns the root logger.

    Args:
        name (str): Name of the logger
        dirname (str): Directory to save log files
        filename (str): Base filename for the log files

    Returns:
        logging.Logger: Logger with the specified name
    """

    # Create log directory if it doesn't exist
    os.makedirs(dirname, exist_ok=True)

    # Configure the file handler
    filename_now = filename + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(os.path.join(dirname, filename_now))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    # Ensure no duplicate file handlers
    logger = logging.getLogger(name)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)

    return logger


def load_config(dirname: str = "config", filename: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        dirname (str): Directory containing the configuration file
        filename (str): Name of the configuration file
    Returns:
        dict: Configuration dictionary
    """
    try:
        config_path = Path(dirname) / filename
        os.makedirs(dirname, exist_ok=True)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")