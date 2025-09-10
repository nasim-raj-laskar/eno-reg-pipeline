import os
import yaml
from src.datascience import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a yaml file and returns a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the yaml file.

    Returns:
        ConfigBox: ConfigBox object containing the yaml file contents.
     """
    try:   
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates directories if they don't exist.

    Args:
        path_to_directories (list): List of directories to create.
        verbose (bool, optional): Whether to print the log messages. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves a dictionary as a json file.

    Args:
        path (Path): Path to the json file.
        data (dict): Dictionary to be saved as a json file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads a json file and returns a ConfigBox object.

    Args:
        path (Path): Path to the json file.

    Returns:
        ConfigBox: ConfigBox object containing the json file contents.
    """
    with open(path) as f:
        content = json.load(f)
    
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data as a binary file using joblib.

    Args:
        data (Any): Data to be saved.
        path (Path): Path to the binary file.
    """
    joblib.dump(data, path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads a binary file using joblib.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Data loaded from the binary file.
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded successfully from: {path}")
    return data