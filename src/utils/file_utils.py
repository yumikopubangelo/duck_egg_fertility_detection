"""
File utilities for egg fertility detection.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Any, Optional


def list_files(directory: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """List all files in a directory with optional extensions filter."""
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if extensions is None:
        return list(directory.glob('*'))
    
    files = []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(files)


def create_directory(directory: str) -> str:
    """Create directory if it doesn't exist."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: str) -> Any:
    """Load JSON file into Python object."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, file_path: str) -> None:
    """Save data to pickle file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: str) -> Any:
    """Load pickle file into Python object."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def create_directories(*directories: str) -> List[str]:
    """Create multiple directories if they don't exist."""
    created = []
    for directory in directories:
        created.append(create_directory(directory))
    return created


def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return file_path.stat().st_size


def get_file_extension(file_path: str) -> str:
    """Get file extension (lowercase)."""
    return Path(file_path).suffix.lower()


def is_image_file(file_path: str) -> bool:
    """Check if file is an image based on extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return get_file_extension(file_path) in image_extensions


def clean_directory(directory: str, pattern: str = '*') -> int:
    """Clean directory by removing files matching pattern."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    removed = 0
    for file in directory.glob(pattern):
        if file.is_file():
            file.unlink()
            removed += 1
    
    return removed
