"""
I/O utilities for STREAM-LesionMem.

Handles JSONL, YAML, JSON, and other file formats.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save dict to JSON file.
    
    Args:
        data: Dict to save
        path: Output path
        indent: JSON indentation
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSONL file.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of dicts
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    Iterate JSONL file line by line (memory efficient).
    
    Args:
        path: Path to JSONL file
        
    Yields:
        Dicts from each line
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(
    data: List[Dict[str, Any]],
    path: Union[str, Path],
    append: bool = False,
) -> None:
    """
    Save list of dicts to JSONL file.
    
    Args:
        data: List of dicts
        path: Output path
        append: Whether to append to existing file
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(
    item: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """
    Append single item to JSONL file.
    
    Args:
        item: Dict to append
        path: Output path
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Loaded dict
    """
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        raise ImportError("PyYAML not installed. Run: pip install pyyaml")


def save_yaml(
    data: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """
    Save dict to YAML file.
    
    Args:
        data: Dict to save
        path: Output path
    """
    try:
        import yaml
        path = Path(path)
        ensure_dir(path.parent)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except ImportError:
        raise ImportError("PyYAML not installed. Run: pip install pyyaml")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load config file (YAML or JSON).
    
    Args:
        path: Path to config file
        
    Returns:
        Config dict
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix in (".yaml", ".yml"):
        return load_yaml(path)
    elif suffix == ".json":
        return load_json(path)
    else:
        # Try YAML first, then JSON
        try:
            return load_yaml(path)
        except Exception:
            return load_json(path)


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deep merge two config dicts.
    
    Args:
        base: Base config
        override: Override config (takes precedence)
        
    Returns:
        Merged config
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_checkpoint_path(
    output_dir: Union[str, Path],
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    best: bool = False,
) -> Path:
    """
    Get checkpoint path.
    
    Args:
        output_dir: Output directory
        epoch: Epoch number
        step: Step number
        best: Whether this is the best checkpoint
        
    Returns:
        Checkpoint path
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    if best:
        return output_dir / "checkpoint_best.pt"
    elif epoch is not None:
        return output_dir / f"checkpoint_epoch{epoch}.pt"
    elif step is not None:
        return output_dir / f"checkpoint_step{step}.pt"
    else:
        return output_dir / "checkpoint_latest.pt"


def find_latest_checkpoint(
    output_dir: Union[str, Path],
) -> Optional[Path]:
    """
    Find latest checkpoint in directory.
    
    Args:
        output_dir: Directory to search
        
    Returns:
        Path to latest checkpoint or None
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        return None
    
    # Check for latest checkpoint
    latest = output_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest
    
    # Find by epoch
    epoch_checkpoints = list(output_dir.glob("checkpoint_epoch*.pt"))
    if epoch_checkpoints:
        # Sort by epoch number
        def get_epoch(p: Path) -> int:
            try:
                return int(p.stem.replace("checkpoint_epoch", ""))
            except ValueError:
                return -1
        
        epoch_checkpoints.sort(key=get_epoch)
        return epoch_checkpoints[-1]
    
    # Find by step
    step_checkpoints = list(output_dir.glob("checkpoint_step*.pt"))
    if step_checkpoints:
        def get_step(p: Path) -> int:
            try:
                return int(p.stem.replace("checkpoint_step", ""))
            except ValueError:
                return -1
        
        step_checkpoints.sort(key=get_step)
        return step_checkpoints[-1]
    
    return None


class ResultsWriter:
    """
    Incremental results writer for JSONL output.
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        overwrite: bool = True,
    ):
        """
        Args:
            output_path: Output JSONL path
            overwrite: Whether to overwrite existing file
        """
        self.output_path = Path(output_path)
        ensure_dir(self.output_path.parent)
        
        if overwrite and self.output_path.exists():
            self.output_path.unlink()
        
        self._count = 0
    
    def write(self, item: Dict[str, Any]) -> None:
        """Write single result."""
        append_jsonl(item, self.output_path)
        self._count += 1
    
    def write_batch(self, items: List[Dict[str, Any]]) -> None:
        """Write batch of results."""
        for item in items:
            self.write(item)
    
    @property
    def count(self) -> int:
        """Number of items written."""
        return self._count
    
    def __enter__(self) -> "ResultsWriter":
        return self
    
    def __exit__(self, *args) -> None:
        pass
