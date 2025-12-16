"""
Data preprocessing utilities for video and reports.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
import numpy as np


def preprocess_video(
    video_path: Union[str, Path],
    target_frames: int = 12,
    frame_size: Tuple[int, int] = (336, 336),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess video to extract and resize frames.
    
    Args:
        video_path: Path to video file or frame directory
        target_frames: Number of frames to extract
        frame_size: Target frame size (H, W)
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        frames: Tensor of shape [K, 3, H, W]
    """
    video_path = Path(video_path)
    
    if video_path.is_dir():
        return _preprocess_frame_dir(video_path, target_frames, frame_size, normalize)
    elif video_path.suffix in [".mp4", ".avi", ".mov", ".mkv"]:
        return _preprocess_video_file(video_path, target_frames, frame_size, normalize)
    else:
        # Return dummy frames
        return create_dummy_frames(target_frames, frame_size)


def _preprocess_frame_dir(
    frame_dir: Path,
    target_frames: int,
    frame_size: Tuple[int, int],
    normalize: bool,
) -> torch.Tensor:
    """Load and preprocess frames from directory."""
    frame_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    
    if len(frame_paths) == 0:
        return create_dummy_frames(target_frames, frame_size)
    
    # Uniform sampling
    indices = np.linspace(0, len(frame_paths) - 1, target_frames, dtype=int)
    
    frames = []
    for idx in indices:
        img = Image.open(frame_paths[idx]).convert("RGB")
        img = img.resize((frame_size[1], frame_size[0]))  # PIL uses (W, H)
        
        img_array = np.array(img)
        if normalize:
            img_array = img_array.astype(np.float32) / 255.0
        
        frames.append(torch.from_numpy(img_array).permute(2, 0, 1))
    
    return torch.stack(frames, dim=0)


def _preprocess_video_file(
    video_path: Path,
    target_frames: int,
    frame_size: Tuple[int, int],
    normalize: bool,
) -> torch.Tensor:
    """Load and preprocess frames from video file."""
    try:
        import cv2
    except ImportError:
        print("Warning: cv2 not available. Returning dummy frames.")
        return create_dummy_frames(target_frames, frame_size)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return create_dummy_frames(target_frames, frame_size)
    
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
            
            if normalize:
                frame = frame.astype(np.float32) / 255.0
            
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
    
    cap.release()
    
    if len(frames) == 0:
        return create_dummy_frames(target_frames, frame_size)
    
    # Pad if needed
    while len(frames) < target_frames:
        frames.append(frames[-1].clone())
    
    return torch.stack(frames[:target_frames], dim=0)


def create_dummy_frames(
    num_frames: int,
    frame_size: Tuple[int, int] = (336, 336),
) -> torch.Tensor:
    """
    Create dummy frames for testing.
    
    Returns normalized float tensor in [0, 1].
    """
    frames = torch.rand(num_frames, 3, frame_size[0], frame_size[1])
    return frames


def preprocess_report(
    report_text: str,
    section_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Parse report text into sections.
    
    Args:
        report_text: Full report text
        section_names: Expected section names
    
    Returns:
        Dict mapping section_name -> section_text
    """
    if section_names is None:
        section_names = [
            "esophagus", "gastroesophageal_junction", "cardia",
            "fundus", "body", "antrum", "pylorus", "duodenum"
        ]
    
    sections = {}
    
    # Try to parse structured report
    for section_name in section_names:
        # Try different patterns
        patterns = [
            rf"{section_name}[:\s]+(.+?)(?=\n[A-Za-z_]+[:\s]|\Z)",
            rf"{section_name.replace('_', ' ')}[:\s]+(.+?)(?=\n[A-Za-z_]+[:\s]|\Z)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, report_text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
                break
        
        if section_name not in sections:
            sections[section_name] = ""
    
    return sections


def normalize_section_text(text: str) -> str:
    """Normalize section text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def extract_findings(text: str) -> List[str]:
    """
    Extract individual findings from section text.
    
    Returns list of finding strings.
    """
    # Split by sentence
    sentences = re.split(r'[.!?]+', text)
    findings = [s.strip() for s in sentences if s.strip()]
    return findings


def compute_pixel_values(
    frames: torch.Tensor,
    mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
) -> torch.Tensor:
    """
    Compute normalized pixel values for vision model.
    
    Args:
        frames: Tensor of shape [K, 3, H, W] or [B, K, 3, H, W] in range [0, 1]
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Normalized tensor of same shape
    """
    mean_tensor = torch.tensor(mean, device=frames.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=frames.device).view(1, 3, 1, 1)
    
    if frames.dim() == 5:
        # [B, K, 3, H, W]
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    
    return (frames - mean_tensor) / std_tensor


def prepare_pixel_values_for_medgemma(
    frames: torch.Tensor,
    processor: Optional[object] = None,
) -> torch.Tensor:
    """
    Prepare pixel values for MedGemma input.
    
    If processor is provided, uses its image processor.
    Otherwise, applies default CLIP normalization.
    
    Args:
        frames: [B, K, 3, H, W] or [K, 3, H, W] in range [0, 1]
        processor: Optional MedGemma processor
    
    Returns:
        Normalized pixel values ready for vision_tower
    """
    if processor is not None and hasattr(processor, "image_processor"):
        # Use processor's normalization
        # This would require converting to PIL and back
        # For efficiency, we do manual normalization matching processor config
        try:
            mean = processor.image_processor.image_mean
            std = processor.image_processor.image_std
            return compute_pixel_values(frames, tuple(mean), tuple(std))
        except AttributeError:
            pass
    
    # Default CLIP/SigLIP normalization
    return compute_pixel_values(frames)
