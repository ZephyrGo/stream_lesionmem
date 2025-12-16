"""
Dataset classes for endoscopy video report generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# TODO: Replace with real video loading
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class EndoscopyDataset(Dataset):
    """
    Dataset for endoscopy video report generation.
    
    Each sample contains:
    - video_path: Path to video file or frame directory
    - report: Ground truth report text
    - sections: Dict mapping section_name -> section_text
    - frame2section: Optional mapping frame_idx -> section_id
    """
    
    def __init__(
        self,
        jsonl_path: str,
        video_dir: str,
        frame_size: Tuple[int, int] = (336, 336),
        max_frames: int = 16,
        transform: Optional[Any] = None,
    ):
        """
        Args:
            jsonl_path: Path to JSONL file with annotations
            video_dir: Directory containing videos/frames
            frame_size: Target frame size (H, W)
            max_frames: Maximum frames to load per video
            transform: Optional transform to apply to frames
        """
        self.video_dir = Path(video_dir)
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.transform = transform
        
        self.samples: List[Dict] = []
        self._load_annotations(jsonl_path)
    
    def _load_annotations(self, jsonl_path: str) -> None:
        """Load annotations from JSONL file."""
        jsonl_path = Path(jsonl_path)
        
        if not jsonl_path.exists():
            # Create dummy data for testing
            print(f"Warning: {jsonl_path} not found. Creating dummy data.")
            self.samples = self._create_dummy_samples()
            return
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
    
    def _create_dummy_samples(self, num_samples: int = 10) -> List[Dict]:
        """Create dummy samples for testing."""
        dummy_samples = []
        section_names = [
            "esophagus", "gastroesophageal_junction", "cardia",
            "fundus", "body", "antrum", "pylorus", "duodenum"
        ]
        
        for i in range(num_samples):
            sections = {}
            for sec_name in section_names:
                if np.random.random() > 0.7:  # 30% chance of abnormal
                    sections[sec_name] = f"Abnormal finding: small lesion observed in {sec_name}."
                else:
                    sections[sec_name] = f"Normal appearance of {sec_name}. No abnormalities detected."
            
            full_report = "\n".join([f"{k}: {v}" for k, v in sections.items()])
            
            dummy_samples.append({
                "sample_id": f"dummy_{i:04d}",
                "video_path": f"video_{i:04d}",
                "report": full_report,
                "sections": sections,
                "frame2section": None,  # Will be inferred or provided
            })
        
        return dummy_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load frames
        frames = self._load_frames(sample.get("video_path", ""))
        
        # Get frame-to-section mapping
        frame2section = sample.get("frame2section", None)
        if frame2section is None:
            # Default: evenly distribute frames across sections
            num_frames = frames.shape[0] if isinstance(frames, torch.Tensor) else len(frames)
            num_sections = len(sample.get("sections", {})) or 8
            frame2section = [i * num_sections // num_frames for i in range(num_frames)]
        
        return {
            "sample_id": sample.get("sample_id", f"sample_{idx}"),
            "frames": frames,  # [K, 3, H, W]
            "report": sample.get("report", ""),
            "sections": sample.get("sections", {}),
            "frame2section": frame2section,
        }
    
    def _load_frames(self, video_path: str) -> torch.Tensor:
        """
        Load frames from video or frame directory.
        
        Returns:
            frames: Tensor of shape [K, 3, H, W] in range [0, 1]
        """
        full_path = self.video_dir / video_path
        
        # Try to load real frames
        if full_path.exists():
            if full_path.is_dir():
                frames = self._load_frames_from_dir(full_path)
            elif full_path.suffix in [".mp4", ".avi", ".mov"]:
                frames = self._load_frames_from_video(full_path)
            else:
                frames = self._create_dummy_frames()
        else:
            # Create dummy frames for testing
            frames = self._create_dummy_frames()
        
        if self.transform is not None:
            frames = self.transform(frames)
        
        return frames
    
    def _load_frames_from_dir(self, frame_dir: Path) -> torch.Tensor:
        """Load frames from a directory of images."""
        frame_paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        
        if len(frame_paths) == 0:
            return self._create_dummy_frames()
        
        # Sample frames
        indices = np.linspace(0, len(frame_paths) - 1, self.max_frames, dtype=int)
        
        frames = []
        for idx in indices:
            img = Image.open(frame_paths[idx]).convert("RGB")
            img = img.resize(self.frame_size[::-1])  # PIL uses (W, H)
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            frames.append(img_tensor)
        
        return torch.stack(frames, dim=0)
    
    def _load_frames_from_video(self, video_path: Path) -> torch.Tensor:
        """Load frames from video file."""
        if not HAS_CV2:
            return self._create_dummy_frames()
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return self._create_dummy_frames()
        
        indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size[::-1])
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame_tensor)
        
        cap.release()
        
        if len(frames) == 0:
            return self._create_dummy_frames()
        
        return torch.stack(frames, dim=0)
    
    def _create_dummy_frames(self) -> torch.Tensor:
        """Create dummy frames for testing."""
        # Random frames simulating endoscopy images
        frames = torch.randn(self.max_frames, 3, self.frame_size[0], self.frame_size[1])
        frames = (frames - frames.min()) / (frames.max() - frames.min())  # Normalize to [0, 1]
        return frames


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Handles variable-length frames by padding.
    """
    sample_ids = [item["sample_id"] for item in batch]
    reports = [item["report"] for item in batch]
    sections = [item["sections"] for item in batch]
    frame2section = [item["frame2section"] for item in batch]
    
    # Stack frames (assume same number of frames per batch for simplicity)
    # In practice, you might need padding
    frames_list = [item["frames"] for item in batch]
    
    # Find max frames
    max_k = max(f.shape[0] for f in frames_list)
    
    # Pad frames
    padded_frames = []
    frame_masks = []
    for f in frames_list:
        k = f.shape[0]
        if k < max_k:
            pad = torch.zeros(max_k - k, *f.shape[1:], dtype=f.dtype)
            f = torch.cat([f, pad], dim=0)
        padded_frames.append(f)
        
        mask = torch.zeros(max_k, dtype=torch.bool)
        mask[:k] = True
        frame_masks.append(mask)
    
    frames = torch.stack(padded_frames, dim=0)  # [B, K, 3, H, W]
    frame_masks = torch.stack(frame_masks, dim=0)  # [B, K]
    
    return {
        "sample_ids": sample_ids,
        "frames": frames,
        "frame_masks": frame_masks,
        "reports": reports,
        "sections": sections,
        "frame2section": frame2section,
    }


def get_dataloader(
    jsonl_path: str,
    video_dir: str,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    dataset = EndoscopyDataset(jsonl_path, video_dir, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
