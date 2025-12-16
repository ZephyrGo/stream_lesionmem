"""
QL_Endo Dataset for STREAM-LesionMem Training.

Dataset class optimized for training memory bank and router components.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from .preprocess_qlendo import SECTION_NAMES


class QLEndoDataset(Dataset):
    """
    Dataset for QL_Endo endoscopy data.
    
    Each sample contains:
    - frames: Sampled video frames [K, 3, H, W]
    - frame2section: Mapping frame_idx -> section_id
    - sections: Dict of section findings
    - abnormal_section_ids: List of abnormal section IDs
    - abnormal_labels: Per-frame abnormality labels [K]
    """
    
    def __init__(
        self,
        data_json: str,
        sample_frames: int = 12,
        min_frames: int = 8,
        max_frames: int = 16,
        image_size: Tuple[int, int] = (336, 336),
        augment: bool = False,
        normalize: bool = True,
        use_cached_frames: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            data_json: Path to preprocessed JSON file
            sample_frames: Target number of frames to sample
            min_frames: Minimum frames to use
            max_frames: Maximum frames to use
            image_size: Target image size (H, W)
            augment: Whether to apply augmentation
            normalize: Whether to normalize pixel values
            use_cached_frames: Whether to use pre-extracted frames
            cache_dir: Directory for cached frames
        """
        self.sample_frames = sample_frames
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        self.use_cached_frames = use_cached_frames
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load data
        self.samples = self._load_data(data_json)
        
        # Image normalization params (CLIP/SigLIP)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    
    def _load_data(self, data_json: str) -> List[Dict]:
        """Load preprocessed data."""
        with open(data_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = data.get("data", data)
        if isinstance(samples, dict):
            samples = [samples]
        
        print(f"Loaded {len(samples)} samples from {data_json}")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Get image folder
        folder_path = Path(sample["folder_path"])
        image_count = sample["image_count"]
        
        # Sample frame indices
        sampled_indices = self._sample_frame_indices(image_count)
        num_sampled = len(sampled_indices)
        
        # Load frames
        frames = self._load_frames(folder_path, sampled_indices)
        
        # Build frame2section mapping for sampled frames
        original_f2s = sample.get("frame2section", {})
        if isinstance(original_f2s, dict):
            # Convert string keys to int
            original_f2s = {int(k): v for k, v in original_f2s.items()}
        
        # Map sampled indices to sections
        frame2section = {}
        for new_idx, orig_idx in enumerate(sampled_indices):
            section_id = original_f2s.get(orig_idx, self._estimate_section(orig_idx, image_count))
            frame2section[new_idx] = section_id
        
        # Get abnormal section info
        abnormal_section_ids = sample.get("abnormal_section_ids", [])
        
        # Build per-frame abnormality labels
        abnormal_labels = torch.zeros(num_sampled, dtype=torch.float32)
        for frame_idx, section_id in frame2section.items():
            if section_id in abnormal_section_ids:
                abnormal_labels[frame_idx] = 1.0
        
        # Get section texts
        sections = {}
        for sec_info in sample.get("sections", []):
            sec_name = sec_info.get("section_name", "")
            sec_text = sec_info.get("text", "")
            if sec_name:
                sections[sec_name] = sec_text
        
        # Build full report from sections
        report_parts = []
        for sec_name in SECTION_NAMES:
            if sec_name in sections and sections[sec_name]:
                report_parts.append(f"**{sec_name}**: {sections[sec_name]}")
        report = "\n\n".join(report_parts) if report_parts else sample.get("full_findings", "")
        
        return {
            "exam_id": sample.get("exam_id", f"sample_{idx}"),
            "frames": frames,  # [K, 3, H, W]
            "frame2section": frame2section,  # Dict[int, int]
            "section_ids": torch.tensor([frame2section[i] for i in range(num_sampled)], dtype=torch.long),
            "abnormal_labels": abnormal_labels,  # [K]
            "abnormal_section_ids": abnormal_section_ids,
            "sections": sections,  # Dict[str, str]
            "report": report,
            "diagnosis": sample.get("diagnosis", ""),
            "sampled_indices": sampled_indices,  # Original frame indices
        }
    
    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """Sample frame indices from video."""
        # Determine number of frames to sample
        n_sample = min(self.sample_frames, total_frames)
        n_sample = max(self.min_frames, n_sample)
        n_sample = min(self.max_frames, n_sample, total_frames)
        
        if n_sample >= total_frames:
            return list(range(total_frames))
        
        # Uniform sampling with small random jitter
        if self.augment:
            # Add jitter for augmentation
            base_indices = np.linspace(0, total_frames - 1, n_sample)
            jitter = np.random.uniform(-0.5, 0.5, n_sample) * (total_frames / n_sample)
            indices = np.clip(base_indices + jitter, 0, total_frames - 1).astype(int)
            indices = sorted(set(indices))
            
            # Ensure we have enough frames
            while len(indices) < n_sample and len(indices) < total_frames:
                missing = n_sample - len(indices)
                extra = np.random.choice(
                    [i for i in range(total_frames) if i not in indices],
                    min(missing, total_frames - len(indices)),
                    replace=False
                )
                indices = sorted(set(indices) | set(extra))
        else:
            indices = np.linspace(0, total_frames - 1, n_sample, dtype=int).tolist()
        
        return indices
    
    def _load_frames(
        self,
        folder_path: Path,
        indices: List[int],
    ) -> torch.Tensor:
        """Load and preprocess frames from folder."""
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(sorted(folder_path.glob(f'*{ext}')))
            image_files.extend(sorted(folder_path.glob(f'*{ext.upper()}')))
        
        image_files = sorted(set(image_files))
        
        # If folder doesn't exist or no images, return dummy frames
        if not image_files:
            return self._create_dummy_frames(len(indices))
        
        frames = []
        for idx in indices:
            if idx < len(image_files):
                frame = self._load_single_frame(image_files[idx])
            else:
                # Use last available frame
                frame = self._load_single_frame(image_files[-1])
            frames.append(frame)
        
        return torch.stack(frames, dim=0)
    
    def _load_single_frame(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess a single frame."""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            
            # Convert to tensor [C, H, W] in [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            frame = torch.from_numpy(img_array).permute(2, 0, 1)
            
            # Apply augmentation
            if self.augment:
                frame = self._augment_frame(frame)
            
            # Normalize
            if self.normalize:
                frame = (frame - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
            
            return frame
            
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            return self._create_dummy_frames(1).squeeze(0)
    
    def _augment_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to frame."""
        # Random horizontal flip
        if random.random() > 0.5:
            frame = torch.flip(frame, dims=[2])
        
        # Random brightness/contrast (small)
        if random.random() > 0.5:
            brightness = random.uniform(0.9, 1.1)
            frame = frame * brightness
        
        if random.random() > 0.5:
            contrast = random.uniform(0.9, 1.1)
            mean = frame.mean()
            frame = (frame - mean) * contrast + mean
        
        # Clamp to valid range
        frame = torch.clamp(frame, 0, 1)
        
        return frame
    
    def _create_dummy_frames(self, num_frames: int) -> torch.Tensor:
        """Create dummy frames for testing or when images unavailable."""
        frames = torch.randn(num_frames, 3, self.image_size[0], self.image_size[1])
        frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8)
        return frames
    
    def _estimate_section(self, frame_idx: int, total_frames: int) -> int:
        """Estimate section ID from frame index."""
        # Simple linear mapping
        num_sections = len(SECTION_NAMES)
        section_id = int(frame_idx * num_sections / total_frames)
        return min(section_id, num_sections - 1)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    
    Handles variable-length frames by padding.
    """
    exam_ids = [item["exam_id"] for item in batch]
    reports = [item["report"] for item in batch]
    diagnoses = [item["diagnosis"] for item in batch]
    sections = [item["sections"] for item in batch]
    frame2section = [item["frame2section"] for item in batch]
    abnormal_section_ids = [item["abnormal_section_ids"] for item in batch]
    sampled_indices = [item["sampled_indices"] for item in batch]
    
    # Stack frames (pad to max length in batch)
    frames_list = [item["frames"] for item in batch]
    max_k = max(f.shape[0] for f in frames_list)
    
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
    
    # Pad section_ids
    section_ids_list = [item["section_ids"] for item in batch]
    padded_section_ids = []
    for s in section_ids_list:
        k = s.shape[0]
        if k < max_k:
            pad = torch.zeros(max_k - k, dtype=s.dtype)
            s = torch.cat([s, pad], dim=0)
        padded_section_ids.append(s)
    section_ids = torch.stack(padded_section_ids, dim=0)  # [B, K]
    
    # Pad abnormal_labels
    abnormal_list = [item["abnormal_labels"] for item in batch]
    padded_abnormal = []
    for a in abnormal_list:
        k = a.shape[0]
        if k < max_k:
            pad = torch.zeros(max_k - k, dtype=a.dtype)
            a = torch.cat([a, pad], dim=0)
        padded_abnormal.append(a)
    abnormal_labels = torch.stack(padded_abnormal, dim=0)  # [B, K]
    
    return {
        "exam_ids": exam_ids,
        "frames": frames,
        "frame_masks": frame_masks,
        "section_ids": section_ids,
        "abnormal_labels": abnormal_labels,
        "frame2section": frame2section,
        "abnormal_section_ids": abnormal_section_ids,
        "sections": sections,
        "reports": reports,
        "diagnoses": diagnoses,
        "sampled_indices": sampled_indices,
    }


def create_dataloaders(
    train_json: str,
    val_json: Optional[str] = None,
    batch_size: int = 2,
    sample_frames: int = 12,
    image_size: Tuple[int, int] = (336, 336),
    num_workers: int = 4,
    augment_train: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_json: Path to training data JSON
        val_json: Optional path to validation data JSON
        batch_size: Batch size
        sample_frames: Frames to sample per video
        image_size: Target image size
        num_workers: DataLoader workers
        augment_train: Whether to augment training data
    
    Returns:
        train_loader, val_loader (or None)
    """
    train_dataset = QLEndoDataset(
        data_json=train_json,
        sample_frames=sample_frames,
        image_size=image_size,
        augment=augment_train,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = None
    if val_json and Path(val_json).exists():
        val_dataset = QLEndoDataset(
            data_json=val_json,
            sample_frames=sample_frames,
            image_size=image_size,
            augment=False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    return train_loader, val_loader
