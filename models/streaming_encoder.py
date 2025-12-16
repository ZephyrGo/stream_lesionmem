"""
Streaming Frame Encoder for STREAM-LesionMem.

Supports chunk-based encoding to avoid OOM when processing many frames.
"""

import torch
import torch.nn as nn
from typing import Dict, Generator, Iterator, List, Optional, Tuple, Union
import numpy as np

from .medgemma_adapter import MedGemmaAdapter


class StreamingFrameSampler:
    """
    Sample candidate frames from video for streaming processing.
    
    Supports:
    - Uniform sampling
    - Quality-based filtering (stub)
    - Scene change detection (stub)
    """
    
    def __init__(
        self,
        min_frames: int = 8,
        max_frames: int = 16,
        target_frames: int = 12,
        quality_threshold: float = 0.5,
    ):
        """
        Args:
            min_frames: Minimum frames to sample
            max_frames: Maximum frames to sample
            target_frames: Target number of frames
            quality_threshold: Quality threshold for filtering
        """
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.target_frames = target_frames
        self.quality_threshold = quality_threshold
    
    def sample(
        self,
        video_input: Union[torch.Tensor, List[str]],
        return_indices: bool = True,
    ) -> Tuple[Union[torch.Tensor, List[str]], List[int]]:
        """
        Sample frames from video input.
        
        Args:
            video_input: Either tensor [N, 3, H, W] or list of frame paths
            return_indices: Whether to return original indices
        
        Returns:
            sampled_frames: Sampled frames (tensor or paths)
            indices: Original frame indices
        """
        if isinstance(video_input, torch.Tensor):
            return self._sample_tensor(video_input)
        else:
            return self._sample_paths(video_input)
    
    def _sample_tensor(
        self,
        frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Sample from tensor frames."""
        N = frames.shape[0]
        
        # Determine number of frames to sample
        n_sample = min(self.target_frames, N)
        n_sample = max(self.min_frames, n_sample)
        n_sample = min(self.max_frames, n_sample)
        
        # Uniform sampling
        indices = np.linspace(0, N - 1, n_sample, dtype=int).tolist()
        
        # Quality filtering (stub - keep all for now)
        # TODO: Implement blur detection, exposure check, etc.
        filtered_indices = self._quality_filter(frames, indices)
        
        sampled = frames[filtered_indices]
        
        return sampled, filtered_indices
    
    def _sample_paths(
        self,
        paths: List[str],
    ) -> Tuple[List[str], List[int]]:
        """Sample from frame paths."""
        N = len(paths)
        
        n_sample = min(self.target_frames, N)
        n_sample = max(self.min_frames, n_sample)
        n_sample = min(self.max_frames, n_sample)
        
        indices = np.linspace(0, N - 1, n_sample, dtype=int).tolist()
        
        sampled_paths = [paths[i] for i in indices]
        
        return sampled_paths, indices
    
    def _quality_filter(
        self,
        frames: torch.Tensor,
        indices: List[int],
    ) -> List[int]:
        """
        Filter frames by quality score.
        
        TODO: Implement real quality metrics:
        - Blur detection (Laplacian variance)
        - Exposure check (histogram analysis)
        - Motion blur detection
        """
        # Stub: keep all frames
        return indices


class StreamingEncoder:
    """
    Streaming frame encoder using MedGemmaAdapter.
    
    Encodes frames in chunks to support streaming without loading
    all frames into memory at once.
    """
    
    def __init__(
        self,
        adapter: MedGemmaAdapter,
        chunk_size: int = 4,
        cache_selected: bool = True,
    ):
        """
        Args:
            adapter: MedGemmaAdapter instance
            chunk_size: Number of frames per encoding chunk
            cache_selected: Whether to cache tokens for selected frames
        """
        self.adapter = adapter
        self.chunk_size = chunk_size
        self.cache_selected = cache_selected
        
        # Cache for selected frame tokens
        self._token_cache: Dict[int, torch.Tensor] = {}
    
    def clear_cache(self) -> None:
        """Clear token cache."""
        self._token_cache.clear()
    
    def encode_streaming(
        self,
        frames: torch.Tensor,
        cache_indices: Optional[List[int]] = None,
    ) -> Generator[Tuple[torch.Tensor, List[int]], None, None]:
        """
        Stream-encode frames in chunks.
        
        Args:
            frames: [B, K, 3, H, W] or [K, 3, H, W] input frames
            cache_indices: Frame indices to cache (for later retrieval)
        
        Yields:
            lm_tokens: [B, chunk_size, num_patches, lm_hidden]
            frame_indices: List of frame indices in this chunk
        """
        cache_indices_set = set(cache_indices) if cache_indices else set()
        
        for lm_tokens, frame_indices in self.adapter.encode_frames_to_lm_tokens(
            frames, 
            chunk_size=self.chunk_size
        ):
            # Cache selected frames
            if self.cache_selected:
                for i, global_idx in enumerate(frame_indices):
                    if global_idx in cache_indices_set:
                        # Cache this frame's tokens
                        self._token_cache[global_idx] = lm_tokens[:, i].clone()
            
            yield lm_tokens, frame_indices
    
    def encode_all(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode all frames and return concatenated result.
        
        Args:
            frames: [B, K, 3, H, W] or [K, 3, H, W]
        
        Returns:
            all_tokens: [B, K, num_patches, lm_hidden]
        """
        all_tokens = []
        
        for lm_tokens, _ in self.encode_streaming(frames):
            all_tokens.append(lm_tokens)
        
        return torch.cat(all_tokens, dim=1)
    
    def get_cached_tokens(
        self,
        frame_indices: List[int],
    ) -> Optional[torch.Tensor]:
        """
        Get cached tokens for specified frame indices.
        
        Args:
            frame_indices: List of frame indices to retrieve
        
        Returns:
            tokens: [B, len(frame_indices), num_patches, lm_hidden] or None if not cached
        """
        if not frame_indices:
            return None
        
        tokens = []
        for idx in frame_indices:
            if idx in self._token_cache:
                tokens.append(self._token_cache[idx])
            else:
                return None  # Not all frames cached
        
        return torch.stack(tokens, dim=1)
    
    def encode_selected_frames(
        self,
        frames: torch.Tensor,
        selected_indices: List[int],
    ) -> torch.Tensor:
        """
        Encode only selected frames.
        
        First checks cache, then encodes missing frames.
        
        Args:
            frames: [B, K, 3, H, W] or [K, 3, H, W]
            selected_indices: Indices of frames to encode
        
        Returns:
            tokens: [B, len(selected_indices), num_patches, lm_hidden]
        """
        # Check cache first
        cached = self.get_cached_tokens(selected_indices)
        if cached is not None:
            return cached
        
        # Need to encode - extract selected frames
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        selected_frames = frames[:, selected_indices]  # [B, n_selected, 3, H, W]

        assert all(isinstance(x, int) for x in selected_indices), f"selected_indices not int: {selected_indices}"
        
        # Encode all selected frames
        return self.adapter.encode_frames_all(selected_frames, chunk_size=self.chunk_size)


class StreamingEncoderIterator:
    """
    Iterator wrapper for streaming encoding.
    
    Provides more control over the streaming process.
    """
    
    def __init__(
        self,
        encoder: StreamingEncoder,
        frames: torch.Tensor,
        sampler: Optional[StreamingFrameSampler] = None,
    ):
        """
        Args:
            encoder: StreamingEncoder instance
            frames: Input frames
            sampler: Optional frame sampler
        """
        self.encoder = encoder
        self.frames = frames
        self.sampler = sampler or StreamingFrameSampler()
        
        # Sample frames
        self.sampled_frames, self.frame_indices = self.sampler.sample(frames)
        
        # Create iterator
        self._iterator = None
        self._current_chunk_idx = 0
        self._total_chunks = (len(self.frame_indices) + self.encoder.chunk_size - 1) // self.encoder.chunk_size
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, List[int], int]]:
        """
        Iterate over chunks.
        
        Yields:
            lm_tokens: Encoded tokens for chunk
            frame_indices: Frame indices in chunk
            chunk_idx: Current chunk index
        """
        self._current_chunk_idx = 0
        
        for lm_tokens, indices in self.encoder.encode_streaming(self.sampled_frames):
            # Map back to original indices
            original_indices = [self.frame_indices[j] for j in indices]
            
            yield lm_tokens, original_indices, self._current_chunk_idx
            self._current_chunk_idx += 1
    
    @property
    def total_chunks(self) -> int:
        return self._total_chunks
    
    @property
    def total_frames(self) -> int:
        return len(self.frame_indices)
