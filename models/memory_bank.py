"""
Lesion Instance Memory Bank for STREAM-LesionMem.

Core module for tracking and de-duplicating lesion instances across video frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field


@dataclass
class EvidenceEntry:
    """Single evidence entry for a lesion slot."""
    frame_idx: int
    token_idx: int
    confidence: float
    token_embed: Optional[torch.Tensor] = None  # [lm_hidden]


@dataclass
class LesionSlot:
    """Lesion instance slot with tracking state."""
    slot_id: int
    section_id: int
    embedding: torch.Tensor  # [slot_dim]
    confidence: float
    evidence: List[EvidenceEntry] = field(default_factory=list)
    update_count: int = 0
    is_active: bool = True


class DetectionOutput(NamedTuple):
    """Output from lesion detection."""
    candidates: torch.Tensor  # [B, num_patches, slot_dim]
    confidences: torch.Tensor  # [B, num_patches]
    top_indices: torch.Tensor  # [B, top_m]
    top_confidences: torch.Tensor  # [B, top_m]


class LesionDetector(nn.Module):
    """
    Detect lesion candidates from frame tokens.
    
    Maps LM tokens to slot-compatible representations with confidence scores.
    """
    
    def __init__(
        self,
        lm_hidden: int = 2560,
        slot_dim: int = 512,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        
        self.lm_hidden = lm_hidden
        self.slot_dim = slot_dim
        
        # MLP: lm_hidden -> hidden -> slot_dim + 1 (confidence)
        self.mlp = nn.Sequential(
            nn.Linear(lm_hidden, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, slot_dim + 1),
        )
        # # 强制设置权重为 bfloat16
        # for layer in self.mlp:
        #     if isinstance(layer, nn.Linear):
        #         layer.weight.data = layer.weight.data.to(dtype=torch.bfloat16)
        #         if layer.bias is not None:
        #             layer.bias.data = layer.bias.data.to(dtype=torch.bfloat16)
    
    def forward(self, lm_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect lesion candidates.
        
        Args:
            lm_tokens: [B, num_patches, lm_hidden]
        
        Returns:
            candidates: [B, num_patches, slot_dim]
            confidences: [B, num_patches] in [0, 1]
        """
        # Assert input shape
        assert lm_tokens.dim() == 3, f"Expected 3D input, got {lm_tokens.dim()}D"
        
        out = self.mlp(lm_tokens)  # [B, num_patches, slot_dim + 1]
        
        candidates = out[..., :-1]  # [B, num_patches, slot_dim]
        confidences = torch.sigmoid(out[..., -1])  # [B, num_patches]
        
        return candidates, confidences


class SlotMatcher(nn.Module):
    """
    Match candidate embeddings to existing slots.
    
    Uses cosine similarity with optional learned refinement.
    """
    
    def __init__(
        self,
        slot_dim: int = 512,
        use_learned: bool = False,
    ):
        super().__init__()
        
        self.slot_dim = slot_dim
        self.use_learned = use_learned
        
        if use_learned:
            # Optional learned similarity
            self.sim_mlp = nn.Sequential(
                nn.Linear(slot_dim * 2, slot_dim),
                nn.ReLU(),
                nn.Linear(slot_dim, 1),
            )
    
    def forward(
        self,
        candidate: torch.Tensor,
        slot_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Match candidate to slots.
        
        Args:
            candidate: [slot_dim]
            slot_embeds: [num_slots, slot_dim]
        
        Returns:
            scores: [num_slots] similarity scores
            best_idx: Index of best matching slot
        """
        if slot_embeds.shape[0] == 0:
            return torch.tensor([]), -1
        
        # Cosine similarity
        candidate_norm = F.normalize(candidate.unsqueeze(0), dim=-1)
        slots_norm = F.normalize(slot_embeds, dim=-1)
        scores = (candidate_norm @ slots_norm.T).squeeze(0)  # [num_slots]
        
        if self.use_learned:
            # Refine with learned similarity
            candidate_exp = candidate.unsqueeze(0).expand(slot_embeds.shape[0], -1)
            concat = torch.cat([candidate_exp, slot_embeds], dim=-1)
            learned_scores = self.sim_mlp(concat).squeeze(-1)
            scores = (scores + torch.sigmoid(learned_scores)) / 2
        
        best_idx = scores.argmax().item()
        
        return scores, best_idx


class SlotUpdater(nn.Module):
    """
    Update slot embeddings with new observations.
    
    Uses gated update mechanism.
    """
    
    def __init__(
        self,
        slot_dim: int = 512,
    ):
        super().__init__()
        
        self.slot_dim = slot_dim
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        old_embed: torch.Tensor,
        new_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update slot embedding.
        
        Args:
            old_embed: [slot_dim] current slot embedding
            new_embed: [slot_dim] new observation embedding
        
        Returns:
            updated: [slot_dim] updated slot embedding
        """
        concat = torch.cat([old_embed, new_embed], dim=-1)
        gate = self.gate(concat)
        
        updated = gate * new_embed + (1 - gate) * old_embed
        
        return updated


class LesionInstanceMemoryBank(nn.Module):
    """
    Memory bank for tracking lesion instances across video frames.
    
    Key features:
    - Detect lesion candidates from frame tokens
    - Match candidates to existing slots (with section constraints)
    - Update slots with gated mechanism
    - Cache evidence (frame_idx, token_idx, confidence, token_embed)
    - Select top evidence frames for final generation
    """
    
    def __init__(
        self,
        num_slots: int = 16,
        slot_dim: int = 512,
        lm_hidden: int = 2560,
        similarity_threshold: float = 0.7,
        top_m: int = 5,
        evidence_topL: int = 4,
        max_evidence_tokens: int = 32,
    ):
        """
        Args:
            num_slots: Maximum number of lesion slots
            slot_dim: Dimension of slot embeddings
            lm_hidden: LM hidden dimension (for projection)
            similarity_threshold: Threshold for slot matching
            top_m: Top candidates per frame to process
            evidence_topL: Max evidence entries per slot
            max_evidence_tokens: Max evidence tokens to include in output
        """
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.lm_hidden = lm_hidden
        self.similarity_threshold = similarity_threshold
        self.top_m = top_m
        self.evidence_topL = evidence_topL
        self.max_evidence_tokens = max_evidence_tokens
        
        # Components
        self.detector = LesionDetector(lm_hidden, slot_dim)
        self.matcher = SlotMatcher(slot_dim)
        self.updater = SlotUpdater(slot_dim)
        
        # Projection from slot_dim back to lm_hidden for output
        self.slot_to_lm_proj = nn.Linear(slot_dim, lm_hidden)
        
        # Slot storage (dynamically allocated)
        self.slots: List[LesionSlot] = []
        
        # Device tracking
        self._device = torch.device("cpu")
        # device = next(self.parameters()).device
        # dtype  = next(self.parameters()).dtype
        # embedding = torch.zeros(self.slot_dim, device=device, dtype=dtype)

    
    def reset(self) -> None:
        """Reset memory bank state."""
        self.slots = []
    
    # @property
    # def device(self) -> torch.device:
    #     return self._device
    
    # def to(self, device: torch.device) -> "LesionInstanceMemoryBank":
    #     """Move to device."""
    #     self._device = device
    #     return super().to(device)
    
    def detect(
        self,
        lm_tokens_frame: torch.Tensor,
    ) -> DetectionOutput:
        """
        Detect lesion candidates from a single frame's tokens.
        
        Args:
            lm_tokens_frame: [B, num_patches, lm_hidden]
        
        Returns:
            DetectionOutput with candidates, confidences, top indices/confidences
        """
        ref = next(self.detector.parameters())
        lm_tokens_frame = lm_tokens_frame.to(device=ref.device, dtype=ref.dtype)

        B, num_patches, _ = lm_tokens_frame.shape
        
        # Detect candidates
        candidates, confidences = self.detector(lm_tokens_frame)
        
        # Select top_m candidates per batch
        top_confidences, top_indices = torch.topk(
            confidences, 
            min(self.top_m, num_patches), 
            dim=-1
        )
        
        return DetectionOutput(
            candidates=candidates,
            confidences=confidences,
            top_indices=top_indices,
            top_confidences=top_confidences,
        )
    
    def match_to_slot(
        self,
        candidate: torch.Tensor,
        section_id: int,
        match_same_section_only: bool = True,
    ) -> Tuple[float, int]:
        """
        Match candidate to existing slots.
        
        Args:
            candidate: [slot_dim] candidate embedding
            section_id: Section ID for matching constraint
            match_same_section_only: Only match within same section
        
        Returns:
            best_score: Best match score
            best_slot_idx: Index of best matching slot (-1 if no match)
        """
        if len(self.slots) == 0:
            return 0.0, -1
        
        # Filter slots by section
        valid_slots = []
        valid_indices = []
        
        for i, slot in enumerate(self.slots):
            if not slot.is_active:
                continue
            if match_same_section_only and slot.section_id != section_id:
                continue
            valid_slots.append(slot)
            valid_indices.append(i)
        
        if len(valid_slots) == 0:
            return 0.0, -1
        
        # Get slot embeddings
        slot_embeds = torch.stack([s.embedding for s in valid_slots])

        # Use a parameterized module as reference (matcher may have 0 params)
        ref = next(self.detector.parameters())  # safe: detector has Linear
        device, dtype = ref.device, ref.dtype

        candidate = candidate.to(device=device, dtype=dtype)
        slot_embeds = slot_embeds.to(device=device, dtype=dtype)

        scores, local_best = self.matcher(candidate, slot_embeds)
        
        if local_best < 0:
            return 0.0, -1
        
        best_score = scores[local_best].item()
        best_slot_idx = valid_indices[local_best]
        
        return best_score, best_slot_idx
    
    def update_slot(
        self,
        slot_idx: int,
        new_embed: torch.Tensor,
        frame_idx: int,
        token_idx: int,
        confidence: float,
        token_embed: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update existing slot with new observation.
        
        Args:
            slot_idx: Slot index to update
            new_embed: [slot_dim] new observation embedding
            frame_idx: Frame index of observation
            token_idx: Token index within frame
            confidence: Detection confidence
            token_embed: [lm_hidden] original token embedding (for evidence)
        """
        ref = next(self.detector.parameters())
        device, dtype = ref.device, ref.dtype

        slot = self.slots[slot_idx]
        
        # Update embedding
        slot.embedding = slot.embedding.to(device=device, dtype=dtype)
        new_embed      = new_embed.to(device=device, dtype=dtype)
        slot.embedding = self.updater(slot.embedding, new_embed)

        slot.update_count += 1
        slot.confidence = max(slot.confidence, confidence)
        
        # Add evidence
        evidence = EvidenceEntry(
            frame_idx=frame_idx,
            token_idx=token_idx,
            confidence=confidence,
            token_embed=token_embed.detach() if token_embed is not None else None,
        )
        slot.evidence.append(evidence)
        
        # Keep only top-L evidence by confidence
        if len(slot.evidence) > self.evidence_topL:
            slot.evidence.sort(key=lambda e: e.confidence, reverse=True)
            slot.evidence = slot.evidence[:self.evidence_topL]
    
    def create_slot(
        self,
        embedding: torch.Tensor,
        section_id: int,
        frame_idx: int,
        token_idx: int,
        confidence: float,
        token_embed: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Create new slot.
        
        Args:
            embedding: [slot_dim] initial slot embedding
            section_id: Section ID
            frame_idx: Frame index
            token_idx: Token index
            confidence: Detection confidence
            token_embed: [lm_hidden] token embedding for evidence
        
        Returns:
            slot_idx: Index of new slot
        """
        if len(self.slots) >= self.num_slots:
            # Find least confident slot to replace
            min_conf_idx = min(range(len(self.slots)), key=lambda i: self.slots[i].confidence)
            if self.slots[min_conf_idx].confidence < confidence:
                slot_idx = min_conf_idx
            else:
                return -1  # Cannot create slot
        else:
            slot_idx = len(self.slots)
            self.slots.append(None)  # Placeholder
        
        evidence = EvidenceEntry(
            frame_idx=frame_idx,
            token_idx=token_idx,
            confidence=confidence,
            token_embed=token_embed.detach() if token_embed is not None else None,
        )
        
        self.slots[slot_idx] = LesionSlot(
            slot_id=slot_idx,
            section_id=section_id,
            embedding=embedding.detach(),
            confidence=confidence,
            evidence=[evidence],
            update_count=1,
            is_active=True,
        )
        
        return slot_idx
    
    def process_frame(
        self,
        lm_tokens_frame: torch.Tensor,
        frame_idx: int,
        section_id: int = 0,
        original_tokens: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Process a single frame and update memory.
        
        Args:
            lm_tokens_frame: [B, num_patches, lm_hidden] (assume B=1 for now)
            frame_idx: Global frame index
            section_id: Section ID for this frame
            original_tokens: Original lm tokens for evidence caching
        
        Returns:
            Dict with processing stats
        """
        assert lm_tokens_frame.shape[0] == 1, "Batch size must be 1 for process_frame"
        
        # Detect candidates
        detection = self.detect(lm_tokens_frame)
        
        stats = {
            "frame_idx": frame_idx,
            "section_id": section_id,
            "num_candidates": detection.top_indices.shape[1],
            "matched": 0,
            "created": 0,
            "skipped": 0,
        }
        
        # Process top candidates
        for i in range(detection.top_indices.shape[1]):
            token_idx = detection.top_indices[0, i].item()
            confidence = detection.top_confidences[0, i].item()
            candidate = detection.candidates[0, token_idx]  # [slot_dim]
            
            # Skip low confidence
            if confidence < 0.3:
                stats["skipped"] += 1
                continue
            
            # Get original token for evidence
            token_embed = None
            if original_tokens is not None:
                token_embed = original_tokens[0, token_idx]  # [lm_hidden]
            
            # Try to match to existing slot
            best_score, best_slot_idx = self.match_to_slot(candidate, section_id)
            
            if best_score >= self.similarity_threshold and best_slot_idx >= 0:
                # Update existing slot
                self.update_slot(
                    best_slot_idx,
                    candidate,
                    frame_idx,
                    token_idx,
                    confidence,
                    token_embed,
                )
                stats["matched"] += 1
            else:
                # Create new slot
                new_idx = self.create_slot(
                    candidate,
                    section_id,
                    frame_idx,
                    token_idx,
                    confidence,
                    token_embed,
                )
                if new_idx >= 0:
                    stats["created"] += 1
                else:
                    stats["skipped"] += 1
        
        return stats
    
    def get_slot_summary(self) -> List[Dict]:
        """
        Get summary of all active slots.
        
        Returns:
            List of slot summaries
        """
        summaries = []
        
        for slot in self.slots:
            if not slot.is_active:
                continue
            
            evidence_frames = list(set(e.frame_idx for e in slot.evidence))
            
            summaries.append({
                "slot_id": slot.slot_id,
                "section_id": slot.section_id,
                "confidence": slot.confidence,
                "update_count": slot.update_count,
                "evidence_frames": evidence_frames,
                "num_evidence": len(slot.evidence),
            })
        
        return summaries
    
    def select_evidence_frames(
        self,
        max_frames: int = 4,
    ) -> List[int]:
        """
        Select top evidence frames from all slots.
        
        Aggregates evidence across slots and selects frames with
        highest cumulative confidence.
        
        Args:
            max_frames: Maximum frames to select
        
        Returns:
            List of selected frame indices (sorted by importance)
        """
        # Aggregate confidence per frame
        frame_scores: Dict[int, float] = {}
        
        for slot in self.slots:
            if not slot.is_active:
                continue
            
            for evidence in slot.evidence:
                frame_idx = evidence.frame_idx
                if frame_idx not in frame_scores:
                    frame_scores[frame_idx] = 0.0
                frame_scores[frame_idx] += evidence.confidence
        
        if not frame_scores:
            return []
        
        # Sort by score and take top
        sorted_frames = sorted(frame_scores.items(), key=lambda x: -x[1])
        selected = [frame_idx for frame_idx, _ in sorted_frames[:max_frames]]
        
        # Sort by frame index for temporal order
        selected.sort()
        
        return selected
    
    def get_memory_image_embeds(
        self,
        selected_frame_tokens: Optional[torch.Tensor] = None,
        max_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get combined memory embeddings for injection into LM.
        
        Combines:
        1. Slot embeddings (projected to lm_hidden)
        2. Cached evidence token embeddings
        
        Args:
            selected_frame_tokens: [B, n_frames, num_patches, lm_hidden] 
                                   Selected frame tokens (optional, for combining)
            max_tokens: Maximum tokens to return
        
        Returns:
            memory_embeds: [B, T_mem, lm_hidden] memory tokens
        """
        max_tokens = max_tokens or self.max_evidence_tokens
        
        embeds = []
        
        # Add slot embeddings (projected)
        for slot in self.slots:
            if not slot.is_active:
                continue
            
            slot_embed = self.slot_to_lm_proj(slot.embedding)  # [lm_hidden]
            embeds.append(slot_embed.unsqueeze(0))
        
        # Add cached evidence token embeddings
        for slot in self.slots:
            if not slot.is_active:
                continue
            
            for evidence in slot.evidence:
                if evidence.token_embed is not None:
                    embeds.append(evidence.token_embed.unsqueeze(0))
        
        if not embeds:
            # Return dummy embedding
            ref = next(self.detector.parameters())
            return torch.zeros(1, 1, self.lm_hidden, device=ref.device, dtype=ref.dtype)

        
        # Stack and truncate
        all_embeds = torch.cat(embeds, dim=0)  # [T_total, lm_hidden]
        
        if all_embeds.shape[0] > max_tokens:
            all_embeds = all_embeds[:max_tokens]
        
        # Add batch dimension
        return all_embeds.unsqueeze(0)  # [1, T_mem, lm_hidden]
    
    def get_combined_image_embeds(
        self,
        selected_frame_tokens: torch.Tensor,
        include_memory: bool = True,
        max_tokens_per_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Combine selected frame tokens with memory embeddings.
        
        Args:
            selected_frame_tokens: [B, n_frames, num_patches, lm_hidden]
            include_memory: Whether to include memory embeddings
            max_tokens_per_frame: Max tokens per frame (for pruning)
        
        Returns:
            combined: [B, T_img, lm_hidden] combined embeddings for injection
        """
        B, n_frames, num_patches, lm_hidden = selected_frame_tokens.shape
        
        # Flatten frame tokens
        frame_embeds = selected_frame_tokens.reshape(B, n_frames * num_patches, lm_hidden)
        
        # Optionally prune tokens per frame
        if max_tokens_per_frame and max_tokens_per_frame < num_patches:
            # Keep top tokens per frame based on some criterion
            # For now, keep first max_tokens_per_frame
            pruned = []
            for i in range(n_frames):
                start = i * num_patches
                pruned.append(frame_embeds[:, start:start+max_tokens_per_frame])
            frame_embeds = torch.cat(pruned, dim=1)
        
        if include_memory:
            memory_embeds = self.get_memory_image_embeds()
            memory_embeds = memory_embeds.to(frame_embeds.device).expand(B, -1, -1)
            combined = torch.cat([frame_embeds, memory_embeds], dim=1)
        else:
            combined = frame_embeds
        
        return combined
