"""
Section Router for STREAM-LesionMem.

Routes frames to anatomical sections and detects abnormality scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class RouterMode(Enum):
    RULE = "rule"
    LEARNED = "learned"


class SectionRouter(nn.Module):
    """
    Route frames to sections and compute abnormality scores.
    
    Supports:
    - Rule-based mode: Uses provided frame2section mapping
    - Learned mode: Predicts section and abnormality from pooled tokens
    """
    
    DEFAULT_SECTION_NAMES = [
        "esophagus",
        "gastroesophageal_junction", 
        "cardia",
        "fundus",
        "body",
        "antrum",
        "pylorus",
        "duodenum",
    ]
    
    def __init__(
        self,
        mode: str = "rule",
        num_sections: int = 8,
        lm_hidden: int = 2560,
        abnormal_threshold: float = 0.5,
        section_names: Optional[List[str]] = None,
    ):
        """
        Args:
            mode: "rule" or "learned"
            num_sections: Number of sections
            lm_hidden: LM hidden dimension for learned mode
            abnormal_threshold: Threshold for abnormality classification
            section_names: Names of sections
        """
        super().__init__()
        
        self.mode = RouterMode(mode)
        self.num_sections = num_sections
        self.lm_hidden = lm_hidden
        self.abnormal_threshold = abnormal_threshold
        self.section_names = section_names or self.DEFAULT_SECTION_NAMES[:num_sections]
        
        # Learned router components
        if self.mode == RouterMode.LEARNED:
            # Pooling + classification
            self.pool_proj = nn.Linear(lm_hidden, lm_hidden // 2)
            self.section_head = nn.Linear(lm_hidden // 2, num_sections)
            self.abnormal_head = nn.Linear(lm_hidden // 2, 1)
    
    def forward(
        self,
        lm_tokens: torch.Tensor,
        frame2section: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route frames to sections and compute abnormality.
        
        Args:
            lm_tokens: [B, K, num_patches, lm_hidden] or [B, num_patches, lm_hidden]
            frame2section: Optional per-frame section labels (for rule mode)
        
        Returns:
            section_ids: [B, K] or [B] section IDs
            abnormal_scores: [B, K] or [B] abnormality scores in [0, 1]
        """
        # Handle both batched frames and single frame
        if lm_tokens.dim() == 3:
            # Single frame: [B, num_patches, lm_hidden]
            return self._forward_single(lm_tokens, frame2section)
        elif lm_tokens.dim() == 4:
            # Multiple frames: [B, K, num_patches, lm_hidden]
            return self._forward_batch(lm_tokens, frame2section)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {lm_tokens.dim()}D")
    
    def _forward_single(
        self,
        lm_tokens: torch.Tensor,
        frame2section: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route single frame."""
        B = lm_tokens.shape[0]
        device = lm_tokens.device
        
        if self.mode == RouterMode.RULE:
            # Use provided mapping or default
            if frame2section is not None:
                section_ids = torch.tensor(frame2section, device=device, dtype=torch.long)
                if section_ids.dim() == 0:
                    section_ids = section_ids.unsqueeze(0)
                section_ids = section_ids[:B]
            else:
                section_ids = torch.zeros(B, device=device, dtype=torch.long)
            
            # Dummy abnormal scores for rule mode
            abnormal_scores = torch.zeros(B, device=device)
            
        else:  # LEARNED
            section_ids, abnormal_scores = self._predict(lm_tokens)
        
        return section_ids, abnormal_scores
    
    def _forward_batch(
        self,
        lm_tokens: torch.Tensor,
        frame2section: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route batch of frames."""
        B, K, num_patches, lm_hidden = lm_tokens.shape
        device = lm_tokens.device
        
        if self.mode == RouterMode.RULE:
            if frame2section is not None:
                # frame2section should be [K] or [B, K]
                if isinstance(frame2section, list):
                    frame2section = torch.tensor(frame2section, device=device, dtype=torch.long)
                
                if frame2section.dim() == 1:
                    section_ids = frame2section.unsqueeze(0).expand(B, -1)
                else:
                    section_ids = frame2section[:B, :K]
            else:
                # Default: evenly distribute across sections
                section_ids = torch.tensor(
                    [i * self.num_sections // K for i in range(K)],
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)
            
            abnormal_scores = torch.zeros(B, K, device=device)
            
        else:  # LEARNED
            # Process each frame
            section_ids_list = []
            abnormal_scores_list = []
            
            for k in range(K):
                frame_tokens = lm_tokens[:, k]  # [B, num_patches, lm_hidden]
                sec_id, abnorm = self._predict(frame_tokens)
                section_ids_list.append(sec_id)
                abnormal_scores_list.append(abnorm)
            
            section_ids = torch.stack(section_ids_list, dim=1)  # [B, K]
            abnormal_scores = torch.stack(abnormal_scores_list, dim=1)  # [B, K]
        
        return section_ids, abnormal_scores
    
    def _predict(
        self,
        lm_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict section and abnormality for single frame.
        
        Args:
            lm_tokens: [B, num_patches, lm_hidden]
        
        Returns:
            section_ids: [B] predicted section IDs
            abnormal_scores: [B] abnormality scores
        """
        # Mean pooling
        pooled = lm_tokens.mean(dim=1)  # [B, lm_hidden]
        
        # Project
        features = F.relu(self.pool_proj(pooled))  # [B, lm_hidden // 2]
        
        # Section prediction
        section_logits = self.section_head(features)  # [B, num_sections]
        section_ids = section_logits.argmax(dim=-1)  # [B]
        
        # Abnormality prediction
        abnormal_logits = self.abnormal_head(features).squeeze(-1)  # [B]
        abnormal_scores = torch.sigmoid(abnormal_logits)
        
        return section_ids, abnormal_scores
    
    def get_section_name(self, section_id: int) -> str:
        """Get section name by ID."""
        if 0 <= section_id < len(self.section_names):
            return self.section_names[section_id]
        return f"section_{section_id}"
    
    def get_abnormal_sections(
        self,
        section_ids: torch.Tensor,
        abnormal_scores: torch.Tensor,
    ) -> List[int]:
        """
        Get list of sections with abnormality above threshold.
        
        Args:
            section_ids: [K] section IDs per frame
            abnormal_scores: [K] abnormality scores
        
        Returns:
            List of abnormal section IDs
        """
        # Group scores by section
        section_max_score: Dict[int, float] = {}
        
        for sec_id, score in zip(section_ids.tolist(), abnormal_scores.tolist()):
            if sec_id not in section_max_score:
                section_max_score[sec_id] = 0.0
            section_max_score[sec_id] = max(section_max_score[sec_id], score)
        
        # Filter by threshold
        abnormal_sections = [
            sec_id for sec_id, score in section_max_score.items()
            if score >= self.abnormal_threshold
        ]
        
        return abnormal_sections
    
    def compute_loss(
        self,
        lm_tokens: torch.Tensor,
        target_sections: torch.Tensor,
        target_abnormal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for learned router.
        
        Args:
            lm_tokens: [B, K, num_patches, lm_hidden]
            target_sections: [B, K] ground truth section IDs
            target_abnormal: [B, K] ground truth abnormality labels
        
        Returns:
            Dict with 'section_loss', 'abnormal_loss', 'total_loss'
        """
        if self.mode != RouterMode.LEARNED:
            return {"total_loss": torch.tensor(0.0, device=lm_tokens.device)}
        
        B, K = target_sections.shape
        
        section_losses = []
        abnormal_losses = []
        
        for k in range(K):
            frame_tokens = lm_tokens[:, k]  # [B, num_patches, lm_hidden]
            
            # Mean pooling
            pooled = frame_tokens.mean(dim=1)
            features = F.relu(self.pool_proj(pooled))
            
            # Section loss
            section_logits = self.section_head(features)
            sec_loss = F.cross_entropy(section_logits, target_sections[:, k])
            section_losses.append(sec_loss)
            
            # Abnormal loss
            abnormal_logits = self.abnormal_head(features).squeeze(-1)
            abn_loss = F.binary_cross_entropy_with_logits(
                abnormal_logits,
                target_abnormal[:, k].float(),
            )
            abnormal_losses.append(abn_loss)
        
        section_loss = torch.stack(section_losses).mean()
        abnormal_loss = torch.stack(abnormal_losses).mean()
        total_loss = section_loss + abnormal_loss
        
        return {
            "section_loss": section_loss,
            "abnormal_loss": abnormal_loss,
            "total_loss": total_loss,
        }
