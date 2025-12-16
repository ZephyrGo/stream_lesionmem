"""
Training Loss Functions for STREAM-LesionMem.

Implements:
- Detection loss: Train lesion detector to identify abnormal tokens
- Router loss: Train section classification and abnormality detection
- Matching loss: Train slot matcher for cross-frame consistency
- Generation loss: (Optional) LM generation loss for abnormal sections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any


class DetectionLoss(nn.Module):
    """
    Loss for training LesionDetector.
    
    Uses frame-level abnormality labels as weak supervision.
    Frames from abnormal sections should have higher detection scores.
    
    NOTE: Uses logits (before sigmoid) for AMP compatibility.
    """
    
    def __init__(
        self,
        pos_weight: float = 2.0,
        temperature: float = 1.0,
        use_focal: bool = True,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            pos_weight: Weight for positive (abnormal) samples
            temperature: Temperature for confidence scores
            use_focal: Whether to use focal loss
            focal_gamma: Gamma for focal loss
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.temperature = temperature
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
    
    def forward(
        self,
        confidences: torch.Tensor,
        abnormal_labels: torch.Tensor,
        frame_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute detection loss.
        
        Args:
            confidences: [B, K, num_patches] detection confidence scores (already sigmoid)
            abnormal_labels: [B, K] frame-level abnormality labels (0 or 1)
            frame_masks: [B, K] valid frame mask
        
        Returns:
            loss: Scalar loss value
        """
        B, K, num_patches = confidences.shape
        
        # Pool confidences to frame level (max pooling)
        frame_conf, _ = confidences.max(dim=-1)  # [B, K]
        
        # Apply temperature
        frame_conf = frame_conf / self.temperature
        
        # Convert sigmoid output back to logits for numerical stability with AMP
        # logit = log(p / (1-p))
        eps = 1e-7
        frame_conf_clamped = frame_conf.clamp(eps, 1 - eps)
        frame_logits = torch.log(frame_conf_clamped / (1 - frame_conf_clamped))
        
        # Ensure labels have correct dtype
        abnormal_labels = abnormal_labels.float()
        
        # Compute loss using logits (AMP-safe)
        if self.use_focal:
            loss = self._focal_loss_with_logits(frame_logits, abnormal_labels)
        else:
            # Weighted BCE with logits
            pos_weight_tensor = torch.tensor([self.pos_weight], device=frame_logits.device, dtype=frame_logits.dtype)
            loss = F.binary_cross_entropy_with_logits(
                frame_logits,
                abnormal_labels,
                pos_weight=pos_weight_tensor.expand_as(abnormal_labels),
                reduction='none',
            )
        
        # Mask invalid frames
        if frame_masks is not None:
            loss = loss * frame_masks.float()
            loss = loss.sum() / frame_masks.float().sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss
    
    def _focal_loss_with_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss using logits (AMP-safe)."""
        # Get probabilities
        p = torch.sigmoid(logits)
        
        # Binary focal loss
        pt = torch.where(target > 0.5, p, 1 - p)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # BCE with logits (AMP-safe)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        
        # Apply pos_weight manually for positive samples
        weight = torch.where(
            target > 0.5,
            torch.full_like(target, self.pos_weight),
            torch.ones_like(target),
        )
        
        return focal_weight * weight * bce


class RouterLoss(nn.Module):
    """
    Loss for training SectionRouter.
    
    Combines:
    - Section classification loss (cross-entropy)
    - Abnormality detection loss (binary cross-entropy with logits)
    """
    
    def __init__(
        self,
        section_weight: float = 1.0,
        abnormal_weight: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        """
        Args:
            section_weight: Weight for section classification loss
            abnormal_weight: Weight for abnormality detection loss
            label_smoothing: Label smoothing for section classification
        """
        super().__init__()
        self.section_weight = section_weight
        self.abnormal_weight = abnormal_weight
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        section_logits: torch.Tensor,
        abnormal_scores: torch.Tensor,
        section_labels: torch.Tensor,
        abnormal_labels: torch.Tensor,
        frame_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute router loss.
        
        Args:
            section_logits: [B, K, num_sections] section predictions
            abnormal_scores: [B, K] abnormality scores (after sigmoid)
            section_labels: [B, K] ground truth section IDs
            abnormal_labels: [B, K] ground truth abnormality labels
            frame_masks: [B, K] valid frame mask
        
        Returns:
            Dict with 'section_loss', 'abnormal_loss', 'total_loss'
        """
        B, K = section_labels.shape
        
        # Section classification loss
        section_logits_flat = section_logits.reshape(B * K, -1)
        section_labels_flat = section_labels.reshape(B * K)
        
        section_loss = F.cross_entropy(
            section_logits_flat,
            section_labels_flat,
            label_smoothing=self.label_smoothing,
            reduction='none',
        ).reshape(B, K)
        
        # Ensure labels have correct dtype
        abnormal_labels = abnormal_labels.float()
        
        # Abnormality detection loss - convert scores back to logits for AMP safety
        eps = 1e-7
        abnormal_scores_clamped = abnormal_scores.clamp(eps, 1 - eps)
        abnormal_logits = torch.log(abnormal_scores_clamped / (1 - abnormal_scores_clamped))
        
        abnormal_loss = F.binary_cross_entropy_with_logits(
            abnormal_logits,
            abnormal_labels,
            reduction='none',
        )
        
        # Mask invalid frames
        if frame_masks is not None:
            section_loss = section_loss * frame_masks.float()
            abnormal_loss = abnormal_loss * frame_masks.float()
            
            n_valid = frame_masks.float().sum().clamp(min=1)
            section_loss = section_loss.sum() / n_valid
            abnormal_loss = abnormal_loss.sum() / n_valid
        else:
            section_loss = section_loss.mean()
            abnormal_loss = abnormal_loss.mean()
        
        total_loss = self.section_weight * section_loss + self.abnormal_weight * abnormal_loss
        
        return {
            "section_loss": section_loss,
            "abnormal_loss": abnormal_loss,
            "total_loss": total_loss,
        }


class MatchingLoss(nn.Module):
    """
    Loss for training slot matching consistency.
    
    Encourages:
    - Same lesion across frames -> similar slot embeddings
    - Different lesions -> different slot embeddings
    
    Uses contrastive learning approach.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
    ):
        """
        Args:
            temperature: Temperature for contrastive loss
            margin: Margin for negative pairs
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        slot_embeddings: torch.Tensor,
        slot_section_ids: torch.Tensor,
        slot_confidences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matching consistency loss.
        
        Args:
            slot_embeddings: [num_slots, slot_dim] slot embeddings
            slot_section_ids: [num_slots] section ID for each slot
            slot_confidences: [num_slots] confidence for each slot
        
        Returns:
            loss: Scalar loss value
        """
        num_slots = slot_embeddings.shape[0]
        
        if num_slots < 2:
            return torch.tensor(0.0, device=slot_embeddings.device, dtype=slot_embeddings.dtype)
        
        # Normalize embeddings
        embeddings_norm = F.normalize(slot_embeddings, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = embeddings_norm @ embeddings_norm.T  # [num_slots, num_slots]
        sim_matrix = sim_matrix / self.temperature
        
        # Build positive/negative masks based on section IDs
        # Slots in same section should be similar (positive)
        section_match = slot_section_ids.unsqueeze(0) == slot_section_ids.unsqueeze(1)
        
        # Mask diagonal
        eye_mask = torch.eye(num_slots, device=slot_embeddings.device, dtype=torch.bool)
        section_match = section_match & ~eye_mask
        
        # Contrastive loss
        # Positive pairs: same section, minimize distance
        # Negative pairs: different section, maximize distance
        
        pos_mask = section_match.float()
        neg_mask = (~section_match & ~eye_mask).float()
        
        if pos_mask.sum() > 0:
            pos_loss = -(sim_matrix * pos_mask).sum() / pos_mask.sum().clamp(min=1)
        else:
            pos_loss = torch.tensor(0.0, device=slot_embeddings.device, dtype=slot_embeddings.dtype)
        
        if neg_mask.sum() > 0:
            neg_loss = F.relu(sim_matrix - self.margin) * neg_mask
            neg_loss = neg_loss.sum() / neg_mask.sum().clamp(min=1)
        else:
            neg_loss = torch.tensor(0.0, device=slot_embeddings.device, dtype=slot_embeddings.dtype)
        
        return pos_loss + neg_loss


class MemoryBankLoss(nn.Module):
    """
    Combined loss for training Memory Bank components.
    
    Combines:
    - Detection loss
    - Router loss  
    - Matching loss
    """
    
    def __init__(
        self,
        detection_weight: float = 1.0,
        router_weight: float = 1.0,
        matching_weight: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            detection_weight: Weight for detection loss
            router_weight: Weight for router loss
            matching_weight: Weight for matching loss
        """
        super().__init__()
        
        self.detection_weight = detection_weight
        self.router_weight = router_weight
        self.matching_weight = matching_weight
        
        self.detection_loss = DetectionLoss(
            pos_weight=kwargs.get("det_pos_weight", 2.0),
            use_focal=kwargs.get("det_use_focal", True),
        )
        
        self.router_loss = RouterLoss(
            section_weight=kwargs.get("router_section_weight", 1.0),
            abnormal_weight=kwargs.get("router_abnormal_weight", 1.0),
        )
        
        self.matching_loss = MatchingLoss(
            temperature=kwargs.get("match_temperature", 0.1),
        )
    
    def forward(
        self,
        # Detection outputs
        detection_confidences: torch.Tensor,
        # Router outputs
        section_logits: torch.Tensor,
        abnormal_scores: torch.Tensor,
        # Labels
        section_labels: torch.Tensor,
        abnormal_labels: torch.Tensor,
        # Masks
        frame_masks: Optional[torch.Tensor] = None,
        # Optional: slot info for matching loss
        slot_embeddings: Optional[torch.Tensor] = None,
        slot_section_ids: Optional[torch.Tensor] = None,
        slot_confidences: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Returns:
            Dict with individual losses and total loss
        """
        device = detection_confidences.device
        dtype = detection_confidences.dtype
        
        losses = {}
        
        # Detection loss
        det_loss = self.detection_loss(
            detection_confidences,
            abnormal_labels,
            frame_masks,
        )
        losses["detection_loss"] = det_loss
        
        # Router loss
        router_losses = self.router_loss(
            section_logits,
            abnormal_scores,
            section_labels,
            abnormal_labels,
            frame_masks,
        )
        losses["router_section_loss"] = router_losses["section_loss"]
        losses["router_abnormal_loss"] = router_losses["abnormal_loss"]
        losses["router_loss"] = router_losses["total_loss"]
        
        # Matching loss (if slot info provided)
        if slot_embeddings is not None and slot_embeddings.shape[0] > 1:
            match_loss = self.matching_loss(
                slot_embeddings,
                slot_section_ids,
                slot_confidences,
            )
            losses["matching_loss"] = match_loss
        else:
            losses["matching_loss"] = torch.tensor(0.0, device=device, dtype=dtype)
        
        # Total loss
        total_loss = (
            self.detection_weight * det_loss +
            self.router_weight * router_losses["total_loss"] +
            self.matching_weight * losses["matching_loss"]
        )
        losses["total_loss"] = total_loss
        
        return losses


class GenerationLoss(nn.Module):
    """
    Loss for LM generation (optional, for end-to-end training).
    
    Only computes loss on abnormal section tokens.
    """
    
    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        abnormal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute generation loss.
        
        Args:
            logits: [B, seq_len, vocab_size] model logits
            labels: [B, seq_len] target token IDs
            abnormal_mask: [B, seq_len] mask for abnormal section tokens
        
        Returns:
            loss: Scalar loss value
        """
        B, seq_len, vocab_size = logits.shape
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )
        
        # Apply abnormal mask if provided
        if abnormal_mask is not None:
            shift_mask = abnormal_mask[:, 1:].contiguous().view(-1)
            loss = loss * shift_mask
            loss = loss.sum() / shift_mask.sum().clamp(min=1)
        else:
            # Ignore padding
            valid_mask = (shift_labels != self.ignore_index).float()
            loss = (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        
        return loss