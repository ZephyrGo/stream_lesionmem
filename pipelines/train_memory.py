"""
Training Pipeline for STREAM-LesionMem.

Trains Memory Bank (detector, matcher, updater) and Router components.
MedGemma backbone is frozen.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from models.stream_lesionmem_model import StreamLesionMemModel
from models.losses import MemoryBankLoss, DetectionLoss, RouterLoss
from data.qlendo_dataset import QLEndoDataset, collate_fn, create_dataloaders


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    train_json: str = "data/train.json"
    val_json: Optional[str] = "data/val.json"
    sample_frames: int = 12
    image_size: Tuple[int, int] = (336, 336)
    
    # Model
    medgemma_path: str = "google/medgemma-4b-it"
    use_dummy: bool = True  # Use dummy model for testing
    num_slots: int = 16
    slot_dim: int = 512
    num_sections: int = 9
    router_mode: str = "learned"  # Must be learned for training
    chunk_size: int = 4
    max_frames_for_llm: int = 4
    
    # Training
    batch_size: int = 2
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    
    # Loss weights
    detection_weight: float = 1.0
    router_weight: float = 1.0
    matching_weight: float = 0.5
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "checkpoints"
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Training Pipeline
# =============================================================================

class TrainingPipeline:
    """
    Training pipeline for STREAM-LesionMem.
    
    Trains:
    - LesionDetector: Detect abnormal tokens
    - SlotMatcher: Match candidates to slots
    - SlotUpdater: Update slot embeddings
    - SectionRouter: Predict section and abnormality
    
    Frozen:
    - MedGemma vision_tower
    - MedGemma multi_modal_projector
    - MedGemma language_model
    """
    
    def __init__(
        self,
        config: TrainConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("TrainingPipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(handler)
        
        return logger
    
    def setup(self) -> None:
        """Setup all components for training."""
        self.logger.info("Setting up training pipeline...")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        self.logger.info(f"Config saved to {config_path}")
        
        # Setup model
        self._setup_model()
        
        # Setup data
        self._setup_data()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Setup loss
        self._setup_loss()
        
        # Setup tensorboard
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(output_dir / "tensorboard")
        
        # Setup mixed precision
        if self.config.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
        
        self.logger.info("Training pipeline setup complete!")
    
    def _setup_model(self) -> None:
        """Setup model."""
        self.logger.info("Setting up model...")
        
        self.model = StreamLesionMemModel(
            medgemma_path=self.config.medgemma_path,
            use_dummy=self.config.use_dummy,
            freeze_medgemma=True,  # Always freeze backbone
            num_slots=self.config.num_slots,
            slot_dim=self.config.slot_dim,
            num_sections=self.config.num_sections,
            router_mode=self.config.router_mode,
            chunk_size=self.config.chunk_size,
            max_frames_for_llm=self.config.max_frames_for_llm,
        )
        
        self.model = self.model.to(self.device)
        
        # Log parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
    def _setup_data(self) -> None:
        """Setup data loaders."""
        self.logger.info("Setting up data loaders...")
        
        self.train_loader, self.val_loader = create_dataloaders(
            train_json=self.config.train_json,
            val_json=self.config.val_json,
            batch_size=self.config.batch_size,
            sample_frames=self.config.sample_frames,
            image_size=self.config.image_size,
            num_workers=self.config.num_workers,
            augment_train=True,
        )
        
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            self.logger.info(f"Val batches: {len(self.val_loader)}")
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and scheduler."""
        self.logger.info("Setting up optimizer...")
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = min(self.config.warmup_steps, total_steps // 10)
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
        )
        
        self.logger.info(f"Total training steps: {total_steps}")
        self.logger.info(f"Warmup steps: {warmup_steps}")
    
    def _setup_loss(self) -> None:
        """Setup loss function."""
        self.loss_fn = MemoryBankLoss(
            detection_weight=self.config.detection_weight,
            router_weight=self.config.router_weight,
            matching_weight=self.config.matching_weight,
        )
    
    def train(self) -> None:
        """Run training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch}/{self.config.num_epochs}")
            self.logger.info(f"{'='*60}")
            
            # Train one epoch
            train_metrics = self._train_epoch(epoch)
            
            # Log epoch metrics
            self.logger.info(f"Epoch {epoch} training metrics:")
            for k, v in train_metrics.items():
                self.logger.info(f"  {k}: {v:.4f}")
            
            # Validation
            if self.val_loader and epoch % 1 == 0:  # Validate every epoch
                val_metrics = self._validate()
                
                self.logger.info(f"Epoch {epoch} validation metrics:")
                for k, v in val_metrics.items():
                    self.logger.info(f"  {k}: {v:.4f}")
                
                # Save best model
                val_loss = val_metrics.get("val_loss", float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best.pt", epoch, val_metrics)
            
            # Save periodic checkpoint
            self._save_checkpoint(f"epoch_{epoch}.pt", epoch, train_metrics)
        
        # Save final model
        self._save_checkpoint("final.pt", self.config.num_epochs, train_metrics)
        
        self.logger.info("\nTraining complete!")
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        # Ensure backbone is frozen
        self.model.adapter.eval()
        for param in self.model.adapter.parameters():
            param.requires_grad = False
        
        epoch_losses = {
            "total_loss": 0.0,
            "detection_loss": 0.0,
            "router_loss": 0.0,
            "matching_loss": 0.0,
        }
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            frames = batch["frames"].to(self.device)
            frame_masks = batch["frame_masks"].to(self.device)
            section_labels = batch["section_ids"].to(self.device)
            abnormal_labels = batch["abnormal_labels"].to(self.device)
            frame2section = batch["frame2section"]
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.config.use_amp and self.scaler is not None):
                losses = self._forward_step(
                    frames=frames,
                    frame_masks=frame_masks,
                    section_labels=section_labels,
                    abnormal_labels=abnormal_labels,
                    frame2section=frame2section,
                )
            
            loss = losses["total_loss"] / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate losses
            for k, v in losses.items():
                if k in epoch_losses:
                    epoch_losses[k] += v.item()
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = epoch_losses["total_loss"] / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                
                self.logger.info(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
                
                # Tensorboard logging
                if self.writer:
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
            
            # Periodic validation
            if self.val_loader and self.global_step > 0 and \
               self.global_step % self.config.eval_interval == 0:
                val_metrics = self._validate()
                self.logger.info(f"  [Step {self.global_step}] Val loss: {val_metrics['val_loss']:.4f}")
            
            # Periodic save
            if self.global_step > 0 and self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f"step_{self.global_step}.pt", epoch, epoch_losses)
        
        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)
        
        return epoch_losses
    
    def _forward_step(
        self,
        frames: torch.Tensor,
        frame_masks: torch.Tensor,
        section_labels: torch.Tensor,
        abnormal_labels: torch.Tensor,
        frame2section: List[Dict[int, int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for one training step.
        
        Returns loss dict.
        """
        B, K = frames.shape[:2]
        device = frames.device
        
        # Reset memory for each sample
        self.model.memory.reset()
        self.model.encoder.clear_cache()
        
        # Process each sample in batch separately (memory bank is stateful)
        all_detection_conf = []
        all_section_logits = []
        all_abnormal_scores = []
        all_slot_embeddings = []
        all_slot_section_ids = []
        all_slot_confidences = []
        
        # Get num_patches from model config (default 256 for dummy, varies for real model)
        num_patches = getattr(self.model.adapter, 'num_vision_tokens', 256)
        
        for b in range(B):
            # Get single sample
            sample_frames = frames[b:b+1]  # [1, K, 3, H, W]
            sample_f2s = frame2section[b]
            sample_mask = frame_masks[b]  # [K]
            
            # Count valid frames for this sample
            valid_k = sample_mask.sum().item()
            
            # Convert frame2section to list
            f2s_list = [sample_f2s.get(i, 0) for i in range(K)]
            
            # Reset memory
            self.model.memory.reset()
            
            # Pre-allocate detection confidence tensor with fixed size K
            detection_conf_b = torch.zeros(1, K, num_patches, device=device, dtype=torch.float32)
            
            # Encode and process frames
            frame_tokens_list = []
            
            for lm_tokens, chunk_indices in self.model.encoder.encode_streaming(
                sample_frames,
                cache_indices=list(range(K)),
            ):
                # lm_tokens: [1, chunk_size, num_patches, lm_hidden]
                frame_tokens_list.append(lm_tokens)
                
                # Update num_patches from actual output if different
                actual_num_patches = lm_tokens.shape[2]
                if actual_num_patches != num_patches:
                    # Reallocate with correct size
                    detection_conf_b = torch.zeros(1, K, actual_num_patches, device=device, dtype=torch.float32)
                    num_patches = actual_num_patches
                
                # Process each frame in chunk through memory
                for i, global_idx in enumerate(chunk_indices):
                    if global_idx >= K or not sample_mask[global_idx]:
                        continue
                    
                    frame_tokens = lm_tokens[:, i]  # [1, num_patches, lm_hidden]
                    section_id = f2s_list[global_idx]
                    
                    # Detect candidates
                    detection = self.model.memory.detect(frame_tokens)
                    # Store detection confidence at the correct position
                    detection_conf_b[0, global_idx] = detection.confidences[0]  # [num_patches]
                    
                    # Process frame in memory
                    self.model.memory.process_frame(
                        frame_tokens,
                        frame_idx=global_idx,
                        section_id=section_id,
                        original_tokens=frame_tokens,
                    )
            
            # Concatenate all frame tokens
            all_lm_tokens = torch.cat(frame_tokens_list, dim=1)  # [1, K, num_patches, lm_hidden]
            
            # Get router predictions
            section_logits_b, abnormal_scores_b = self._get_router_predictions(
                all_lm_tokens, f2s_list
            )
            
            all_detection_conf.append(detection_conf_b)
            all_section_logits.append(section_logits_b)
            all_abnormal_scores.append(abnormal_scores_b)
            
            # Collect slot info for matching loss
            if self.model.memory.slots:
                slot_embs = torch.stack([s.embedding for s in self.model.memory.slots])
                slot_secs = torch.tensor([s.section_id for s in self.model.memory.slots], device=device)
                slot_confs = torch.tensor([s.confidence for s in self.model.memory.slots], device=device)
                
                all_slot_embeddings.append(slot_embs)
                all_slot_section_ids.append(slot_secs)
                all_slot_confidences.append(slot_confs)
        
        # Stack batch
        detection_conf = torch.cat(all_detection_conf, dim=0)  # [B, K, num_patches]
        section_logits = torch.cat(all_section_logits, dim=0)  # [B, K, num_sections]
        abnormal_scores = torch.cat(all_abnormal_scores, dim=0)  # [B, K]
        
        # Compute loss
        losses = self.loss_fn(
            detection_confidences=detection_conf,
            section_logits=section_logits,
            abnormal_scores=abnormal_scores,
            section_labels=section_labels,
            abnormal_labels=abnormal_labels,
            frame_masks=frame_masks,
            slot_embeddings=all_slot_embeddings[0] if all_slot_embeddings else None,
            slot_section_ids=all_slot_section_ids[0] if all_slot_section_ids else None,
            slot_confidences=all_slot_confidences[0] if all_slot_confidences else None,
        )
        
        return losses
    
    def _get_router_predictions(
        self,
        lm_tokens: torch.Tensor,
        frame2section: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get router predictions with learned mode."""
        B, K, num_patches, lm_hidden = lm_tokens.shape
        device = lm_tokens.device
        
        if self.model.router.mode.value == "learned":
            # Get predictions from learned router
            section_logits_list = []
            abnormal_scores_list = []
            
            for k in range(K):
                frame_tokens = lm_tokens[:, k]  # [B, num_patches, lm_hidden]
                
                # Mean pooling
                pooled = frame_tokens.mean(dim=1)  # [B, lm_hidden]
                
                # Get logits
                features = torch.relu(self.model.router.pool_proj(pooled))
                section_logits_k = self.model.router.section_head(features)  # [B, num_sections]
                abnormal_logits_k = self.model.router.abnormal_head(features).squeeze(-1)  # [B]
                
                section_logits_list.append(section_logits_k)
                abnormal_scores_list.append(torch.sigmoid(abnormal_logits_k))
            
            section_logits = torch.stack(section_logits_list, dim=1)  # [B, K, num_sections]
            abnormal_scores = torch.stack(abnormal_scores_list, dim=1)  # [B, K]
        else:
            # Rule mode - use labels directly
            num_sections = self.model.router.num_sections
            section_logits = torch.zeros(B, K, num_sections, device=device)
            for k, sec_id in enumerate(frame2section):
                if sec_id < num_sections:
                    section_logits[:, k, sec_id] = 10.0  # High logit for correct section
            
            abnormal_scores = torch.zeros(B, K, device=device)
        
        return section_logits, abnormal_scores
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        val_losses = {
            "val_loss": 0.0,
            "val_detection_loss": 0.0,
            "val_router_loss": 0.0,
        }
        num_batches = 0
        
        for batch in self.val_loader:
            frames = batch["frames"].to(self.device)
            frame_masks = batch["frame_masks"].to(self.device)
            section_labels = batch["section_ids"].to(self.device)
            abnormal_labels = batch["abnormal_labels"].to(self.device)
            frame2section = batch["frame2section"]
            
            with autocast('cuda', enabled=self.config.use_amp and self.scaler is not None):
                losses = self._forward_step(
                    frames=frames,
                    frame_masks=frame_masks,
                    section_labels=section_labels,
                    abnormal_labels=abnormal_labels,
                    frame2section=frame2section,
                )
            
            val_losses["val_loss"] += losses["total_loss"].item()
            val_losses["val_detection_loss"] += losses["detection_loss"].item()
            val_losses["val_router_loss"] += losses["router_loss"].item()
            num_batches += 1
        
        # Average
        for k in val_losses:
            val_losses[k] /= max(num_batches, 1)
        
        # Log to tensorboard
        if self.writer:
            for k, v in val_losses.items():
                self.writer.add_scalar(k, v, self.global_step)
        
        self.model.train()
        return val_losses
    
    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save training checkpoint."""
        output_dir = Path(self.config.output_dir)
        
        # Save only trainable components
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": {
                "memory_bank": self.model.memory.state_dict(),
                "router": self.model.router.state_dict(),
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        save_path = output_dir / filename
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return starting epoch."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if "memory_bank" in state_dict:
                self.model.memory.load_state_dict(state_dict["memory_bank"])
            if "router" in state_dict:
                self.model.router.load_state_dict(state_dict["router"])
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint.get("epoch", 0) + 1
        
        self.logger.info(f"Resumed from epoch {start_epoch - 1}, step {self.global_step}")
        
        return start_epoch


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train STREAM-LesionMem")
    
    # Data
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, default=None)
    
    # Model
    parser.add_argument("--medgemma_path", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--use_dummy", action="store_true", help="Use dummy model")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--slot_dim", type=int, default=512)
    parser.add_argument("--num_sections", type=int, default=9)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Build config
    config = TrainConfig(
        train_json=args.train_json,
        val_json=args.val_json,
        medgemma_path=args.medgemma_path,
        use_dummy=args.use_dummy,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        num_sections=args.num_sections,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
    )
    
    # Create pipeline
    pipeline = TrainingPipeline(config)
    pipeline.setup()
    
    # Resume if specified
    if args.resume:
        start_epoch = pipeline.load_checkpoint(args.resume)
    
    # Train
    pipeline.train()


if __name__ == "__main__":
    main()