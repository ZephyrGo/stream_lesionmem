#!/usr/bin/env python3
"""
Train STREAM-LesionMem model.

Trains memory bank (detector, matcher, updater) and router components.
MedGemma weights are frozen.

Usage:
    python scripts/run_train.py \
        --config configs/default.yaml \
        --train_jsonl data/train.jsonl \
        --val_jsonl data/val.jsonl \
        --output_dir checkpoints/
    
    # Resume training:
    python scripts/run_train.py \
        --config configs/default.yaml \
        --resume checkpoints/latest.pt
"""

import argparse
import json
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stream_lesionmem_model import StreamLesionMemModel
from models.medgemma_adapter import MedGemmaAdapter, DummyMedGemmaModel
from data.dataset import EndoscopyDataset, collate_fn
from data.templates import TemplateLibrary
from utils.logging import get_logger, setup_logging
from utils.metrics import compute_report_metrics


logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Tuple[StreamLesionMemModel, Any]:
    """Create model and return with processor."""
    
    model_config = config.get('model', {})
    
    # Initialize MedGemma adapter
    medgemma_path = model_config.get('medgemma_path', None)
    use_dummy = model_config.get('use_dummy', True)
    
    if use_dummy or medgemma_path is None:
        logger.info("Using DummyMedGemmaModel for training")
        medgemma = DummyMedGemmaModel(
            lm_hidden_size=model_config.get('lm_hidden_size', 2560),
            vision_hidden_size=model_config.get('vision_hidden_size', 1152),
            vocab_size=model_config.get('vocab_size', 262144)
        )
        processor = None
    else:
        logger.info(f"Loading MedGemma from {medgemma_path}")
        adapter = MedGemmaAdapter(medgemma_path, device=device)
        medgemma = adapter.model
        processor = adapter.processor
    
    # Load template library
    template_path = config.get('data', {}).get('template_path', 'data/templates.json')
    if Path(template_path).exists():
        templates = TemplateLibrary.load(template_path)
    else:
        templates = TemplateLibrary()
    
    # Create model
    model = StreamLesionMemModel(
        medgemma=medgemma,
        processor=processor,
        template_library=templates,
        num_slots=model_config.get('num_slots', 16),
        slot_dim=model_config.get('slot_dim', 512),
        lm_hidden_size=model_config.get('lm_hidden_size', 2560),
        vision_hidden_size=model_config.get('vision_hidden_size', 1152),
        top_m_candidates=model_config.get('top_m_candidates', 8),
        top_l_evidence=model_config.get('top_l_evidence', 4),
        similarity_threshold=model_config.get('similarity_threshold', 0.7),
        num_sections=model_config.get('num_sections', 8),
        abnormal_threshold=model_config.get('abnormal_threshold', 0.5),
        max_evidence_frames=model_config.get('max_evidence_frames', 4),
        chunk_size=model_config.get('chunk_size', 4),
        router_mode=model_config.get('router_mode', 'learned')  # Use learned for training
    )
    
    return model, processor


def create_dataloaders(
    config: Dict[str, Any],
    processor: Any
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders."""
    
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    
    # Create datasets
    train_dataset = EndoscopyDataset(
        data_path=data_config.get('train_path', 'data/train.jsonl'),
        max_frames=data_config.get('max_frames', 16),
        image_size=data_config.get('image_size', 224),
        sample_strategy=data_config.get('sample_strategy', 'uniform'),
        use_dummy=data_config.get('use_dummy', True)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 2),
        shuffle=True,
        num_workers=train_config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation loader
    val_loader = None
    val_path = data_config.get('val_path', 'data/val.jsonl')
    if val_path and Path(val_path).exists():
        val_dataset = EndoscopyDataset(
            data_path=val_path,
            max_frames=data_config.get('max_frames', 16),
            image_size=data_config.get('image_size', 224),
            sample_strategy=data_config.get('sample_strategy', 'uniform'),
            use_dummy=data_config.get('use_dummy', True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.get('batch_size', 2),
            shuffle=False,
            num_workers=train_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, val_loader


def get_trainable_parameters(model: StreamLesionMemModel) -> List[nn.Parameter]:
    """Get list of trainable parameters (memory bank + router only)."""
    trainable_params = []
    
    # Memory bank parameters
    trainable_params.extend(model.memory_bank.parameters())
    
    # Router parameters (if learned mode)
    trainable_params.extend(model.router.parameters())
    
    return trainable_params


def train_epoch(
    model: StreamLesionMemModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    device: str,
    epoch: int,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    
    # Freeze MedGemma
    model.streaming_encoder.medgemma_adapter.eval()
    for param in model.streaming_encoder.medgemma_adapter.parameters():
        param.requires_grad = False
    
    total_loss = 0.0
    total_lm_loss = 0.0
    total_router_loss = 0.0
    num_batches = 0
    
    train_config = config.get('training', {})
    grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
    max_grad_norm = train_config.get('max_grad_norm', 1.0)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        frames = batch['frames'].to(device)
        frame2section = batch['frame2section']
        
        # Prepare labels (simplified - in practice, need proper tokenization)
        # For now, we'll use dummy labels
        labels = None  # TODO: Proper label preparation from batch['report']
        
        # Forward pass
        outputs = model.forward(
            frames=frames,
            frame2section=frame2section,
            labels=labels,
            return_loss=True
        )
        
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps
        loss.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                get_trainable_parameters(model),
                max_grad_norm
            )
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        # Track losses
        total_loss += outputs['loss'].item()
        if 'lm_loss' in outputs:
            total_lm_loss += outputs['lm_loss'].item()
        if 'router_loss' in outputs:
            total_router_loss += outputs['router_loss'].item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / num_batches,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    metrics = {
        'train_loss': total_loss / num_batches,
        'train_lm_loss': total_lm_loss / num_batches,
        'train_router_loss': total_router_loss / num_batches
    }
    
    return metrics


@torch.no_grad()
def validate(
    model: StreamLesionMemModel,
    val_loader: DataLoader,
    device: str,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Run validation."""
    
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_references = []
    
    for batch in tqdm(val_loader, desc="Validation"):
        frames = batch['frames'].to(device)
        frame2section = batch['frame2section']
        
        # Generate reports
        reports = model.generate(
            frames=frames,
            frame2section=frame2section,
            max_new_tokens=config.get('inference', {}).get('max_new_tokens', 256)
        )
        
        all_predictions.extend(reports)
        
        # Collect references if available
        if 'report' in batch:
            all_references.extend(batch['report'])
        
        num_batches += 1
    
    metrics = {}
    
    # Compute generation metrics if we have references
    if all_references:
        gen_metrics = compute_report_metrics(all_predictions, all_references)
        metrics.update({f'val_{k}': v for k, v in gen_metrics.items()})
    
    return metrics


def save_checkpoint(
    model: StreamLesionMemModel,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    output_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    
    # Save only trainable parameters
    trainable_state = {
        'memory_bank': model.memory_bank.state_dict(),
        'router': model.router.state_dict()
    }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': trainable_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    # Save latest
    torch.save(checkpoint, output_dir / 'latest.pt')
    
    # Save periodic checkpoint
    if epoch % 5 == 0:
        torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Save best
    if is_best:
        torch.save(checkpoint, output_dir / 'best.pt')
        logger.info(f"New best model saved at epoch {epoch}")


def load_checkpoint(
    model: StreamLesionMemModel,
    optimizer: optim.Optimizer,
    scheduler: Any,
    checkpoint_path: str,
    device: str
) -> int:
    """Load checkpoint and return starting epoch."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load trainable parameters
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'memory_bank' in state_dict:
            model.memory_bank.load_state_dict(state_dict['memory_bank'])
        if 'router' in state_dict:
            model.router.load_state_dict(state_dict['router'])
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    
    logger.info(f"Resumed from epoch {start_epoch - 1}")
    
    return start_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Train STREAM-LesionMem model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--train_jsonl',
        type=str,
        default=None,
        help='Override train data path'
    )
    parser.add_argument(
        '--val_jsonl',
        type=str,
        default=None,
        help='Override validation data path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/',
        help='Output directory for checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir / 'train.log')
    
    # Load config
    config = load_config(args.config)
    
    # Override data paths if provided
    if args.train_jsonl:
        config.setdefault('data', {})['train_path'] = args.train_jsonl
    if args.val_jsonl:
        config.setdefault('data', {})['val_path'] = args.val_jsonl
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Create model
    model, processor = create_model(config, args.device)
    model = model.to(args.device)
    
    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in get_trainable_parameters(model))
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, processor)
    logger.info(f"Train batches: {len(train_loader)}")
    if val_loader:
        logger.info(f"Val batches: {len(val_loader)}")
    
    # Setup optimizer
    train_config = config.get('training', {})
    
    optimizer = optim.AdamW(
        get_trainable_parameters(model),
        lr=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
        betas=(0.9, 0.999)
    )
    
    # Setup scheduler
    num_epochs = train_config.get('num_epochs', 20)
    warmup_steps = train_config.get('warmup_steps', 500)
    total_steps = len(train_loader) * num_epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config.get('learning_rate', 1e-4),
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos'
    )
    
    # Resume if specified
    start_epoch = 1
    best_val_metric = float('inf')
    
    if args.resume:
        start_epoch = load_checkpoint(
            model, optimizer, scheduler, args.resume, args.device
        )
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            args.device, epoch, config
        )
        
        # Validate
        val_metrics = {}
        if val_loader and epoch % train_config.get('val_every', 1) == 0:
            val_metrics = validate(model, val_loader, args.device, config)
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
        
        epoch_time = time.time() - epoch_start
        
        # Log
        logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        logger.info(f"  Train loss: {train_metrics['train_loss']:.4f}")
        if val_metrics:
            for k, v in val_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
        
        # Save checkpoint
        is_best = False
        if 'val_rouge1' in val_metrics:
            # Higher ROUGE is better, so negate for comparison
            current_metric = -val_metrics['val_rouge1']
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                is_best = True
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, metrics,
            output_dir, is_best
        )
        
        # Log to file
        with open(output_dir / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    logger.info("Training complete!")
    logger.info(f"Best model saved to {output_dir / 'best.pt'}")


if __name__ == '__main__':
    main()
