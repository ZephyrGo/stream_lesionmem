#!/usr/bin/env python3
"""
Train STREAM-LesionMem on QL_Endo Dataset.

Usage:
    # Step 1: Preprocess data
    python scripts/train_qlendo.py preprocess \
        --input /path/to/diagnosis_en.json \
        --output data/qlendo/preprocessed.json
    
    # Step 2: Train model
    python scripts/train_qlendo.py train \
        --data_dir data/qlendo \
        --medgemma_path /path/to/medgemma \
        --output_dir checkpoints/qlendo
    
    # Or use dummy model for testing:
    python scripts/train_qlendo.py train \
        --data_dir data/qlendo \
        --use_dummy \
        --output_dir checkpoints/test
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def preprocess_command(args):
    """Run data preprocessing."""
    from data.preprocess_qlendo import preprocess_qlendo_dataset, create_train_val_split
    
    print("="*60)
    print("STREAM-LesionMem Data Preprocessing")
    print("="*60)
    
    # Preprocess
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = preprocess_qlendo_dataset(
        input_json=args.input,
        output_json=str(output_path),
        min_images=args.min_images,
        max_images=args.max_images,
    )
    
    # Create train/val split
    if args.split:
        output_dir = output_path.parent
        create_train_val_split(
            str(output_path),
            str(output_dir / "train.json"),
            str(output_dir / "val.json"),
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    
    print("\nPreprocessing complete!")


def train_command(args):
    """Run training."""
    from pipelines.train_memory import TrainingPipeline, TrainConfig
    
    print("="*60)
    print("STREAM-LesionMem Training")
    print("="*60)
    
    # Build paths
    data_dir = Path(args.data_dir)
    train_json = str(data_dir / "train.json")
    val_json = str(data_dir / "val.json") if (data_dir / "val.json").exists() else None
    
    # Build config
    config = TrainConfig(
        # Data
        train_json=train_json,
        val_json=val_json,
        sample_frames=args.sample_frames,
        image_size=(args.image_size, args.image_size),
        
        # Model
        medgemma_path=args.medgemma_path,
        use_dummy=args.use_dummy,
        num_slots=args.num_slots,
        slot_dim=args.slot_dim,
        num_sections=args.num_sections,
        router_mode="learned",
        chunk_size=args.chunk_size,
        max_frames_for_llm=args.max_frames_for_llm,
        
        # Training
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Loss weights
        detection_weight=args.detection_weight,
        router_weight=args.router_weight,
        matching_weight=args.matching_weight,
        
        # Mixed precision
        use_amp=not args.no_amp,
        
        # Logging
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        
        # Hardware
        device=args.device,
        num_workers=args.num_workers,
    )
    
    # Print config summary
    print(f"\nConfiguration:")
    print(f"  Train data: {train_json}")
    print(f"  Val data: {val_json}")
    print(f"  MedGemma: {'Dummy' if args.use_dummy else args.medgemma_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output: {args.output_dir}")
    print()
    
    # Create pipeline
    pipeline = TrainingPipeline(config)
    pipeline.setup()
    
    # Resume if specified
    if args.resume:
        pipeline.load_checkpoint(args.resume)
    
    # Train
    pipeline.train()
    
    print("\nTraining complete!")


def eval_command(args):
    """Run evaluation."""
    import torch
    from models.stream_lesionmem_model import StreamLesionMemModel
    from data.qlendo_dataset import QLEndoDataset, collate_fn
    from torch.utils.data import DataLoader
    from utils.metrics import compute_report_metrics
    import json
    
    print("="*60)
    print("STREAM-LesionMem Evaluation")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    
    model = StreamLesionMemModel(
        medgemma_path=args.medgemma_path,
        use_dummy=args.use_dummy,
        num_slots=args.num_slots,
        num_sections=args.num_sections,
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if "memory_bank" in state_dict:
            model.memory.load_state_dict(state_dict["memory_bank"])
        if "router" in state_dict:
            model.router.load_state_dict(state_dict["router"])
    
    model = model.to(device)
    model.eval()
    
    # Load test data
    print(f"Loading test data from {args.test_json}...")
    
    test_dataset = QLEndoDataset(
        data_json=args.test_json,
        sample_frames=args.sample_frames,
        augment=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Generate one at a time
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Generate reports
    print(f"\nGenerating reports for {len(test_dataset)} samples...")
    
    all_predictions = []
    all_references = []
    results = []
    
    for batch in test_loader:
        frames = batch["frames"].to(device)
        frame2section = batch["frame2section"]
        reference = batch["reports"][0]
        exam_id = batch["exam_ids"][0]
        
        with torch.no_grad():
            output = model.generate(
                frames=frames,
                frame2section=frame2section,
                max_new_tokens=args.max_new_tokens,
            )
        
        prediction = output["final_report"]
        
        all_predictions.append(prediction)
        all_references.append(reference)
        
        results.append({
            "exam_id": exam_id,
            "prediction": prediction,
            "reference": reference,
            "abnormal_sections": output["abnormal_sections"],
            "selected_frames": output["selected_frames"],
        })
        
        if len(results) % 10 == 0:
            print(f"  Processed {len(results)}/{len(test_dataset)}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_report_metrics(all_predictions, all_references)
    
    print("\nEvaluation Results:")
    print("-"*40)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "predictions.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="STREAM-LesionMem Training and Evaluation on QL_Endo"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    preprocess_parser.add_argument("--input", type=str, required=True,
                                    help="Input diagnosis_en.json path")
    preprocess_parser.add_argument("--output", type=str, required=True,
                                    help="Output preprocessed JSON path")
    preprocess_parser.add_argument("--min_images", type=int, default=10)
    preprocess_parser.add_argument("--max_images", type=int, default=200)
    preprocess_parser.add_argument("--split", action="store_true",
                                    help="Create train/val split")
    preprocess_parser.add_argument("--val_ratio", type=float, default=0.1)
    preprocess_parser.add_argument("--seed", type=int, default=42)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    
    # Data args
    train_parser.add_argument("--data_dir", type=str, required=True,
                              help="Directory with train.json and val.json")
    train_parser.add_argument("--sample_frames", type=int, default=12)
    train_parser.add_argument("--image_size", type=int, default=336)
    
    # Model args
    train_parser.add_argument("--medgemma_path", type=str, 
                              default="google/medgemma-4b-it")
    train_parser.add_argument("--use_dummy", action="store_true",
                              help="Use dummy model for testing")
    train_parser.add_argument("--num_slots", type=int, default=16)
    train_parser.add_argument("--slot_dim", type=int, default=512)
    train_parser.add_argument("--num_sections", type=int, default=9)
    train_parser.add_argument("--chunk_size", type=int, default=4)
    train_parser.add_argument("--max_frames_for_llm", type=int, default=4)
    
    # Training args
    train_parser.add_argument("--batch_size", type=int, default=2)
    train_parser.add_argument("--num_epochs", type=int, default=20)
    train_parser.add_argument("--learning_rate", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.01)
    train_parser.add_argument("--warmup_steps", type=int, default=500)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Loss weights
    train_parser.add_argument("--detection_weight", type=float, default=1.0)
    train_parser.add_argument("--router_weight", type=float, default=1.0)
    train_parser.add_argument("--matching_weight", type=float, default=0.5)
    
    # Other args
    train_parser.add_argument("--no_amp", action="store_true",
                              help="Disable mixed precision")
    train_parser.add_argument("--log_interval", type=int, default=10)
    train_parser.add_argument("--eval_interval", type=int, default=500)
    train_parser.add_argument("--save_interval", type=int, default=1000)
    train_parser.add_argument("--output_dir", type=str, default="checkpoints")
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--resume", type=str, default=None,
                              help="Resume from checkpoint")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--test_json", type=str, required=True)
    eval_parser.add_argument("--medgemma_path", type=str,
                             default="google/medgemma-4b-it")
    eval_parser.add_argument("--use_dummy", action="store_true")
    eval_parser.add_argument("--num_slots", type=int, default=16)
    eval_parser.add_argument("--num_sections", type=int, default=9)
    eval_parser.add_argument("--sample_frames", type=int, default=12)
    eval_parser.add_argument("--max_new_tokens", type=int, default=512)
    eval_parser.add_argument("--output_dir", type=str, default="eval_results")
    eval_parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        preprocess_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "eval":
        eval_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
