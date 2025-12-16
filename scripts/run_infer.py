#!/usr/bin/env python3
"""
Run inference with STREAM-LesionMem model.

Usage:
    python scripts/run_infer.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/best.pt \
        --input_video data/test_video.mp4 \
        --output_dir outputs/
    
    # Or batch inference:
    python scripts/run_infer.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/best.pt \
        --test_jsonl data/test.jsonl \
        --output_dir outputs/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stream_lesionmem_model import StreamLesionMemModel
from models.medgemma_adapter import MedGemmaAdapter, DummyMedGemmaModel
from data.templates import TemplateLibrary
from data.preprocess import load_video_frames, preprocess_frames
from utils.logging import get_logger
from utils.metrics import compute_report_metrics


logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda'
) -> StreamLesionMemModel:
    """Load model with optional checkpoint."""
    
    model_config = config.get('model', {})
    
    # Initialize MedGemma adapter
    medgemma_path = model_config.get('medgemma_path', None)
    use_dummy = model_config.get('use_dummy', True)
    
    if use_dummy or medgemma_path is None:
        logger.info("Using DummyMedGemmaModel for inference")
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
        logger.info(f"Loading templates from {template_path}")
        templates = TemplateLibrary.load(template_path)
    else:
        logger.info("Using default templates")
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
        router_mode=model_config.get('router_mode', 'rule')
    )
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load trainable parameters only
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded successfully")
    
    model = model.to(device)
    model.eval()
    
    return model


def run_single_inference(
    model: StreamLesionMemModel,
    video_path: str,
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Dict[str, Any]:
    """Run inference on a single video."""
    
    infer_config = config.get('inference', {})
    data_config = config.get('data', {})
    
    # Load and preprocess video frames
    logger.info(f"Loading video: {video_path}")
    
    if Path(video_path).exists():
        frames = load_video_frames(
            video_path,
            max_frames=infer_config.get('max_frames', 16),
            sample_strategy=infer_config.get('sample_strategy', 'uniform')
        )
        frames = preprocess_frames(
            frames,
            image_size=data_config.get('image_size', 224)
        )
    else:
        # Use dummy frames for testing
        logger.warning(f"Video not found, using dummy frames")
        num_frames = infer_config.get('max_frames', 16)
        frames = torch.randn(num_frames, 3, 224, 224)
    
    frames = frames.unsqueeze(0).to(device)  # Add batch dimension
    
    # Create frame2section mapping (if not available, use uniform assignment)
    num_frames = frames.shape[1]
    num_sections = config.get('model', {}).get('num_sections', 8)
    frame2section = {i: i % num_sections for i in range(num_frames)}
    
    # Run generation
    start_time = time.time()
    
    with torch.no_grad():
        report = model.generate(
            frames=frames,
            frame2section=[frame2section],
            max_new_tokens=infer_config.get('max_new_tokens', 512),
            temperature=infer_config.get('temperature', 0.7),
            top_p=infer_config.get('top_p', 0.9)
        )
    
    inference_time = time.time() - start_time
    
    # Report is a list (batch), get first item
    if isinstance(report, list):
        report = report[0]
    
    result = {
        'video_path': video_path,
        'report': report,
        'inference_time_seconds': inference_time,
        'num_frames_processed': num_frames
    }
    
    return result


def run_batch_inference(
    model: StreamLesionMemModel,
    test_jsonl: str,
    config: Dict[str, Any],
    device: str = 'cuda',
    output_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Run inference on a batch of videos from JSONL."""
    
    # Load test records
    records = []
    with open(test_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    logger.info(f"Running inference on {len(records)} samples")
    
    results = []
    all_predictions = []
    all_references = []
    
    for record in tqdm(records, desc="Inference"):
        video_path = record.get('video_path', record.get('video_id', ''))
        
        # Run inference
        result = run_single_inference(model, video_path, config, device)
        
        # Add ground truth if available
        if 'report' in record:
            result['reference_report'] = record['report']
            all_predictions.append(result['report'])
            all_references.append(record['report'])
        
        result['video_id'] = record.get('video_id', Path(video_path).stem)
        results.append(result)
        
        # Save intermediate results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / 'predictions.jsonl', 'a') as f:
                f.write(json.dumps(result) + '\n')
    
    # Compute metrics if we have references
    if all_references:
        logger.info("Computing evaluation metrics...")
        metrics = compute_report_metrics(all_predictions, all_references)
        
        logger.info("Evaluation Results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        if output_dir:
            with open(Path(output_dir) / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run STREAM-LesionMem inference"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input_video',
        type=str,
        default=None,
        help='Path to single input video'
    )
    parser.add_argument(
        '--test_jsonl',
        type=str,
        default=None,
        help='Path to test JSONL for batch inference'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.input_video is None and args.test_jsonl is None:
        logger.warning("No input specified, running with dummy data for testing")
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Load model
    model = load_model(config, args.checkpoint, args.device)
    logger.info("Model loaded successfully")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.test_jsonl:
        # Batch inference
        results = run_batch_inference(
            model, args.test_jsonl, config, args.device, args.output_dir
        )
        logger.info(f"Batch inference complete. {len(results)} samples processed.")
        
    elif args.input_video:
        # Single video inference
        result = run_single_inference(model, args.input_video, config, args.device)
        
        # Print result
        print("\n" + "="*60)
        print("GENERATED REPORT")
        print("="*60)
        print(result['report'])
        print("="*60)
        print(f"Inference time: {result['inference_time_seconds']:.2f}s")
        print(f"Frames processed: {result['num_frames_processed']}")
        
        # Save result
        output_path = output_dir / 'single_inference_result.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {output_path}")
        
    else:
        # Demo with dummy data
        logger.info("Running demo inference with dummy data...")
        result = run_single_inference(
            model, 
            'dummy_video.mp4',  # Will use dummy frames
            config, 
            args.device
        )
        
        print("\n" + "="*60)
        print("DEMO GENERATED REPORT")
        print("="*60)
        print(result['report'])
        print("="*60)
        print(f"Inference time: {result['inference_time_seconds']:.2f}s")


if __name__ == '__main__':
    main()
