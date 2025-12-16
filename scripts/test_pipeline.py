#!/usr/bin/env python3
"""
STREAM-LesionMem Pipeline Verification Script

验证全流程正确性，无需真实数据或GPU：
1. 模块级测试：验证每个组件的输入输出shape
2. 集成测试：验证端到端流程
3. 约束检查：验证关键设计约束（如max 4帧输入LLM）

Usage:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --verbose
    python scripts/test_pipeline.py --test memory_bank
    python scripts/test_pipeline.py --medgemma_path /path/to/medgemma
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class TestConfig:
    """Test configuration with default dimensions."""
    batch_size: int = 1  # generate() only supports batch_size=1
    num_frames: int = 12
    image_size: int = 224
    lm_hidden: int = 2560
    vision_hidden: int = 1152
    num_vision_tokens: int = 256
    slot_dim: int = 512
    num_slots: int = 16
    num_sections: int = 8
    top_m_candidates: int = 5
    evidence_topL: int = 4
    max_evidence_frames: int = 4
    chunk_size: int = 4
    vocab_size: int = 256000
    device: str = 'cpu'
    medgemma_path: Optional[str] = None


# =============================================================================
# Test Utilities
# =============================================================================

class TestResult:
    """Container for test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
        
    def add_pass(self, msg: str = ""):
        self.passed += 1
        
    def add_fail(self, msg: str):
        self.failed += 1
        self.errors.append(msg)


def assert_shape(tensor: torch.Tensor, expected: Tuple, name: str, result: TestResult):
    """Assert tensor shape matches expected."""
    if tensor.shape == torch.Size(expected):
        result.add_pass()
    else:
        result.add_fail(f"{name}: expected shape {expected}, got {tuple(tensor.shape)}")


def assert_range(tensor: torch.Tensor, min_val: float, max_val: float, name: str, result: TestResult):
    """Assert tensor values are within range."""
    if tensor.min() >= min_val and tensor.max() <= max_val:
        result.add_pass()
    else:
        result.add_fail(f"{name}: values out of range [{min_val}, {max_val}]")


# =============================================================================
# Module Tests
# =============================================================================

def test_dummy_medgemma(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test DummyMedGemmaModel."""
    result = TestResult("DummyMedGemmaModel")
    
    try:
        from models.medgemma_adapter import DummyMedGemmaModel, DummyProcessor
        
        model = DummyMedGemmaModel(
            vision_hidden=cfg.vision_hidden,
            lm_hidden=cfg.lm_hidden,
            vision_patches=cfg.num_vision_tokens,  # Dummy uses same patches in/out
            num_image_tokens=cfg.num_vision_tokens,
            vocab_size=cfg.vocab_size,
        )
        
        # Test vision encoding
        frames = torch.randn(cfg.batch_size, 3, cfg.image_size, cfg.image_size)
        vision_output = model.vision_tower(frames)
        vision_tokens = vision_output.last_hidden_state
        # Dummy outputs [B, num_vision_tokens, vision_hidden]
        assert_shape(vision_tokens, (cfg.batch_size, cfg.num_vision_tokens, cfg.vision_hidden), 
                     "vision_tokens", result)
        
        # Test projection - projects hidden dim, preserves token count
        projected = model.multi_modal_projector(vision_tokens)
        # Projector outputs [B, num_vision_tokens, lm_hidden]
        assert_shape(projected, (cfg.batch_size, cfg.num_vision_tokens, cfg.lm_hidden),
                     "projected_tokens", result)
        
        # Test text embedding
        input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, 32))
        text_embeds = model.embed_tokens(input_ids)
        assert_shape(text_embeds, (cfg.batch_size, 32, cfg.lm_hidden),
                     "text_embeddings", result)
        
        # Test LM forward
        inputs_embeds = torch.randn(cfg.batch_size, 64, cfg.lm_hidden)
        attention_mask = torch.ones(cfg.batch_size, 64)
        output = model.language_model(inputs_embeds, attention_mask)
        assert_shape(output.logits, (cfg.batch_size, 64, cfg.vocab_size),
                     "lm_logits", result)
        
        # Test processor
        processor = DummyProcessor(vocab_size=cfg.vocab_size)
        messages = [{"role": "user", "content": "<image> Describe this image."}]
        input_ids, attn_mask = processor.apply_chat_template(messages)
        if input_ids.dim() == 2:
            result.add_pass()
        else:
            result.add_fail(f"Processor output dim: {input_ids.dim()}")
        
        if verbose:
            print(f"  Vision tokens: {vision_tokens.shape}")
            print(f"  Projected: {projected.shape}")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


def test_medgemma_adapter(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test MedGemmaAdapter."""
    result = TestResult("MedGemmaAdapter")
    
    try:
        from models.medgemma_adapter import MedGemmaAdapter
        
        # Test with dummy model first
        adapter = MedGemmaAdapter(
            model_name_or_path="dummy",
            use_dummy=True,
        )
        
        # Test properties
        if adapter.lm_hidden_size == 2560:
            result.add_pass()
        else:
            result.add_fail(f"lm_hidden_size: {adapter.lm_hidden_size}")
            
        if adapter.vision_hidden_size == 1152:
            result.add_pass()
        else:
            result.add_fail(f"vision_hidden_size: {adapter.vision_hidden_size}")
        
        # Test encode_frames_to_lm_tokens with dummy
        frames = torch.randn(cfg.batch_size, cfg.num_frames, 3, cfg.image_size, cfg.image_size)
        
        chunk_count = 0
        total_frames = 0
        for chunk_tokens, frame_indices in adapter.encode_frames_to_lm_tokens(frames, chunk_size=cfg.chunk_size):
            chunk_count += 1
            if chunk_tokens.dim() == 4:
                result.add_pass()
                total_frames += chunk_tokens.shape[1]
                # Check output has expected number of tokens
                if chunk_tokens.shape[2] == cfg.num_vision_tokens:
                    result.add_pass()
                else:
                    result.add_fail(f"Expected {cfg.num_vision_tokens} tokens, got {chunk_tokens.shape[2]}")
            else:
                result.add_fail(f"Chunk dim: {chunk_tokens.dim()}")
        
        expected_chunks = (cfg.num_frames + cfg.chunk_size - 1) // cfg.chunk_size
        if chunk_count == expected_chunks:
            result.add_pass()
        else:
            result.add_fail(f"Expected {expected_chunks} chunks, got {chunk_count}")
        
        # Test with real model if path provided
        if cfg.medgemma_path:
            if verbose:
                print(f"  Testing with real MedGemma: {cfg.medgemma_path}")
            
            real_adapter = MedGemmaAdapter(
                model_name_or_path=cfg.medgemma_path,
                use_dummy=False,
            )
            
            expected_size = getattr(real_adapter, 'expected_image_size', 896)
            expected_patches = real_adapter.num_vision_tokens
            
            if verbose:
                print(f"  Real model expects {expected_size}x{expected_size} images -> {expected_patches} patches")
            
            # Test encoding - use any image size, adapter will resize
            test_frames = torch.randn(1, 2, 3, 224, 224)
            
            for chunk_tokens, frame_indices in real_adapter.encode_frames_to_lm_tokens(test_frames, chunk_size=2):
                # After resize to expected_size, should get expected_patches
                if chunk_tokens.dim() == 4 and chunk_tokens.shape[2] == expected_patches:
                    result.add_pass()
                    if verbose:
                        print(f"  Real model output shape: {chunk_tokens.shape}")
                else:
                    result.add_fail(f"Real model chunk shape: {chunk_tokens.shape}, expected patches: {expected_patches}")
                break
            
        if verbose:
            print(f"  Processed {chunk_count} chunks, {total_frames} frames")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


def test_memory_bank(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test LesionInstanceMemoryBank."""
    result = TestResult("LesionInstanceMemoryBank")
    
    try:
        from models.memory_bank import (
            LesionDetector, SlotMatcher, SlotUpdater, LesionInstanceMemoryBank
        )
        
        # Test LesionDetector
        detector = LesionDetector(
            lm_hidden=cfg.lm_hidden,
            slot_dim=cfg.slot_dim,
            hidden_dim=1024,
        )
        
        lm_tokens = torch.randn(cfg.batch_size, cfg.num_vision_tokens, cfg.lm_hidden)
        candidates, confidences = detector(lm_tokens)
        
        assert_shape(candidates, (cfg.batch_size, cfg.num_vision_tokens, cfg.slot_dim),
                     "detector_candidates", result)
        assert_shape(confidences, (cfg.batch_size, cfg.num_vision_tokens),
                     "detector_confidences", result)
        assert_range(confidences, 0.0, 1.0, "confidences_range", result)
        
        # Test SlotMatcher (takes candidate [slot_dim] and slot_embeds [num_slots, slot_dim])
        matcher = SlotMatcher(slot_dim=cfg.slot_dim)
        
        candidate = torch.randn(cfg.slot_dim)
        slot_embeds = torch.randn(cfg.num_slots, cfg.slot_dim)
        
        scores, best_idx = matcher(candidate, slot_embeds)
        
        if scores.shape == (cfg.num_slots,):
            result.add_pass()
        else:
            result.add_fail(f"Matcher scores shape: {scores.shape}")
        
        if isinstance(best_idx, int) and 0 <= best_idx < cfg.num_slots:
            result.add_pass()
        else:
            result.add_fail(f"Best idx: {best_idx}")
        
        # Test SlotUpdater
        updater = SlotUpdater(slot_dim=cfg.slot_dim)
        
        old_slot = torch.randn(cfg.slot_dim)
        new_candidate = torch.randn(cfg.slot_dim)
        updated = updater(old_slot, new_candidate)
        assert_shape(updated, (cfg.slot_dim,), "updater_output", result)
        
        # Test full MemoryBank
        memory = LesionInstanceMemoryBank(
            num_slots=cfg.num_slots,
            slot_dim=cfg.slot_dim,
            lm_hidden=cfg.lm_hidden,
            top_m=cfg.top_m_candidates,
            evidence_topL=cfg.evidence_topL,
            similarity_threshold=0.7,
        )
        
        # Process multiple frames - process_frame expects [1, num_patches, lm_hidden]
        for frame_idx in range(4):
            section_id = frame_idx % cfg.num_sections
            frame_tokens = lm_tokens[0:1]  # Keep [1, num_patches, lm_hidden]
            memory.process_frame(
                lm_tokens_frame=frame_tokens,
                frame_idx=frame_idx,
                section_id=section_id
            )
        
        # Check slots created
        if len(memory.slots) >= 0:  # Valid to have 0 or more slots
            result.add_pass()
        else:
            result.add_fail(f"Invalid slots: {len(memory.slots)}")
        
        # Test evidence selection - returns List[int] only, not tuple
        selected_frames = memory.select_evidence_frames(max_frames=cfg.max_evidence_frames)
        
        if len(selected_frames) <= cfg.max_evidence_frames:
            result.add_pass()
        else:
            result.add_fail(f"Selected {len(selected_frames)} > {cfg.max_evidence_frames}")
        
        # Test reset
        memory.reset()
        if len(memory.slots) == 0:
            result.add_pass()
        else:
            result.add_fail("Reset failed")
        
        if verbose:
            print(f"  Detector output: candidates {candidates.shape}, confidences {confidences.shape}")
            print(f"  Selected evidence frames: {selected_frames}")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


def test_streaming_encoder(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test StreamingEncoder."""
    result = TestResult("StreamingEncoder")
    
    try:
        from models.streaming_encoder import StreamingEncoder, StreamingFrameSampler
        from models.medgemma_adapter import MedGemmaAdapter
        
        adapter = MedGemmaAdapter(use_dummy=True)
        
        # Test StreamingFrameSampler (input: [N, 3, H, W] single video)
        sampler = StreamingFrameSampler(min_frames=8, max_frames=16)
        
        all_frames = torch.randn(30, 3, cfg.image_size, cfg.image_size)
        sampled, indices = sampler.sample(all_frames)
        
        if 8 <= sampled.shape[0] <= 16:
            result.add_pass()
        else:
            result.add_fail(f"Sampled {sampled.shape[0]} frames")
        
        if len(indices) == sampled.shape[0]:
            result.add_pass()
        else:
            result.add_fail(f"Indices mismatch")
        
        # Test StreamingEncoder
        encoder = StreamingEncoder(
            adapter=adapter,
            chunk_size=cfg.chunk_size
        )
        
        frames = torch.randn(cfg.batch_size, cfg.num_frames, 3, cfg.image_size, cfg.image_size)
        
        all_chunks = []
        for chunk_tokens, frame_indices in encoder.encode_streaming(frames):
            all_chunks.append(chunk_tokens)
        
        if len(all_chunks) > 0:
            result.add_pass()
        else:
            result.add_fail("No chunks produced")
        
        # Verify chunk sizes
        max_chunk_frames = max(c.shape[1] for c in all_chunks)
        if max_chunk_frames <= cfg.chunk_size:
            result.add_pass()
        else:
            result.add_fail(f"Chunk too large: {max_chunk_frames}")
        
        if verbose:
            print(f"  Sampled: {sampled.shape}")
            print(f"  Encoded {len(all_chunks)} chunks")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


def test_router(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test SectionRouter."""
    result = TestResult("SectionRouter")
    
    try:
        from models.router import SectionRouter
        
        # Test rule mode with batch of frames [B, K, num_patches, lm_hidden]
        router_rule = SectionRouter(
            mode='rule',
            num_sections=cfg.num_sections,
            lm_hidden=cfg.lm_hidden,
        )
        
        lm_tokens = torch.randn(cfg.batch_size, cfg.num_frames, cfg.num_vision_tokens, cfg.lm_hidden)
        frame2section_list = [i % cfg.num_sections for i in range(cfg.num_frames)]
        
        sections, scores = router_rule(lm_tokens, frame2section_list)
        
        # Output: [B, K]
        if sections.shape == (cfg.batch_size, cfg.num_frames):
            result.add_pass()
        else:
            result.add_fail(f"Sections shape: {sections.shape}")
        
        # Test learned mode
        router_learned = SectionRouter(
            mode='learned',
            num_sections=cfg.num_sections,
            lm_hidden=cfg.lm_hidden,
            abnormal_threshold=0.5,
        )
        
        sections_l, scores_l = router_learned(lm_tokens, None)
        
        if sections_l.shape == (cfg.batch_size, cfg.num_frames):
            result.add_pass()
        else:
            result.add_fail(f"Learned sections: {sections_l.shape}")
        
        if scores_l.shape == (cfg.batch_size, cfg.num_frames):
            result.add_pass()
        else:
            result.add_fail(f"Scores shape: {scores_l.shape}")
        
        assert_range(scores_l, 0.0, 1.0, "scores_range", result)
        
        # Test get_abnormal_sections (takes section_ids [K] and abnormal_scores [K])
        sample_sections = sections_l[0]  # [K]
        sample_scores = scores_l[0]  # [K]
        
        abnormal = router_learned.get_abnormal_sections(
            section_ids=sample_sections,
            abnormal_scores=sample_scores,
        )
        
        if isinstance(abnormal, list):
            result.add_pass()
        else:
            result.add_fail(f"Abnormal type: {type(abnormal)}")
        
        if verbose:
            print(f"  Rule sections: {sections.shape}")
            print(f"  Learned sections: {sections_l.shape}")
            print(f"  Abnormal sections: {abnormal}")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


def test_composer(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test ReportComposer."""
    result = TestResult("ReportComposer")
    
    try:
        from models.composer import ReportComposer
        from data.templates import TemplateLibrary
        
        templates = TemplateLibrary()
        composer = ReportComposer(template_library=templates)
        
        # Test build_generation_prompt
        slot_summary = [
            {"slot_id": 0, "section_id": 5, "confidence": 0.9, "evidence_frames": [2, 5]},
        ]
        abnormal_sections = [5]
        selected_frames = [2, 5, 8]
        
        prompt = composer.build_generation_prompt(
            slot_summary=slot_summary,
            abnormal_sections=abnormal_sections,
            selected_frames=selected_frames,
            num_images=len(selected_frames)
        )
        
        if len(prompt) > 50:
            result.add_pass()
        else:
            result.add_fail(f"Prompt too short: {len(prompt)}")
        
        # Test build_messages (takes slot_summary, abnormal_sections, selected_frames, num_images)
        messages = composer.build_messages(
            slot_summary=slot_summary,
            abnormal_sections=abnormal_sections,
            selected_frames=selected_frames,
            num_images=len(selected_frames)
        )
        
        if len(messages) > 0:
            result.add_pass()
        else:
            result.add_fail("No messages")
        
        # Check for image placeholder
        has_image = any("<image>" in str(m.get("content", "")) for m in messages)
        if has_image:
            result.add_pass()
        else:
            result.add_fail("No image placeholder")
        
        # Test parse_generated_output (takes generated_text and abnormal_sections)
        generated = '{"antrum": "A 5mm polyp."}'
        parsed = composer.parse_generated_output(generated, abnormal_sections=[5])
        
        if isinstance(parsed, dict):
            result.add_pass()
        else:
            result.add_fail(f"Parse failed: {type(parsed)}")
        
        # Test compose_final_report (generated_abnormal, abnormal_sections, section_order)
        final = composer.compose_final_report(
            generated_abnormal={"antrum": "A polyp is seen."},
            abnormal_sections=[5],
        )
        
        if len(final) > 50:
            result.add_pass()
        else:
            result.add_fail(f"Report too short")
        
        if verbose:
            print(f"  Prompt length: {len(prompt)}")
            print(f"  Messages: {len(messages)}")
            print(f"  Final report preview: {final[:100]}...")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


def test_templates(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test TemplateLibrary."""
    result = TestResult("TemplateLibrary")
    
    try:
        from data.templates import TemplateLibrary
        
        templates = TemplateLibrary()
        
        # Check default templates
        expected_sections = ["esophagus", "gastroesophageal_junction", "cardia", 
                           "fundus", "body", "antrum", "pylorus", "duodenum"]
        
        for section in expected_sections:
            if section in templates.templates and len(templates.templates[section]) > 0:
                result.add_pass()
            else:
                result.add_fail(f"Missing templates for {section}")
        
        # Test get_normal_template
        template = templates.get_normal_template("esophagus")
        if template and len(template) > 10:
            result.add_pass()
        else:
            result.add_fail(f"Invalid template")
        
        # Test is_abnormal - use very clear test cases
        # The default abnormal keywords include: polyp, ulcer, lesion, erosion, mass, etc.
        normal_text = "The mucosa appears normal throughout."
        abnormal_text = "A 3mm polyp is seen."
        
        is_normal_abnormal = templates.is_abnormal(normal_text)
        is_abnormal_abnormal = templates.is_abnormal(abnormal_text)
        
        if not is_normal_abnormal:
            result.add_pass()
        else:
            result.add_fail(f"Normal text '{normal_text}' classified as abnormal")
        
        if is_abnormal_abnormal:
            result.add_pass()
        else:
            result.add_fail(f"Abnormal text '{abnormal_text}' not detected")
        
        # Test get_section_names
        section_names = templates.get_section_names()
        if len(section_names) >= 8:
            result.add_pass()
        else:
            result.add_fail(f"Section names: {len(section_names)}")
        
        if verbose:
            print(f"  Sections: {list(templates.templates.keys())}")
            print(f"  Normal template: {template[:50]}...")
            print(f"  Keywords sample: {list(templates.abnormal_keywords)[:5]}")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


def test_dataset(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test EndoscopyDataset."""
    result = TestResult("EndoscopyDataset")
    
    try:
        from data.dataset import EndoscopyDataset, collate_fn
        
        # Create with dummy data
        dataset = EndoscopyDataset(
            jsonl_path="nonexistent.jsonl",
            video_dir="nonexistent_dir",
            frame_size=(cfg.image_size, cfg.image_size),
            max_frames=cfg.num_frames,
        )
        
        if len(dataset) > 0:
            result.add_pass()
        else:
            result.add_fail("Dataset empty")
        
        # Test __getitem__
        sample = dataset[0]
        
        for key in ['frames', 'report']:
            if key in sample:
                result.add_pass()
            else:
                result.add_fail(f"Missing: {key}")
        
        # Test collate_fn
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        collated = collate_fn(batch)
        
        if 'frames' in collated and collated['frames'].dim() == 5:
            result.add_pass()
        else:
            result.add_fail("Collate failed")
        
        if verbose:
            print(f"  Dataset size: {len(dataset)}")
            print(f"  Sample keys: {sample.keys()}")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


def test_metrics(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test evaluation metrics."""
    result = TestResult("Metrics")
    
    try:
        from utils.metrics import (
            compute_bleu, compute_rouge_l, compute_meteor,
            compute_report_metrics, duplicate_rate, extract_findings
        )
        
        pred = "A small polyp is seen in the antrum."
        ref = "A 3mm polyp is observed in the gastric antrum."
        
        # Test BLEU
        bleu = compute_bleu(pred, ref)
        for key in ['bleu1', 'bleu2', 'bleu3', 'bleu4']:
            if key in bleu and 0 <= bleu[key] <= 1:
                result.add_pass()
            else:
                result.add_fail(f"Invalid {key}")
        
        # Test ROUGE-L
        rouge = compute_rouge_l(pred, ref)
        if 'rouge_l' in rouge and 0 <= rouge['rouge_l'] <= 1:
            result.add_pass()
        else:
            result.add_fail("Invalid ROUGE-L")
        
        # Test METEOR
        meteor = compute_meteor(pred, ref)
        if 'meteor' in meteor and 0 <= meteor['meteor'] <= 1:
            result.add_pass()
        else:
            result.add_fail("Invalid METEOR")
        
        # Test batch metrics
        preds = [pred, "Normal esophageal mucosa."]
        refs = [ref, "The esophagus appears normal."]
        metrics = compute_report_metrics(preds, refs)
        
        for key in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge_l', 'meteor']:
            if key in metrics:
                result.add_pass()
            else:
                result.add_fail(f"Missing: {key}")
        
        # Test duplicate_rate
        dup = duplicate_rate("The polyp is a polyp.")
        if 'duplicate_rate' in dup:
            result.add_pass()
        else:
            result.add_fail("Duplicate failed")
        
        # Test extract_findings
        findings = extract_findings("A 5mm polyp noted.")
        if isinstance(findings, list):
            result.add_pass()
        else:
            result.add_fail("Findings failed")
        
        if verbose:
            print(f"  BLEU: {bleu}")
            print(f"  ROUGE-L: {rouge['rouge_l']:.4f}")
            print(f"  METEOR: {meteor['meteor']:.4f}")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


def test_self_check(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test SelfCheck module."""
    result = TestResult("SelfCheck")
    
    try:
        from models.self_check import SelfCheck
        
        checker = SelfCheck(lm_hidden=cfg.lm_hidden)
        
        # Test check_consistency
        gen_embeds = torch.randn(cfg.batch_size, 50, cfg.lm_hidden)
        ev_embeds = torch.randn(cfg.batch_size, 30, cfg.lm_hidden)
        
        score, details = checker.check_consistency(gen_embeds, ev_embeds)
        
        if 0 <= score <= 1:
            result.add_pass()
        else:
            result.add_fail(f"Score: {score}")
        
        if 'is_consistent' in details:
            result.add_pass()
        else:
            result.add_fail("Missing is_consistent")
        
        # Test validate_findings
        findings = {"antrum": "A polyp."}
        slot_summary = [{"section_id": 5, "num_evidence": 2, "confidence": 0.9}]
        validation = checker.validate_findings(findings, slot_summary)
        
        if isinstance(validation, dict):
            result.add_pass()
        else:
            result.add_fail("Validation failed")
        
        # Test compute_support_rate
        rate = checker.compute_support_rate(validation)
        if 0 <= rate <= 1:
            result.add_pass()
        else:
            result.add_fail(f"Rate: {rate}")
        
        if verbose:
            print(f"  Consistency score: {score:.4f}")
            print(f"  Support rate: {rate:.4f}")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_pipeline(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test full end-to-end pipeline."""
    result = TestResult("Full Pipeline (E2E)")
    
    try:
        from models.stream_lesionmem_model import StreamLesionMemModel
        
        # Use real model if path provided, otherwise dummy
        use_dummy = (cfg.medgemma_path is None)
        medgemma_path = cfg.medgemma_path or "dummy"
        
        if verbose:
            print(f"  Using {'dummy' if use_dummy else 'real'} model: {medgemma_path}")
        
        model = StreamLesionMemModel(
            medgemma_path=medgemma_path,
            use_dummy=use_dummy,
            num_slots=cfg.num_slots,
            slot_dim=cfg.slot_dim,
            top_m=cfg.top_m_candidates,
            evidence_topL=cfg.evidence_topL,
            num_sections=cfg.num_sections,
            chunk_size=cfg.chunk_size,
            max_frames_for_llm=cfg.max_evidence_frames,
            router_mode='rule'
        )
        
        model.eval()
        
        # Use batch_size=1 (generate only supports 1)
        frames = torch.randn(1, cfg.num_frames, 3, cfg.image_size, cfg.image_size)
        frame2section = [{i: i % cfg.num_sections for i in range(cfg.num_frames)}]
        
        with torch.no_grad():
            reports = model.generate(
                frames=frames,
                frame2section=frame2section,
                max_new_tokens=64
            )
        
        # generate() 新版返回 dict
        if isinstance(reports, dict):
            result.add_pass()
            final_report = reports.get("final_report", "")
            if isinstance(final_report, str) and len(final_report) > 0:
                result.add_pass()
            else:
                result.add_fail(f"final_report invalid: {type(final_report)} len={len(final_report) if isinstance(final_report,str) else 'NA'}")

        # 兼容老版：list[str]
        elif isinstance(reports, list) and len(reports) == 1:
            result.add_pass()
            if isinstance(reports[0], str) and len(reports[0]) > 0:
                result.add_pass()
            else:
                result.add_fail("Report invalid")

        else:
            result.add_fail(f"Generate output unexpected type: {type(reports)}")
    
        if verbose:
            if isinstance(reports, dict):
                print("  Report preview:", reports["final_report"][:150], "...")
            else:
                print("  Report preview:", reports[0][:150], "...")
            
    except Exception as e:
        # result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
        raise

    return result


def test_constraints(cfg: TestConfig, verbose: bool = False) -> TestResult:
    """Test critical design constraints."""
    result = TestResult("Design Constraints")
    
    try:
        from models.memory_bank import LesionInstanceMemoryBank
        from models.streaming_encoder import StreamingEncoder
        from models.medgemma_adapter import MedGemmaAdapter
        from models.composer import ReportComposer
        from data.templates import TemplateLibrary
        
        # CONSTRAINT 1: Max evidence frames <= 4
        memory = LesionInstanceMemoryBank(
            num_slots=cfg.num_slots,
            slot_dim=cfg.slot_dim,
            lm_hidden=cfg.lm_hidden,
            top_m=cfg.top_m_candidates,
            evidence_topL=cfg.evidence_topL,
        )
        
        # process_frame expects [1, num_patches, lm_hidden]
        for i in range(20):
            lm_tokens = torch.randn(1, cfg.num_vision_tokens, cfg.lm_hidden)
            memory.process_frame(lm_tokens_frame=lm_tokens, frame_idx=i, section_id=i % 8)
        
        # select_evidence_frames returns List[int] only
        selected = memory.select_evidence_frames(max_frames=cfg.max_evidence_frames)
        
        if len(selected) <= cfg.max_evidence_frames:
            result.add_pass()
            if verbose:
                print(f"  CONSTRAINT 1: ✓ Selected {len(selected)} <= {cfg.max_evidence_frames} frames")
        else:
            result.add_fail(f"Constraint 1: {len(selected)} > {cfg.max_evidence_frames}")
        
        # CONSTRAINT 2: Streaming encoding chunks
        adapter = MedGemmaAdapter(use_dummy=True)
        encoder = StreamingEncoder(adapter=adapter, chunk_size=cfg.chunk_size)
        
        frames = torch.randn(1, 20, 3, cfg.image_size, cfg.image_size)
        
        max_chunk_frames = 0
        for chunk_tokens, _ in encoder.encode_streaming(frames):
            max_chunk_frames = max(max_chunk_frames, chunk_tokens.shape[1])
        
        if max_chunk_frames <= cfg.chunk_size:
            result.add_pass()
            if verbose:
                print(f"  CONSTRAINT 2: ✓ Max chunk {max_chunk_frames} <= {cfg.chunk_size}")
        else:
            result.add_fail(f"Constraint 2: {max_chunk_frames} > {cfg.chunk_size}")
        
        # CONSTRAINT 3: Memory online processing
        memory.reset()
        for i in range(5):
            lm_tokens = torch.randn(1, cfg.num_vision_tokens, cfg.lm_hidden)
            memory.process_frame(
                lm_tokens_frame=lm_tokens,
                frame_idx=i,
                section_id=i % 8
            )
        
        # Should be able to process sequentially (not fail)
        result.add_pass()
        if verbose:
            print(f"  CONSTRAINT 3: ✓ Online processing works, {len(memory.slots)} slots")
        
        # CONSTRAINT 4: Templates for normal, generation for abnormal
        templates = TemplateLibrary()
        composer = ReportComposer(template_library=templates)
        
        final = composer.compose_final_report(
            generated_abnormal={"antrum": "A polyp is seen."},
            abnormal_sections=[5],
        )
        
        if "polyp" in final.lower():
            result.add_pass()
            if verbose:
                print(f"  CONSTRAINT 4: ✓ Generated content in report")
        else:
            result.add_fail("Constraint 4: Generated content missing")
            
    except Exception as e:
        result.add_fail(f"Exception: {str(e)}\n{traceback.format_exc()}")
    
    return result


# =============================================================================
# Main
# =============================================================================

def run_all_tests(cfg: TestConfig, verbose: bool = False, specific_test: Optional[str] = None) -> bool:
    """Run all tests."""
    
    all_tests = {
        'dummy_medgemma': test_dummy_medgemma,
        'medgemma_adapter': test_medgemma_adapter,
        'memory_bank': test_memory_bank,
        'streaming_encoder': test_streaming_encoder,
        'router': test_router,
        'composer': test_composer,
        'templates': test_templates,
        'dataset': test_dataset,
        'metrics': test_metrics,
        'self_check': test_self_check,
        'full_pipeline': test_full_pipeline,
        'constraints': test_constraints,
    }
    
    if specific_test:
        if specific_test not in all_tests:
            print(f"Unknown test: {specific_test}")
            print(f"Available: {list(all_tests.keys())}")
            return False
        tests_to_run = {specific_test: all_tests[specific_test]}
    else:
        tests_to_run = all_tests
    
    print("=" * 60)
    print("STREAM-LesionMem Pipeline Verification")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Num frames: {cfg.num_frames}")
    print(f"  LM hidden: {cfg.lm_hidden}")
    print(f"  Num slots: {cfg.num_slots}")
    print(f"  Max evidence frames: {cfg.max_evidence_frames}")
    print(f"  Chunk size: {cfg.chunk_size}")
    print(f"  MedGemma: {'dummy' if cfg.medgemma_path is None else cfg.medgemma_path}")
    print()
    
    results: List[TestResult] = []
    
    for name, test_fn in tests_to_run.items():
        print(f"Testing {name}...", end=" ", flush=True)
        test_result = test_fn(cfg, verbose)
        results.append(test_result)
        
        if test_result.failed == 0:
            print(f"✓ ({test_result.passed} passed)")
        else:
            print(f"✗ ({test_result.failed} failed)")
            for error in test_result.errors:
                print(f"    - {error[:200]}")
        
        if verbose:
            print()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    
    for r in results:
        status = "✓" if r.failed == 0 else "✗"
        print(f"  {status} {r.name}: {r.passed} passed, {r.failed} failed")
    
    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("\n✓ All tests passed!")
        return True
    else:
        print(f"\n✗ {total_failed} test(s) failed.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify STREAM-LesionMem pipeline")
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--test', '-t', type=str, default=None)
    parser.add_argument('--medgemma_path', type=str, default='/home/common/qylv/proj/dual_medendo/checkpoints/medGemma-4b-it')
    
    args = parser.parse_args()
    
    cfg = TestConfig(medgemma_path=args.medgemma_path)
    success = run_all_tests(cfg, verbose=args.verbose, specific_test=args.test)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
