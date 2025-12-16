"""
STREAM-LesionMem Main Model.

Combines all components for streaming lesion memory-based report generation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

from .medgemma_adapter import MedGemmaAdapter
from .streaming_encoder import StreamingEncoder, StreamingFrameSampler
from .memory_bank import LesionInstanceMemoryBank
from .router import SectionRouter
from .composer import ReportComposer, ComposerOutput
from .self_check import SelfCheck
from data.templates import TemplateLibrary


class StreamLesionMemModel(nn.Module):
    """
    STREAM-LesionMem: Streaming Lesion Instance Memory for Report Generation.
    
    Architecture:
    1. StreamingFrameSampler: Sample K frames from video (8-16)
    2. StreamingEncoder: Encode frames in chunks via MedGemma vision
    3. SectionRouter: Route frames to anatomical sections
    4. LesionInstanceMemoryBank: Track lesion instances with evidence
    5. Evidence Selection: Select top-4 frames for LLM input
    6. ReportComposer: Build prompts and combine templates
    7. MedGemma LLM: Generate abnormal sections only
    8. SelfCheck: Optional consistency validation
    
    Key constraints:
    - LLM input: max 4 frames
    - Memory can process 8-16 frames via streaming
    - No retrieval - templates for normal, generation for abnormal
    """
    
    def __init__(
        self,
        # MedGemma settings
        medgemma_path: str = "google/medgemma-4b-it",
        torch_dtype: torch.dtype = torch.bfloat16,
        freeze_medgemma: bool = True,
        use_dummy: bool = False,
        
        # Memory settings
        num_slots: int = 16,
        slot_dim: int = 512,
        lm_hidden: int = 2560,
        similarity_threshold: float = 0.7,
        top_m: int = 5,
        evidence_topL: int = 4,
        
        # Router settings
        router_mode: str = "rule",
        num_sections: int = 8,
        abnormal_threshold: float = 0.5,
        
        # Streaming settings
        chunk_size: int = 4,
        sample_frames: int = 12,
        max_frames_for_llm: int = 4,
        
        # Template library
        templates_path: Optional[str] = None,
        
        # Self-check
        enable_self_check: bool = False,
    ):
        super().__init__()
        
        self.max_frames_for_llm = max_frames_for_llm
        self.chunk_size = chunk_size
        self.enable_self_check = enable_self_check
        
        # Initialize MedGemma adapter
        self.adapter = MedGemmaAdapter(
            model_name_or_path=medgemma_path,
            torch_dtype=torch_dtype,
            freeze=freeze_medgemma,
            use_dummy=use_dummy,
        )
        
        # Update lm_hidden from loaded model
        lm_hidden = self.adapter.lm_hidden_size
        
        # Initialize streaming components
        self.sampler = StreamingFrameSampler(
            min_frames=8,
            max_frames=16,
            target_frames=sample_frames,
        )
        
        self.encoder = StreamingEncoder(
            adapter=self.adapter,
            chunk_size=chunk_size,
            cache_selected=True,
        )
        
        # Initialize router
        self.router = SectionRouter(
            mode=router_mode,
            num_sections=num_sections,
            lm_hidden=lm_hidden,
            abnormal_threshold=abnormal_threshold,
        )
        
        # Initialize memory bank
        self.memory = LesionInstanceMemoryBank(
            num_slots=num_slots,
            slot_dim=slot_dim,
            lm_hidden=lm_hidden,
            similarity_threshold=similarity_threshold,
            top_m=top_m,
            evidence_topL=evidence_topL,
        )
        
        # Initialize composer
        self.template_library = TemplateLibrary(templates_path)
        self.composer = ReportComposer(
            template_library=self.template_library,
        )
        
        # Initialize self-check (optional)
        if enable_self_check:
            self.self_check = SelfCheck(lm_hidden=lm_hidden)
        else:
            self.self_check = None
        # Ensure external modules are on same device as adapter outputs
        device = self.adapter.device  # 或 self.encoder.adapter.device
        # self.memory.to(device=self.adapter.device, dtype=torch_dtype)
        # self.router.to(device=self.adapter.device, dtype=torch_dtype)

            
    def forward(
        self,
        frames: torch.Tensor,
        gt_report: Optional[str] = None,
        frame2section: Optional[List[int]] = None,
        gt_sections: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Training forward pass.
        
        Args:
            frames: [B, K, 3, H, W] input frames
            gt_report: Ground truth report (optional)
            frame2section: Per-frame section labels
            gt_sections: Ground truth section texts
        
        Returns:
            Dict with loss and intermediate outputs
        """
        B = frames.shape[0]
        assert B == 1, "Training currently supports batch size 1"
        
        device = frames.device
        self.memory.reset()
        self.encoder.clear_cache()
        
        # Sample frames
        sampled_frames, frame_indices = self.sampler.sample(frames.squeeze(0))
        sampled_frames = sampled_frames.unsqueeze(0).to(device)
        
        K = sampled_frames.shape[1]
        
        # Prepare frame2section
        if frame2section is None:
            frame2section = [i * self.router.num_sections // K for i in range(K)]
        if isinstance(frame2section, dict):
            frame2section = [frame2section.get(i, 0) for i in range(K)]
        elif isinstance(frame2section, list) and len(frame2section) > 0 and isinstance(frame2section[0], dict):
            mapping = frame2section[0]
            frame2section = [mapping.get(i, 0) for i in range(K)]
        
        # Determine which frames to cache (for evidence)
        # Initially cache all, then prune
        cache_indices = list(range(K))
        
        # Streaming encode and process
        all_tokens = []
        for lm_tokens, chunk_indices in self.encoder.encode_streaming(
            sampled_frames,
            cache_indices=cache_indices,
        ):
            # lm_tokens: [B, chunk_size, num_patches, lm_hidden]
            all_tokens.append(lm_tokens)
            
            # Route and update memory for each frame in chunk
            for i, global_idx in enumerate(chunk_indices):
                frame_tokens = lm_tokens[:, i]  # [B, num_patches, lm_hidden]
                section_id = frame2section[global_idx] if global_idx < len(frame2section) else 0
                
                # Process frame in memory
                self.memory.process_frame(
                    frame_tokens,
                    frame_idx=global_idx,
                    section_id=section_id,
                    original_tokens=frame_tokens,
                )
        
        # Concatenate all tokens
        all_lm_tokens = torch.cat(all_tokens, dim=1)  # [B, K, num_patches, lm_hidden]
        
        # Get router predictions
        section_ids, abnormal_scores = self.router(all_lm_tokens, frame2section)
        
        # Determine abnormal sections
        abnormal_sections = self.router.get_abnormal_sections(
            section_ids.squeeze(0),
            abnormal_scores.squeeze(0),
        )
        
        # If no abnormal detected via router, use template deviation
        if not abnormal_sections and gt_sections:
            for sec_name, sec_text in gt_sections.items():
                if self.template_library.is_abnormal(sec_text):
                    sec_id = self.composer._get_section_id(sec_name)
                    if sec_id >= 0:
                        abnormal_sections.append(sec_id)
        
        # Select evidence frames
        selected_frames = self.memory.select_evidence_frames(max_frames=self.max_frames_for_llm)
        
        if not selected_frames:
            # Fallback: use first few frames
            selected_frames = list(range(min(self.max_frames_for_llm, K)))
        
        # Get selected frame tokens
        selected_tokens = self.encoder.get_cached_tokens(selected_frames)
        if selected_tokens is None:
            selected_tokens = all_lm_tokens[:, selected_frames]
        
        # Build image embeds
        image_embeds = self.memory.get_combined_image_embeds(
            selected_tokens,
            include_memory=True,
        )
        
        # Build messages
        slot_summary = self.memory.get_slot_summary()
        messages = self.composer.build_messages(
            slot_summary=slot_summary,
            abnormal_sections=abnormal_sections,
            selected_frames=selected_frames,
            num_images=len(selected_frames),
        )
        
        # Get input_ids
        input_ids, attention_mask = self.adapter.build_inputs(messages)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Inject image embeds
        inputs_embeds = self.adapter.inject_image_embeds(
            input_ids,
            attention_mask,
            image_embeds,
        )
        
        # Compute loss if gt_report provided
        loss = None
        if gt_report is not None:
            # TODO: Proper label preparation
            # For now, use dummy loss
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return {
            "loss": loss,
            "selected_frames": selected_frames,
            "abnormal_sections": abnormal_sections,
            "slot_summary": slot_summary,
            "section_ids": section_ids,
            "abnormal_scores": abnormal_scores,
        }
    
    @torch.no_grad()
    def generate(
        self,
        frames: torch.Tensor,
        frame2section: Optional[List[int]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate report from video frames.
        
        Args:
            frames: [B, K, 3, H, W] or [K, 3, H, W] input frames
            frame2section: Per-frame section labels
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample
            **gen_kwargs: Additional generation kwargs
        
        Returns:
            Dict with generated report and debug info
        """
        # Ensure batch dimension
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        B = frames.shape[0]
        assert B == 1, "Generation supports batch size 1"
        
        device = frames.device
        self.memory.reset()
        self.encoder.clear_cache()
        
        # Step 1: Sample K frames (8-16)
        sampled_frames, frame_indices = self.sampler.sample(frames.squeeze(0))
        sampled_frames = sampled_frames.unsqueeze(0).to(device)
        
        K = sampled_frames.shape[1] if sampled_frames.dim() == 5 else sampled_frames.shape[0]

        # normalize ASAP (BEFORE any router call)
        if frame2section is None:
            frame2section = [i * self.router.num_sections // K for i in range(K)]
        if isinstance(frame2section, dict):
            frame2section = [int(frame2section.get(i, 0)) for i in range(K)]
        elif isinstance(frame2section, list) and len(frame2section) > 0 and isinstance(frame2section[0], dict):
            mapping = frame2section[0]
            frame2section = [int(mapping.get(i, 0)) for i in range(K)]
        else:
            frame2section = [int(x) for x in frame2section]
        
        # Step 2: Streaming encode and build memory
        all_tokens = []
        
        for lm_tokens, chunk_indices in self.encoder.encode_streaming(
            sampled_frames,
            cache_indices=list(range(K)),  # Cache all for now
        ):
                    # one-time dtype/device alignment (do this only once)
            if not hasattr(self, "_external_aligned") or not self._external_aligned:
                dev = lm_tokens.device
                dt  = lm_tokens.dtype  # likely torch.bfloat16

                self.memory.to(device=dev, dtype=dt)
                self.router.to(device=dev, dtype=dt)   # 如果 router 里有 Linear/Embedding

                self._external_aligned = True
            all_tokens.append(lm_tokens)
            
            # Route and process each frame
            for i, global_idx in enumerate(chunk_indices):
                assert isinstance(global_idx, int), f"global_idx not int: {type(global_idx)} {global_idx}"

                frame_tokens = lm_tokens[:, i]
                section_id = frame2section[global_idx] if global_idx < len(frame2section) else 0
                
                # Route
                _, abnormal_score = self.router._forward_single(
                    frame_tokens,
                    [section_id],
                )
                
                # Update memory
                self.memory.process_frame(
                    frame_tokens,
                    frame_idx=global_idx,
                    section_id=section_id,
                    original_tokens=frame_tokens,
                )


        all_lm_tokens = torch.cat(all_tokens, dim=1)
        
        # Step 3: Get router predictions for abnormal detection
        section_ids, abnormal_scores = self.router(all_lm_tokens, frame2section)
        
        abnormal_sections = self.router.get_abnormal_sections(
            section_ids.squeeze(0),
            abnormal_scores.squeeze(0),
        )
        
        # Step 4: Select evidence frames (max 4)
        selected_frames = self.memory.select_evidence_frames(max_frames=self.max_frames_for_llm)
        
        if not selected_frames:
            selected_frames = list(range(min(self.max_frames_for_llm, K)))
        
        # Step 5: Get selected frame tokens
        selected_tokens = self.encoder.get_cached_tokens(selected_frames)
        if selected_tokens is None:
            # Re-encode selected frames
            selected_tokens = self.encoder.encode_selected_frames(
                sampled_frames,
                selected_frames,
            )
        
        # Step 6: Build image embeds with memory
        image_embeds = self.memory.get_combined_image_embeds(
            selected_tokens,
            include_memory=True,
        )
        
        # Step 7: Build messages
        slot_summary = self.memory.get_slot_summary()
        messages = self.composer.build_messages(
            slot_summary=slot_summary,
            abnormal_sections=abnormal_sections,
            selected_frames=selected_frames,
            num_images=len(selected_frames),
        )
        
        # Step 8: Build inputs and inject
        input_ids, attention_mask = self.adapter.build_inputs(messages)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        inputs_embeds = self.adapter.inject_image_embeds(
            input_ids,
            attention_mask,
            image_embeds,
        )
        
        # Step 9: Generate
        generated_ids = self.adapter.generate_with_embeds(
            inputs_embeds,
            attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **gen_kwargs,
        )
        
        generated_text = self.adapter.decode(generated_ids)[0]
        
        # Step 10: Parse generated and compose final report
        parsed_abnormal = self.composer.parse_generated_output(
            generated_text,
            abnormal_sections,
        )
        
        final_report = self.composer.compose_final_report(
            parsed_abnormal,
            abnormal_sections,
        )
        
        # Step 11: Optional self-check
        validation = None
        if self.enable_self_check and self.self_check is not None:
            validation = self.self_check.validate_findings(
                parsed_abnormal,
                slot_summary,
            )
        
        return {
            "final_report": final_report,
            "generated_abnormal": parsed_abnormal,
            "raw_generated": generated_text,
            "selected_frames": selected_frames,
            "original_frame_indices": [frame_indices[i] for i in selected_frames],
            "abnormal_sections": abnormal_sections,
            "slot_summary": slot_summary,
            "validation": validation,
        }
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get parameters that should be trained."""
        params = []
        
        # Memory bank parameters
        params.extend(self.memory.parameters())
        
        # Router parameters (if learned mode)
        if hasattr(self.router, 'pool_proj'):
            params.extend(self.router.parameters())
        
        return params
    
    def save_pretrained(self, save_path: str) -> None:
        """Save trainable components."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save memory bank
        torch.save(
            self.memory.state_dict(),
            os.path.join(save_path, "memory_bank.pt"),
        )
        
        # Save router
        torch.save(
            self.router.state_dict(),
            os.path.join(save_path, "router.pt"),
        )
        
        # Save config
        import json
        config = {
            "num_slots": self.memory.num_slots,
            "slot_dim": self.memory.slot_dim,
            "lm_hidden": self.memory.lm_hidden,
            "router_mode": self.router.mode.value,
            "num_sections": self.router.num_sections,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    def load_pretrained(self, load_path: str) -> None:
        """Load trainable components."""
        import os
        
        # Load memory bank
        memory_path = os.path.join(load_path, "memory_bank.pt")
        if os.path.exists(memory_path):
            self.memory.load_state_dict(torch.load(memory_path))
        
        # Load router
        router_path = os.path.join(load_path, "router.pt")
        if os.path.exists(router_path):
            self.router.load_state_dict(torch.load(router_path))


def create_model(config: Dict) -> StreamLesionMemModel:
    """Create model from config dict."""
    return StreamLesionMemModel(
        medgemma_path=config.get("model", {}).get("medgemma_path", "google/medgemma-4b-it"),
        torch_dtype=getattr(torch, config.get("model", {}).get("torch_dtype", "bfloat16")),
        freeze_medgemma=config.get("model", {}).get("freeze_medgemma", True),
        use_dummy=config.get("model", {}).get("use_dummy", True),
        num_slots=config.get("memory", {}).get("num_slots", 16),
        slot_dim=config.get("memory", {}).get("slot_dim", 512),
        similarity_threshold=config.get("memory", {}).get("similarity_threshold", 0.7),
        top_m=config.get("memory", {}).get("top_m", 5),
        evidence_topL=config.get("memory", {}).get("evidence_topL", 4),
        router_mode=config.get("router", {}).get("mode", "rule"),
        num_sections=config.get("router", {}).get("num_sections", 8),
        abnormal_threshold=config.get("router", {}).get("abnormal_threshold", 0.5),
        chunk_size=config.get("streaming", {}).get("chunk_size", 4),
        sample_frames=config.get("streaming", {}).get("sample_frames", 12),
        max_frames_for_llm=config.get("inference", {}).get("max_frames", 4),
        templates_path=config.get("templates", {}).get("path", None),
        enable_self_check=config.get("self_check", {}).get("enabled", False),
    )
