"""
MedGemma Adapter for STREAM-LesionMem.

This module provides the interface to MedGemma following the dual_medendo style:
- Load model via AutoModelForImageTextToText with trust_remote_code
- Encode images via vision_tower + multi_modal_projector
- Inject image embeddings via input_ids mask replacement
- Forward/generate via language_model with inputs_embeds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Generator
from PIL import Image
import numpy as np

# Import transformers components
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. Using dummy model.")

def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: [B, L], 1=valid, 0=pad
    # 生成 0..L-1，但 pad 位置给 0（或任意值都行，只要不参与即可）
    pos = attention_mask.long().cumsum(-1) - 1
    pos = pos.clamp(min=0)
    return pos

class MedGemmaAdapter(nn.Module):
    """
    Adapter for MedGemma VLM following dual_medendo interface style.
    
    Key interfaces:
    - build_inputs: Construct input_ids/attention_mask from messages via chat template
    - encode_frames_to_lm_tokens: Chunk-based vision encoding via vision_tower + projector
    - inject_image_embeds: Replace <image> token spans with custom embeddings
    - forward_with_embeds / generate_with_embeds: LM forward/generate with inputs_embeds
    """
    
    def __init__(
        self,
        model_name_or_path: str = "google/medgemma-4b-it",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        freeze: bool = True,
        trust_remote_code: bool = True,
        use_dummy: bool = False,  # For testing without real model
    ):
        """
        Initialize MedGemma adapter.
        
        Args:
            model_name_or_path: HuggingFace model path
            torch_dtype: Model dtype (bfloat16 recommended)
            device_map: Device mapping strategy
            freeze: Whether to freeze model parameters
            trust_remote_code: Trust remote code for model loading
            use_dummy: Use dummy model for testing
        """
        super().__init__()
        
        self.model_name = model_name_or_path
        self.torch_dtype = torch_dtype
        self.freeze = freeze
        self.use_dummy = use_dummy
        
        # Model dimensions (will be updated after loading)
        self.vision_hidden_size: int = 1152  # SigLIP hidden size
        self.lm_hidden_size: int = 2560  # Gemma hidden size
        self.num_vision_tokens: int = 256  # Patches per image
        self.image_token_index: int = 255999  # Default, will be updated
        
        if use_dummy or not HAS_TRANSFORMERS:
            self._init_dummy_model()
        else:
            self._init_real_model(model_name_or_path, torch_dtype, device_map, trust_remote_code)
        
        if freeze:
            self._freeze_model()
    
    def _init_real_model(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype,
        device_map: str,
        trust_remote_code: bool,
    ) -> None:
        """Initialize real MedGemma model."""
        print(f"Loading MedGemma from {model_name_or_path}...")
        
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
            )
            
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            
            # Update dimensions from loaded model
            if hasattr(self.model.config, "vision_config"):
                self.vision_hidden_size = self.model.config.vision_config.hidden_size
                # Get image size from vision config
                if hasattr(self.model.config.vision_config, "image_size"):
                    self.expected_image_size = self.model.config.vision_config.image_size
                else:
                    self.expected_image_size = 896  # Default for MedGemma
                # Calculate num_patches from image_size and patch_size
                patch_size = getattr(self.model.config.vision_config, "patch_size", 14)
                # self.num_vision_tokens = (self.expected_image_size // patch_size) ** 2
            else:
                self.expected_image_size = 896
            if hasattr(self.model.config, "text_config"):
                self.lm_hidden_size = self.model.config.text_config.hidden_size
            if hasattr(self.model.config, "image_token_index"):
                self.image_token_index = self.model.config.image_token_index
            
            self._is_dummy = False
            print(f"Model loaded. Vision hidden: {self.vision_hidden_size}, LM hidden: {self.lm_hidden_size}")
            print(f"Expected image size: {self.expected_image_size}x{self.expected_image_size}, Patches: {self.num_vision_tokens}")
            # --- Probe actual projected token count (what LM really receives) ---
            with torch.no_grad():
                device = next(self.model.vision_tower.parameters()).device
                dummy = torch.zeros(1, 3, self.expected_image_size, self.expected_image_size, device=device)
                if self.torch_dtype is not None:
                    dummy = dummy.to(dtype=self.torch_dtype)
                vout = self.model.vision_tower(dummy, output_hidden_states=False)
                vtoks = vout.last_hidden_state              # [1, vision_patches, vision_hidden]
                ptoks = self.model.multi_modal_projector(vtoks)  # [1, projected_tokens, lm_hidden]
                self.num_vision_tokens = int(ptoks.shape[1])
            
        except Exception as e:
            print(f"Failed to load real model: {e}. Using dummy model.")
            self._init_dummy_model()

    def _normalize_messages_for_chat_template(self, messages: List[Dict]) -> List[Dict]:
        """
        HF processor.apply_chat_template for multimodal expects:
        message["content"] = [{"type":"text","text":...}, {"type":"image"}, ...]
        Your pipeline currently uses string content with '<image>' placeholders.
        Convert automatically.
        """
        normed = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            # already structured
            if isinstance(content, list):
                normed.append({"role": role, "content": content})
                continue

            # string -> structured segments
            if not isinstance(content, str):
                content = str(content)

            segments = []
            if "<image>" in content:
                parts = content.split("<image>")
                for i, part in enumerate(parts):
                    if part.strip():
                        segments.append({"type": "text", "text": part})
                    if i != len(parts) - 1:
                        segments.append({"type": "image"})
            else:
                segments.append({"type": "text", "text": content})

            normed.append({"role": role, "content": segments})
        return normed

    
    def _init_dummy_model(self) -> None:
        """Initialize dummy model for testing."""
        print("Initializing dummy MedGemma model for testing...")
        
        self._is_dummy = True
        self.expected_image_size = 224  # Dummy model uses smaller images
        self.model = DummyMedGemmaModel(
            vision_hidden=self.vision_hidden_size,
            lm_hidden=self.lm_hidden_size,
            vision_patches=self.num_vision_tokens,  # 256 for dummy
            num_image_tokens=self.num_vision_tokens,
        )
        self.processor = DummyProcessor()
        
    def _freeze_model(self) -> None:
        """Freeze all model parameters."""
        if hasattr(self, 'model'):
            for param in self.model.parameters():
                param.requires_grad = False
    
    def build_inputs(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build input_ids and attention_mask from chat messages.
        
        Uses processor.apply_chat_template for Gemma3 chat format.
        
        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            add_generation_prompt: Whether to add generation prompt
        
        Returns:
            input_ids: [1, seq_len]
            attention_mask: [1, seq_len]
        """
        messages = self._normalize_messages_for_chat_template(messages)
        if self._is_dummy:
            return self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=add_generation_prompt,
            )
        
        # Use real processor
        result = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
        )
        
        if isinstance(result, dict):
            input_ids = result.get("input_ids", result.get("tokens"))
            attention_mask = result.get("attention_mask", torch.ones_like(input_ids))
        else:
            input_ids = result
            attention_mask = torch.ones_like(input_ids)

        dev = self.device
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        return input_ids, attention_mask
    
    @torch.no_grad()
    def encode_frames_to_lm_tokens(
        self,
        frames: torch.Tensor,
        chunk_size: int = 4,
    ) -> Generator[Tuple[torch.Tensor, List[int]], None, None]:
        """
        Encode frames to LM-compatible tokens via vision_tower + projector.
        
        Processes frames in chunks to support streaming without OOM.
        
        Args:
            frames: [B, K, 3, H, W] or [K, 3, H, W] - input frames in [0, 1]
            chunk_size: Number of frames to process per chunk
        
        Yields:
            lm_tokens: [B, chunk_size, num_patches, lm_hidden]
            frame_indices: List of frame indices in this chunk
            
        Note:
            Images are automatically resized to the expected size for the model.
            Real MedGemma typically expects 896x896 images (4096 patches).
        """
        # Ensure batch dimension
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)  # [1, K, 3, H, W]
        
        B, K, C, H, W = frames.shape
        device = next(self.model.parameters()).device
        
        # Get expected image size
        expected_size = getattr(self, 'expected_image_size', 224)
        
        # Process in chunks
        for start_idx in range(0, K, chunk_size):
            end_idx = min(start_idx + chunk_size, K)
            chunk_frames = frames[:, start_idx:end_idx]  # [B, chunk, 3, H, W]
            chunk_k = chunk_frames.shape[1]
            
            # Reshape for vision_tower: [B*chunk, 3, H, W]
            pixel_values = chunk_frames.reshape(B * chunk_k, C, H, W).to(device)
            
            # Resize to expected image size if needed (critical for real model!)
            if H != expected_size or W != expected_size:
                pixel_values = F.interpolate(
                    pixel_values,
                    size=(expected_size, expected_size),
                    mode='bilinear',
                    align_corners=False,
                )
            
            # Only convert to bfloat16 for real model, not dummy
            if not self._is_dummy:
                pixel_values = pixel_values.to(dtype=self.torch_dtype)
            
            # Normalize (if not already done)
            # Assuming frames are in [0, 1], apply CLIP normalization
            pixel_values = self._normalize_pixels(pixel_values)
            
            # Vision encoding
            if self._is_dummy:
                vision_out = self.model.vision_tower(pixel_values, output_hidden_states=True)
                vision_tokens = vision_out.last_hidden_state
            else:
                vision_out = self.model.vision_tower(
                    pixel_values,
                    output_hidden_states=True,
                )
                vision_tokens = vision_out.last_hidden_state  # [B*chunk, num_patches, vision_hidden]
            
            # Project to LM hidden size
            if self._is_dummy:
                lm_tokens = self.model.multi_modal_projector(vision_tokens)
            else:
                lm_tokens = self.model.multi_modal_projector(vision_tokens)  # [B*chunk, num_patches, lm_hidden]
            
            # Reshape back: [B, chunk, num_patches, lm_hidden]
            num_patches = lm_tokens.shape[1]
            lm_hidden = lm_tokens.shape[2]
            lm_tokens = lm_tokens.reshape(B, chunk_k, num_patches, lm_hidden)
            
            # Update num_vision_tokens based on actual output
            if self.num_vision_tokens != num_patches:
                self.num_vision_tokens = num_patches
            
            frame_indices = list(range(start_idx, end_idx))
            
            yield lm_tokens, frame_indices
    
    def encode_frames_all(
        self,
        frames: torch.Tensor,
        chunk_size: int = 4,
    ) -> torch.Tensor:
        """
        Encode all frames and return concatenated result.
        
        Args:
            frames: [B, K, 3, H, W] or [K, 3, H, W]
            chunk_size: Processing chunk size
        
        Returns:
            lm_tokens: [B, K, num_patches, lm_hidden]
        """
        all_tokens = []
        for lm_tokens, _ in self.encode_frames_to_lm_tokens(frames, chunk_size):
            all_tokens.append(lm_tokens)
        
        return torch.cat(all_tokens, dim=1)
    
    def _normalize_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Apply CLIP/SigLIP normalization."""
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], 
                           device=pixel_values.device, dtype=pixel_values.dtype)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                          device=pixel_values.device, dtype=pixel_values.dtype)
        
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        
        return (pixel_values - mean) / std
    
    def inject_image_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject image embeddings into inputs_embeds at <image> token positions.
        
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            image_embeds: [B, T_img, lm_hidden] - custom image tokens to inject
        
        Returns:
            inputs_embeds: [B, seq_len, lm_hidden] with image tokens injected
        """
        dev = next(self.model.get_input_embeddings().parameters()).device
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        image_embeds = image_embeds.to(dev)

        device = input_ids.device
        B, seq_len = input_ids.shape
        T_img = image_embeds.shape[1]
        lm_hidden = image_embeds.shape[2]
        
        # Get text embeddings
        if self._is_dummy:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        else:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Find image token positions
        image_token_mask = (input_ids == self.image_token_index)  # [B, seq_len]
        
        # Count image tokens per sample
        num_image_tokens = image_token_mask.sum(dim=1)  # [B]
        
        # Handle length mismatch
        for b in range(B):
            n_tokens = num_image_tokens[b].item()
            
            if n_tokens == 0:
                # No image tokens in this sample - skip
                continue
            
            # Get image embeds for this sample
            sample_embeds = image_embeds[b]  # [T_img, lm_hidden]
            
            # Align length
            aligned_embeds = self._align_length(sample_embeds, n_tokens)  # [n_tokens, lm_hidden]
            
            # Get positions
            positions = torch.where(image_token_mask[b])[0]
            
            # Inject
            inputs_embeds[b, positions] = aligned_embeds.to(inputs_embeds.dtype)
        
        return inputs_embeds
    
    def _align_length(
        self,
        embeds: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """
        Align embedding length to target via truncation or interpolation.
        
        Args:
            embeds: [T, hidden]
            target_length: Target sequence length
        
        Returns:
            Aligned embeddings [target_length, hidden]
        """
        T, hidden = embeds.shape
        
        if T == target_length:
            return embeds
        elif T > target_length:
            # Truncate - keep most important (assume first ones or uniform sample)
            indices = torch.linspace(0, T - 1, target_length, dtype=torch.long, device=embeds.device)
            return embeds[indices]
        else:
            # Interpolate
            embeds_3d = embeds.unsqueeze(0).permute(0, 2, 1)  # [1, hidden, T]
            interpolated = F.interpolate(embeds_3d, size=target_length, mode='linear', align_corners=False)
            return interpolated.permute(0, 2, 1).squeeze(0)  # [target_length, hidden]
    
    def forward_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with inputs_embeds through language model.
        
        Args:
            inputs_embeds: [B, seq_len, lm_hidden]
            attention_mask: [B, seq_len]
            labels: [B, seq_len] for computing loss (optional)
        
        Returns:
            Dict with 'loss', 'logits', etc.
        """
        if self._is_dummy:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=_build_position_ids(attention_mask).to(inputs_embeds.device),
                labels=labels,
            )
        else:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=_build_position_ids(attention_mask).to(inputs_embeds.device),
                labels=labels,
            )
        
        return outputs
    
    def generate_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **gen_kwargs,
    ) -> torch.Tensor:
        """
        Generate text with inputs_embeds via language model.
        
        Args:
            inputs_embeds: [B, seq_len, lm_hidden]
            attention_mask: [B, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or greedy decode
            **gen_kwargs: Additional generation kwargs
        
        Returns:
            generated_ids: [B, gen_len]
        """
        gen_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            **gen_kwargs,
        }
        device = inputs_embeds.device
        attention_mask = attention_mask.to(device)

        position_ids = _build_position_ids(attention_mask).to(device)
        
        if self._is_dummy:
            generated = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **gen_config,
            )
        else:
            generated = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **gen_config,
            )
            
        return generated
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs to text."""
        if self._is_dummy:
            return self.processor.decode(token_ids)
        
        texts = []
        for ids in token_ids:
            text = self.processor.decode(ids, skip_special_tokens=True)
            texts.append(text)
        return texts
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device


# ==============================================================================
# Dummy classes for testing without real MedGemma
# ==============================================================================

class DummyVisionOutput:
    """Dummy vision tower output."""
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class DummyVisionTower(nn.Module):
    """
    Dummy vision tower mimicking SigLIP.
    
    SigLIP outputs patches based on image size:
    - 896x896 with 14x14 patch -> 64x64 = 4096 patches
    - 336x336 with 14x14 patch -> 24x24 = 576 patches
    - 224x224 with 14x14 patch -> 16x16 = 256 patches
    
    For dummy model, we output 256 patches by default (matching small test images).
    """
    
    def __init__(self, hidden_size: int = 1152, num_patches: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        # Simple conv to simulate encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, hidden_size, 3, stride=2, padding=1),
        )
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> DummyVisionOutput:
        B = pixel_values.shape[0]
        # Simple encoding
        features = self.encoder(pixel_values)  # [B, hidden, h, w]
        features = features.flatten(2).permute(0, 2, 1)  # [B, patches, hidden]
        
        # Adjust number of patches
        if features.shape[1] != self.num_patches:
            features = F.interpolate(
                features.permute(0, 2, 1),
                size=self.num_patches,
                mode='linear',
                align_corners=False,
            ).permute(0, 2, 1)
        
        features = self.proj(features)
        
        return DummyVisionOutput(last_hidden_state=features)


class DummyProjector(nn.Module):
    """
    Dummy multi-modal projector mimicking MedGemma's projector.
    
    Real MedGemma projector only projects hidden dimension, preserving token count.
    For dummy model, we also just project hidden dimension (no token compression).
    
    Input: [B, num_patches, 1152] (SigLIP output)
    Output: [B, num_patches, 2560] (LM input tokens)
    """
    
    def __init__(
        self, 
        vision_hidden: int = 1152, 
        lm_hidden: int = 2560,
        vision_patches: int = 4096,  # Not used, kept for compatibility
        output_tokens: int = 256,  # Not used, kept for compatibility
    ):
        super().__init__()
        self.vision_hidden = vision_hidden
        self.lm_hidden = lm_hidden
        
        # Only project hidden dimension (like real MedGemma)
        self.hidden_proj = nn.Linear(vision_hidden, lm_hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_patches, 1152] vision tokens from SigLIP
        
        Returns:
            [B, num_patches, 2560] LM tokens (same num_patches)
        """
        # Just project hidden dimension, preserve num_patches
        return self.hidden_proj(x)


class DummyEmbedding(nn.Module):
    """Dummy embedding layer."""
    
    def __init__(self, vocab_size: int = 256000, hidden: int = 2560):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp indices to valid range
        x = x.clamp(0, self.embedding.num_embeddings - 1)
        return self.embedding(x)


class DummyLMOutput:
    """Dummy LM output."""
    def __init__(self, loss: Optional[torch.Tensor], logits: torch.Tensor):
        self.loss = loss
        self.logits = logits


class DummyLanguageModel(nn.Module):
    """Dummy language model."""
    
    def __init__(self, hidden: int = 2560, vocab_size: int = 256000):
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden, vocab_size)
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> DummyLMOutput:
        logits = self.lm_head(inputs_embeds)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return DummyLMOutput(loss=loss, logits=logits)
    
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> torch.Tensor:
        """Dummy generation - returns random tokens."""
        B, seq_len, _ = inputs_embeds.shape
        # Generate random tokens for testing
        gen_tokens = torch.randint(0, 1000, (B, max_new_tokens), device=inputs_embeds.device)
        return gen_tokens


class DummyMedGemmaModel(nn.Module):
    """
    Dummy MedGemma model for testing.
    
    Mimics the real MedGemma architecture:
    - SigLIP vision tower: outputs [B, num_patches, 1152]
    - Projector: projects hidden dim [B, num_patches, 2560]
    - LM: receives image tokens
    
    Note: Real MedGemma's projector only projects hidden dimension,
    it does NOT compress the token count.
    """
    
    def __init__(
        self,
        vision_hidden: int = 1152,
        lm_hidden: int = 2560,
        vision_patches: int = 256,  # Number of vision patches
        num_image_tokens: int = 256,  # Same as vision_patches (no compression)
        vocab_size: int = 256000,
    ):
        super().__init__()
        
        # Vision tower outputs vision_patches tokens
        self.vision_tower = DummyVisionTower(vision_hidden, vision_patches)
        
        # Projector only projects hidden dimension (no token compression)
        self.multi_modal_projector = DummyProjector(
            vision_hidden=vision_hidden, 
            lm_hidden=lm_hidden,
            vision_patches=vision_patches,
            output_tokens=num_image_tokens,
        )
        self.embed_tokens = DummyEmbedding(vocab_size, lm_hidden)
        self.language_model = DummyLanguageModel(lm_hidden, vocab_size)
        
        # Config
        self.config = type('Config', (), {
            'image_token_index': 255999,
            'vision_config': type('VisionConfig', (), {'hidden_size': vision_hidden})(),
            'text_config': type('TextConfig', (), {'hidden_size': lm_hidden})(),
        })()
    
    def get_input_embeddings(self) -> DummyEmbedding:
        return self.embed_tokens


class DummyProcessor:
    """Dummy processor for testing."""
    
    def __init__(self, vocab_size: int = 256000):
        self.vocab_size = vocab_size
        self.image_token_index = 255999
    
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_tensors: str = "pt",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create dummy input_ids with image token placeholders."""
        # Count images in messages
        num_images = sum(
            1 for m in messages 
            if "<image>" in m.get("content", "") or m.get("role") == "user"
        )
        num_images = max(1, num_images)  # At least 1 image placeholder
        
        # Create sequence with image tokens
        # Format: [BOS, image_tokens * 256 * num_images, text_tokens, ...]
        image_tokens = 256 * num_images  # 256 patches per image
        text_tokens = 50  # Dummy text length
        
        seq_len = 1 + image_tokens + text_tokens
        
        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        input_ids[0, 0] = 1  # BOS
        input_ids[0, 1:1+image_tokens] = self.image_token_index  # Image tokens
        input_ids[0, 1+image_tokens:] = torch.randint(100, 1000, (text_tokens,))  # Text
        
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        
        return input_ids, attention_mask
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """Dummy decode."""
        if token_ids.dim() == 1:
            return "This is a dummy generated report for testing purposes."
        return ["This is a dummy generated report for testing purposes."] * token_ids.shape[0]
