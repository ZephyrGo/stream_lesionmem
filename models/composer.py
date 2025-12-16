"""
Report Composer for STREAM-LesionMem.

Composes prompts and final reports from templates and generated content.
"""

import json
from typing import Dict, List, Optional, Tuple, Any
import torch

from data.templates import TemplateLibrary


class ReportComposer:
    """
    Compose prompts for generation and assemble final reports.
    
    Key responsibilities:
    - Build structured JSON prompts for abnormal section generation
    - Create Gemma3 chat format messages
    - Combine normal templates with generated abnormal sections
    """
    
    def __init__(
        self,
        template_library: Optional[TemplateLibrary] = None,
        section_names: Optional[List[str]] = None,
        image_placeholder: str = "<image>",
    ):
        """
        Args:
            template_library: Library for normal section templates
            section_names: List of section names
            image_placeholder: Placeholder for image tokens in prompt
        """
        self.template_library = template_library or TemplateLibrary()
        self.section_names = section_names or self.template_library.get_section_names()
        self.image_placeholder = image_placeholder
    
    def build_generation_prompt(
        self,
        slot_summary: List[Dict],
        abnormal_sections: List[int],
        selected_frames: List[int],
        num_images: int = 4,
    ) -> str:
        """
        Build structured prompt for abnormal section generation.
        
        Args:
            slot_summary: Summary of memory slots
            abnormal_sections: List of section IDs flagged as abnormal
            selected_frames: Indices of selected evidence frames
            num_images: Number of images (for image token count)
        
        Returns:
            Structured JSON prompt string
        """
        # Build slot meta information
        slot_meta = []
        for slot in slot_summary:
            slot_meta.append({
                "slot_id": slot["slot_id"],
                "section": self._get_section_name(slot["section_id"]),
                "confidence": round(slot["confidence"], 3),
                "evidence_frames": slot["evidence_frames"],
            })
        
        # Build abnormal section list
        abnormal_section_names = [
            self._get_section_name(sec_id) 
            for sec_id in abnormal_sections
        ]
        
        # Create structured prompt
        prompt_data = {
            "task": "Generate detailed findings for abnormal sections",
            "instructions": [
                "Analyze the provided endoscopy images carefully",
                "Generate descriptions ONLY for the abnormal sections listed below",
                "Each finding should reference specific visual evidence",
                "Use professional medical terminology",
                "Be specific about lesion characteristics (size, shape, color, location)",
            ],
            "abnormal_sections": abnormal_section_names,
            "slot_metadata": slot_meta,
            "selected_evidence_frames": selected_frames,
            "output_format": {
                "type": "json",
                "schema": {
                    section: "detailed finding description"
                    for section in abnormal_section_names
                }
            }
        }
        
        return json.dumps(prompt_data, indent=2)
    
    def build_messages(
        self,
        slot_summary: List[Dict],
        abnormal_sections: List[int],
        selected_frames: List[int],
        num_images: int = 4,
    ) -> List[Dict[str, str]]:
        """
        Build Gemma3 chat format messages.
        
        Args:
            slot_summary: Summary of memory slots
            abnormal_sections: List of abnormal section IDs
            selected_frames: Selected frame indices
            num_images: Number of images
        
        Returns:
            List of message dicts for chat template
        """
        # Build image placeholders
        image_placeholders = " ".join([self.image_placeholder] * num_images)
        
        # Build structured prompt
        structured_prompt = self.build_generation_prompt(
            slot_summary,
            abnormal_sections,
            selected_frames,
            num_images,
        )
        
        # Build user message
        user_content = f"""You are analyzing endoscopy images for medical report generation.

{image_placeholders}

Based on the images above and the analysis metadata below, generate detailed findings for the abnormal sections.

{structured_prompt}

Please provide your findings in JSON format with one key per abnormal section."""
        
        messages = [
            {
                "role": "user",
                "content": user_content,
            }
        ]
        
        return messages
    
    def compose_final_report(
        self,
        generated_abnormal: Dict[str, str],
        abnormal_sections: List[int],
        section_order: Optional[List[str]] = None,
    ) -> str:
        """
        Compose final report from templates and generated content.
        
        Args:
            generated_abnormal: Dict mapping section_name -> generated text
            abnormal_sections: List of abnormal section IDs
            section_order: Optional custom section order
        
        Returns:
            Complete report text
        """
        section_order = section_order or self.section_names
        abnormal_set = set(abnormal_sections)
        
        report_parts = []
        
        for section_name in section_order:
            section_id = self._get_section_id(section_name)
            
            if section_id in abnormal_set and section_name in generated_abnormal:
                # Use generated content for abnormal sections
                section_text = generated_abnormal[section_name]
            else:
                # Use template for normal sections
                section_text = self.template_library.get_normal_template(section_name)
            
            # Format section
            section_header = section_name.replace("_", " ").title()
            report_parts.append(f"**{section_header}**: {section_text}")
        
        return "\n\n".join(report_parts)
    
    def parse_generated_output(
        self,
        generated_text: str,
        abnormal_sections: List[int],
    ) -> Dict[str, str]:
        """
        Parse generated text into section-wise findings.
        
        Args:
            generated_text: Raw generated text from LLM
            abnormal_sections: Expected abnormal section IDs
        
        Returns:
            Dict mapping section_name -> finding text
        """
        findings = {}
        
        # Try JSON parsing first
        try:
            # Find JSON in text
            json_start = generated_text.find("{")
            json_end = generated_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = generated_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                if isinstance(parsed, dict):
                    findings = parsed
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract by section names
        if not findings:
            for sec_id in abnormal_sections:
                section_name = self._get_section_name(sec_id)
                
                # Look for section mention
                lower_text = generated_text.lower()
                section_lower = section_name.lower().replace("_", " ")
                
                idx = lower_text.find(section_lower)
                if idx >= 0:
                    # Extract text after section name
                    start = idx + len(section_lower)
                    # Find next section or end
                    end = len(generated_text)
                    for other_sec in self.section_names:
                        other_lower = other_sec.lower().replace("_", " ")
                        other_idx = lower_text.find(other_lower, start)
                        if other_idx > start:
                            end = min(end, other_idx)
                    
                    section_text = generated_text[start:end].strip(": \n")
                    if section_text:
                        findings[section_name] = section_text
        
        # Fill missing with placeholder
        for sec_id in abnormal_sections:
            section_name = self._get_section_name(sec_id)
            if section_name not in findings:
                findings[section_name] = "Abnormality observed. Further evaluation recommended."
        
        return findings
    
    def build_image_embeds(
        self,
        selected_frame_tokens: torch.Tensor,
        memory_embeds: Optional[torch.Tensor] = None,
        max_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build final image embeddings for injection.
        
        Args:
            selected_frame_tokens: [B, n_frames, num_patches, lm_hidden]
            memory_embeds: [B, T_mem, lm_hidden] optional memory tokens
            max_tokens: Optional max tokens limit
        
        Returns:
            image_embeds: [B, T_img, lm_hidden] for injection
        """
        B, n_frames, num_patches, lm_hidden = selected_frame_tokens.shape
        
        # Flatten frame tokens
        frame_embeds = selected_frame_tokens.reshape(B, n_frames * num_patches, lm_hidden)
        
        # Combine with memory
        if memory_embeds is not None:
            combined = torch.cat([frame_embeds, memory_embeds], dim=1)
        else:
            combined = frame_embeds
        
        # Truncate if needed
        if max_tokens and combined.shape[1] > max_tokens:
            combined = combined[:, :max_tokens]
        
        return combined
    
    def _get_section_name(self, section_id: int) -> str:
        """Get section name by ID."""
        if 0 <= section_id < len(self.section_names):
            return self.section_names[section_id]
        return f"section_{section_id}"
    
    def _get_section_id(self, section_name: str) -> int:
        """Get section ID by name."""
        try:
            return self.section_names.index(section_name)
        except ValueError:
            return -1


class ComposerOutput:
    """Output from composer's build_for_generation method."""
    
    def __init__(
        self,
        messages: List[Dict[str, str]],
        image_embeds: torch.Tensor,
        abnormal_sections: List[int],
        normal_sections: Dict[str, str],
    ):
        self.messages = messages
        self.image_embeds = image_embeds
        self.abnormal_sections = abnormal_sections
        self.normal_sections = normal_sections


def create_default_composer(
    templates_path: Optional[str] = None,
) -> ReportComposer:
    """Create composer with default settings."""
    template_library = TemplateLibrary(templates_path)
    return ReportComposer(template_library=template_library)
