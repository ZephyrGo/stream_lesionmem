"""
Template library for normal section generation.

Normal sections are generated from templates, only abnormal sections require LLM generation.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np


class TemplateLibrary:
    """
    Template library for generating normal section text.
    
    Built from training reports, stores top-K templates per section.
    Provides abnormal detection based on keyword matching and template deviation.
    """
    
    # Default abnormal keywords for endoscopy
    DEFAULT_ABNORMAL_KEYWORDS = {
        "lesion", "polyp", "ulcer", "erosion", "bleeding", "mass", "tumor",
        "abnormal", "irregular", "suspicious", "nodule", "stricture",
        "inflammation", "erythema", "edema", "necrosis", "perforation",
        "foreign body", "diverticulum", "varices", "hemorrhage",
    }
    
    def __init__(
        self,
        templates_path: Optional[str] = None,
        abnormal_keywords: Optional[Set[str]] = None,
        top_k: int = 5,
    ):
        """
        Args:
            templates_path: Path to JSON file with templates
            abnormal_keywords: Set of keywords indicating abnormality
            top_k: Number of templates to keep per section
        """
        self.top_k = top_k
        self.abnormal_keywords = abnormal_keywords or self.DEFAULT_ABNORMAL_KEYWORDS
        
        # section_name -> list of (template_text, frequency)
        self.templates: Dict[str, List[Tuple[str, int]]] = {}
        
        # section_name -> set of normal template hashes (for fast lookup)
        self._normal_hashes: Dict[str, Set[int]] = defaultdict(set)
        
        if templates_path and Path(templates_path).exists():
            self.load(templates_path)
        else:
            self._init_default_templates()
    
    def _init_default_templates(self) -> None:
        """Initialize with default templates for common endoscopy sections."""
        default_templates = {
            "esophagus": [
                ("The esophageal mucosa appears normal with no lesions or abnormalities.", 100),
                ("Normal esophagus. No strictures, masses, or ulcerations seen.", 80),
                ("Esophageal examination unremarkable. Normal Z-line.", 60),
            ],
            "gastroesophageal_junction": [
                ("GE junction appears normal. No evidence of Barrett's esophagus.", 100),
                ("Normal gastroesophageal junction. No hiatal hernia present.", 80),
                ("GEJ examination normal. Regular Z-line at expected location.", 60),
            ],
            "cardia": [
                ("Cardia appears normal with intact mucosa.", 100),
                ("Normal cardiac examination. No masses or lesions.", 80),
            ],
            "fundus": [
                ("Fundus examination normal on retroflexion. No lesions identified.", 100),
                ("Normal fundic mucosa. No polyps or masses.", 80),
            ],
            "body": [
                ("Gastric body mucosa appears normal. No ulcers or erosions.", 100),
                ("Normal body examination. Rugal folds intact.", 80),
            ],
            "antrum": [
                ("Antral mucosa normal. No erythema or erosions.", 100),
                ("Normal antrum. No evidence of gastritis.", 80),
            ],
            "pylorus": [
                ("Pylorus opens and closes normally. No deformity.", 100),
                ("Normal pyloric examination. Patent pyloric channel.", 80),
            ],
            "duodenum": [
                ("Duodenal bulb and second portion normal. No ulcers.", 100),
                ("Normal duodenal examination. Mucosa intact.", 80),
            ],
        }
        
        for section, templates in default_templates.items():
            self.templates[section] = templates[:self.top_k]
            for template, _ in templates:
                self._normal_hashes[section].add(hash(self._normalize_text(template)))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def build_from_reports(
        self,
        reports: List[Dict[str, str]],
        min_frequency: int = 3,
    ) -> None:
        """
        Build template library from training reports.
        
        Args:
            reports: List of dicts mapping section_name -> section_text
            min_frequency: Minimum frequency to include a template
        """
        # Count section texts
        section_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for report in reports:
            for section_name, section_text in report.items():
                if not self.is_abnormal(section_text):
                    normalized = self._normalize_text(section_text)
                    section_counts[section_name][normalized] += 1
        
        # Build templates
        for section_name, text_counts in section_counts.items():
            # Sort by frequency
            sorted_texts = sorted(text_counts.items(), key=lambda x: -x[1])
            
            # Filter by min frequency and take top_k
            templates = []
            for text, count in sorted_texts:
                if count >= min_frequency:
                    templates.append((text, count))
                    self._normal_hashes[section_name].add(hash(text))
                if len(templates) >= self.top_k:
                    break
            
            self.templates[section_name] = templates
    
    def is_abnormal(self, text: str) -> bool:
        """
        Check if section text indicates abnormality.
        
        Uses keyword matching.
        """
        text_lower = text.lower()
        
        for keyword in self.abnormal_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    def is_template_deviation(self, section_name: str, text: str) -> bool:
        """
        Check if text deviates significantly from normal templates.
        
        This can be used as weak label for abnormality.
        """
        normalized = self._normalize_text(text)
        text_hash = hash(normalized)
        
        # Check exact match
        if text_hash in self._normal_hashes.get(section_name, set()):
            return False
        
        # Check similarity with templates
        if section_name in self.templates:
            for template, _ in self.templates[section_name]:
                similarity = self._text_similarity(normalized, template)
                if similarity > 0.8:
                    return False
        
        return True
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute simple word overlap similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_normal_template(self, section_name: str) -> str:
        """
        Get the most common normal template for a section.
        
        Returns empty string if no template available.
        """
        if section_name not in self.templates:
            return f"Normal {section_name} examination."
        
        templates = self.templates[section_name]
        if not templates:
            return f"Normal {section_name} examination."
        
        # Return most frequent template
        return templates[0][0]
    
    def get_random_template(self, section_name: str) -> str:
        """Get a random template weighted by frequency."""
        if section_name not in self.templates:
            return f"Normal {section_name} examination."
        
        templates = self.templates[section_name]
        if not templates:
            return f"Normal {section_name} examination."
        
        texts, weights = zip(*templates)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        
        idx = np.random.choice(len(texts), p=weights)
        return texts[idx]
    
    def get_section_names(self) -> List[str]:
        """Get list of all section names."""
        return list(self.templates.keys())
    
    def save(self, path: str) -> None:
        """Save template library to JSON file."""
        data = {
            "templates": self.templates,
            "abnormal_keywords": list(self.abnormal_keywords),
            "top_k": self.top_k,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str) -> None:
        """Load template library from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.templates = {k: [tuple(t) for t in v] for k, v in data.get("templates", {}).items()}
        self.abnormal_keywords = set(data.get("abnormal_keywords", self.DEFAULT_ABNORMAL_KEYWORDS))
        self.top_k = data.get("top_k", self.top_k)
        
        # Rebuild normal hashes
        self._normal_hashes.clear()
        for section, templates in self.templates.items():
            for text, _ in templates:
                self._normal_hashes[section].add(hash(self._normalize_text(text)))


def build_template_library_from_jsonl(
    jsonl_path: str,
    output_path: str,
    top_k: int = 5,
    min_frequency: int = 3,
) -> TemplateLibrary:
    """
    Build template library from JSONL file of reports.
    
    Args:
        jsonl_path: Path to JSONL file with reports
        output_path: Path to save template library
        top_k: Number of templates per section
        min_frequency: Minimum frequency for templates
    
    Returns:
        Built TemplateLibrary
    """
    reports = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if "sections" in data:
                    reports.append(data["sections"])
    
    library = TemplateLibrary(top_k=top_k)
    library.build_from_reports(reports, min_frequency=min_frequency)
    library.save(output_path)
    
    return library
