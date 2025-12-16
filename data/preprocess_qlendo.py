"""
QL_Endo Data Preprocessing for STREAM-LesionMem.

Parses findings text into anatomical sections and detects abnormalities.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict


# =============================================================================
# Section Definitions for Upper GI Endoscopy
# =============================================================================

# Standard anatomical sections in examination order
SECTION_NAMES = [
    "esophagus",
    "gastroesophageal_junction",  # cardia / GEJ
    "fundus",
    "body",
    "angle",  # angulus
    "antrum",
    "pylorus",
    "duodenal_bulb",
    "descending_duodenum",
]

# Section name aliases (for parsing)
SECTION_ALIASES = {
    # Esophagus
    "esophagus": ["esophageal", "esophagus", "upper esophagus", "middle esophagus", 
                  "lower esophagus", "esophageal mucosa"],
    
    # GEJ / Cardia
    "gastroesophageal_junction": ["cardia", "cardiac", "cardial", "gastroesophageal junction", 
                                   "ge junction", "gej", "z-line", "dentate line",
                                   "squamocolumnar junction"],
    
    # Fundus
    "fundus": ["fundus", "gastric fundus", "fundic"],
    
    # Body
    "body": ["gastric body", "body", "gastric remnant body", "remnant gastric body",
             "lesser curvature", "greater curvature"],
    
    # Angle
    "angle": ["angle", "angulus", "gastric angle", "gastric angulus"],
    
    # Antrum
    "antrum": ["antrum", "gastric antrum", "antral"],
    
    # Pylorus
    "pylorus": ["pylorus", "pyloric", "pyloric orifice", "pyloric mucosa"],
    
    # Duodenal bulb
    "duodenal_bulb": ["duodenal bulb", "bulb", "bulbar"],
    
    # Descending duodenum
    "descending_duodenum": ["descending duodenum", "duodenal descending", 
                            "descending duodenal", "second portion", "d2"],
}

# Build reverse mapping: alias -> canonical name
ALIAS_TO_SECTION = {}
for section, aliases in SECTION_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_SECTION[alias.lower()] = section


# =============================================================================
# Abnormality Detection Keywords
# =============================================================================

# Keywords indicating abnormal findings (grouped by severity/type)
ABNORMAL_KEYWORDS = {
    # Lesions & Masses
    "lesion": ["lesion", "mass", "tumor", "neoplasm", "carcinoma", "cancer",
               "neoplastic", "malignant", "suspicious"],
    
    # Polyps
    "polyp": ["polyp", "polyps", "polypoid", "yamada"],
    
    # Ulcers & Erosions  
    "ulcer": ["ulcer", "ulcers", "ulcerated", "ulceration", "erosion", "erosions",
              "erosive", "defect", "mucosal defect"],
    
    # Inflammation
    "inflammation": ["inflammation", "inflammatory", "erythema", "erythematous",
                     "congestion", "congested", "edema", "edematous", "swelling",
                     "hyperemia", "redness"],
    
    # Bleeding
    "bleeding": ["bleeding", "hemorrhage", "hemorrhagic", "blood", "bloody",
                 "hematin", "old hemorrhagic spots"],
    
    # Structural
    "structural": ["stricture", "stenosis", "stenotic", "narrowing", "obstruction",
                   "diverticulum", "hernia", "hiatal hernia", "varices"],
    
    # Atrophy & Metaplasia
    "atrophy": ["atrophy", "atrophic", "metaplasia", "intestinal metaplasia",
                "barrett", "dysplasia"],
    
    # Surface changes
    "surface": ["rough", "roughness", "irregular", "irregularity", "nodular",
                "nodule", "elevation", "elevated", "depression", "depressed",
                "friable", "fragile"],
    
    # Heterotopic
    "heterotopic": ["heterotopic", "ectopic", "ectopic mucosa"],
    
    # Biopsy indicators (usually means abnormal)
    "biopsy": ["biopsy", "biopsies", "specimen", "specimens"],
}

# Flatten for quick lookup
ALL_ABNORMAL_KEYWORDS = set()
for keywords in ABNORMAL_KEYWORDS.values():
    ALL_ABNORMAL_KEYWORDS.update(k.lower() for k in keywords)

# Normal indicators (to filter out false positives)
NORMAL_INDICATORS = [
    "smooth", "normal", "unremarkable", "no abnormalities", "no ulcers",
    "no varices", "no lesions", "patent", "clear", "regular",
    "good peristalsis", "normal peristalsis", "normal contraction",
    "color consistent with surrounding", "no significant abnormalities",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SectionFinding:
    """Parsed finding for a single section."""
    section_id: int
    section_name: str
    text: str
    is_abnormal: bool
    abnormal_keywords: List[str] = field(default_factory=list)
    confidence: float = 1.0  # Confidence in abnormality detection


@dataclass
class ParsedExam:
    """Fully parsed examination record."""
    exam_id: str
    folder_path: str
    image_count: int
    diagnosis: str
    full_findings: str
    sections: List[SectionFinding] = field(default_factory=list)
    abnormal_section_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "exam_id": self.exam_id,
            "folder_path": self.folder_path,
            "image_count": self.image_count,
            "diagnosis": self.diagnosis,
            "full_findings": self.full_findings,
            "sections": [asdict(s) for s in self.sections],
            "abnormal_section_ids": self.abnormal_section_ids,
        }


# =============================================================================
# Parsing Functions
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for matching."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def find_section_for_text(text: str) -> Optional[str]:
    """
    Find which anatomical section a text segment belongs to.
    
    Returns canonical section name or None.
    """
    text_lower = normalize_text(text)
    
    # Check each alias
    best_match = None
    best_pos = len(text_lower)  # Position of match (earlier is better)
    
    for alias, section in ALIAS_TO_SECTION.items():
        pos = text_lower.find(alias)
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best_match = section
    
    return best_match


def detect_abnormality(text: str) -> Tuple[bool, List[str], float]:
    """
    Detect if text describes abnormal findings.
    
    Returns:
        is_abnormal: Whether abnormality detected
        keywords: List of detected abnormal keywords
        confidence: Confidence score (0-1)
    """
    text_lower = normalize_text(text)
    
    # Check for normal indicators first
    is_clearly_normal = False
    for indicator in NORMAL_INDICATORS:
        if indicator in text_lower:
            # Check if this is the main description (not negated)
            # e.g., "no ulcers" contains "ulcers" but is normal
            is_clearly_normal = True
            break
    
    # Find abnormal keywords
    found_keywords = []
    for keyword in ALL_ABNORMAL_KEYWORDS:
        if keyword in text_lower:
            # Check it's not negated
            negation_patterns = [f"no {keyword}", f"without {keyword}", f"no obvious {keyword}"]
            is_negated = any(neg in text_lower for neg in negation_patterns)
            
            if not is_negated:
                found_keywords.append(keyword)
    
    # Determine abnormality
    if not found_keywords:
        return False, [], 0.0
    
    # Filter out mild inflammation if clearly normal otherwise
    mild_keywords = {"congestion", "congested", "edema", "edematous", "erythema"}
    severe_keywords = set(found_keywords) - mild_keywords
    
    if not severe_keywords and is_clearly_normal:
        # Only mild inflammation, probably normal variant
        return False, found_keywords, 0.3
    
    # Calculate confidence based on keyword severity
    confidence = min(0.5 + 0.1 * len(severe_keywords), 1.0)
    
    return True, found_keywords, confidence


def split_findings_into_paragraphs(text: str) -> List[str]:
    """Split findings text into paragraphs."""
    # Split by double newline or single newline with leading capital
    paragraphs = re.split(r'\n\s*\n|\n(?=[A-Z])', text)
    
    # Also try splitting by "The [section]" pattern
    if len(paragraphs) <= 2:
        paragraphs = re.split(r'(?=The [a-z])', text)
    
    # Clean up
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def parse_findings(findings_text: str) -> List[SectionFinding]:
    """
    Parse findings text into section-wise findings.
    
    Strategy:
    1. Split text into paragraphs
    2. For each paragraph, identify which section it belongs to
    3. Detect abnormality in each section
    """
    paragraphs = split_findings_into_paragraphs(findings_text)
    
    # Track which sections we've found
    section_texts: Dict[str, List[str]] = defaultdict(list)
    unassigned_texts: List[str] = []
    
    for para in paragraphs:
        section = find_section_for_text(para)
        if section:
            section_texts[section].append(para)
        else:
            unassigned_texts.append(para)
    
    # Build section findings
    findings = []
    
    for section_id, section_name in enumerate(SECTION_NAMES):
        texts = section_texts.get(section_name, [])
        
        if texts:
            combined_text = " ".join(texts)
        else:
            combined_text = ""
        
        # Detect abnormality
        is_abnormal, keywords, confidence = detect_abnormality(combined_text)
        
        finding = SectionFinding(
            section_id=section_id,
            section_name=section_name,
            text=combined_text,
            is_abnormal=is_abnormal,
            abnormal_keywords=keywords,
            confidence=confidence,
        )
        findings.append(finding)
    
    return findings


def estimate_frame2section(
    image_count: int,
    num_sections: int = 9,
) -> Dict[int, int]:
    """
    Estimate frame-to-section mapping based on typical examination order.
    
    Heuristic: Frames are roughly evenly distributed across sections,
    following the standard examination order.
    
    Args:
        image_count: Total number of images
        num_sections: Number of sections (default 9)
    
    Returns:
        Dict mapping frame_idx -> section_id
    """
    # Typical distribution weights (some sections get more frames)
    # esophagus, gej, fundus, body, angle, antrum, pylorus, bulb, descending
    weights = [0.10, 0.05, 0.10, 0.20, 0.10, 0.20, 0.05, 0.10, 0.10]
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Calculate frame ranges for each section
    frame2section = {}
    current_frame = 0
    
    for section_id, weight in enumerate(weights):
        num_frames = max(1, int(image_count * weight))
        
        # Assign frames to this section
        for i in range(num_frames):
            if current_frame < image_count:
                frame2section[current_frame] = section_id
                current_frame += 1
    
    # Assign remaining frames to last section
    while current_frame < image_count:
        frame2section[current_frame] = num_sections - 1
        current_frame += 1
    
    return frame2section


def parse_exam_record(record: Dict[str, Any]) -> ParsedExam:
    """
    Parse a single examination record.
    
    Args:
        record: Raw record from JSON
    
    Returns:
        ParsedExam object
    """
    exam_id = record.get("exam_id", "")
    folder_path = record.get("folder_path", "")
    image_count = record.get("image_count", 0)
    diagnosis = record.get("diagnosis", "")
    findings = record.get("findings", "")
    
    # Parse findings into sections
    sections = parse_findings(findings)
    
    # Get abnormal section IDs
    abnormal_section_ids = [s.section_id for s in sections if s.is_abnormal]
    
    return ParsedExam(
        exam_id=exam_id,
        folder_path=folder_path,
        image_count=image_count,
        diagnosis=diagnosis,
        full_findings=findings,
        sections=sections,
        abnormal_section_ids=abnormal_section_ids,
    )


# =============================================================================
# Main Preprocessing Pipeline
# =============================================================================

def preprocess_qlendo_dataset(
    input_json: str,
    output_json: str,
    min_images: int = 10,
    max_images: int = 200,
    require_findings: bool = True,
) -> Dict[str, Any]:
    """
    Preprocess QL_Endo dataset for training.
    
    Args:
        input_json: Path to input diagnosis_en.json
        output_json: Path to output preprocessed JSON
        min_images: Minimum images per exam
        max_images: Maximum images per exam
        require_findings: Skip exams without findings
    
    Returns:
        Statistics dict
    """
    print(f"Loading data from {input_json}...")
    
    with open(input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    records = raw_data.get("data", raw_data)
    if isinstance(records, dict):
        records = [records]
    
    print(f"Total records: {len(records)}")
    
    # Process each record
    processed = []
    stats = {
        "total": len(records),
        "processed": 0,
        "skipped_no_findings": 0,
        "skipped_image_count": 0,
        "skipped_no_folder": 0,
        "total_abnormal_sections": 0,
        "section_abnormal_counts": defaultdict(int),
    }
    
    for record in records:
        # Filter checks
        if require_findings and not record.get("findings", "").strip():
            stats["skipped_no_findings"] += 1
            continue
        
        image_count = record.get("image_count", 0)
        if image_count < min_images or image_count > max_images:
            stats["skipped_image_count"] += 1
            continue
        
        if not record.get("folder_exists", True):
            stats["skipped_no_folder"] += 1
            continue
        
        # Parse record
        parsed = parse_exam_record(record)
        
        # Add frame2section estimate
        frame2section = estimate_frame2section(parsed.image_count)
        
        # Convert to dict and add frame mapping
        parsed_dict = parsed.to_dict()
        parsed_dict["frame2section"] = frame2section
        
        processed.append(parsed_dict)
        
        # Update stats
        stats["processed"] += 1
        stats["total_abnormal_sections"] += len(parsed.abnormal_section_ids)
        for section_id in parsed.abnormal_section_ids:
            section_name = SECTION_NAMES[section_id]
            stats["section_abnormal_counts"][section_name] += 1
    
    # Save processed data
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({"data": processed, "stats": dict(stats)}, f, indent=2, ensure_ascii=False)
    
    print(f"\nPreprocessing complete!")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped (no findings): {stats['skipped_no_findings']}")
    print(f"  Skipped (image count): {stats['skipped_image_count']}")
    print(f"  Skipped (no folder): {stats['skipped_no_folder']}")
    print(f"  Total abnormal sections: {stats['total_abnormal_sections']}")
    print(f"\nAbnormal section distribution:")
    for section, count in sorted(stats["section_abnormal_counts"].items(), key=lambda x: -x[1]):
        print(f"    {section}: {count}")
    
    print(f"\nSaved to {output_json}")
    
    return stats


def create_train_val_split(
    preprocessed_json: str,
    train_json: str,
    val_json: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Split preprocessed data into train/val sets.
    
    Args:
        preprocessed_json: Path to preprocessed JSON
        train_json: Output path for training set
        val_json: Output path for validation set
        val_ratio: Fraction for validation
        seed: Random seed
    """
    import random
    random.seed(seed)
    
    with open(preprocessed_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = data["data"]
    random.shuffle(records)
    
    val_size = int(len(records) * val_ratio)
    val_records = records[:val_size]
    train_records = records[val_size:]
    
    # Save splits
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump({"data": train_records}, f, indent=2, ensure_ascii=False)
    
    with open(val_json, 'w', encoding='utf-8') as f:
        json.dump({"data": val_records}, f, indent=2, ensure_ascii=False)
    
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")
    print(f"Saved to {train_json} and {val_json}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess QL_Endo dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSON path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--min_images", type=int, default=10)
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--split", action="store_true", help="Also create train/val split")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    
    args = parser.parse_args()
    
    stats = preprocess_qlendo_dataset(
        args.input,
        args.output,
        min_images=args.min_images,
        max_images=args.max_images,
    )
    
    if args.split:
        output_dir = Path(args.output).parent
        create_train_val_split(
            args.output,
            str(output_dir / "train.json"),
            str(output_dir / "val.json"),
            val_ratio=args.val_ratio,
        )
