#!/usr/bin/env python3
"""
Build template library from training JSONL data.

Usage:
    python scripts/build_templates.py \
        --train_jsonl data/train.jsonl \
        --output_path data/templates.json \
        --top_k 5
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.templates import TemplateLibrary, ABNORMAL_KEYWORDS


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file with one record per line."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
    return records


def extract_section_texts(records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extract section texts from training records.
    
    Expected record format:
    {
        "video_id": "...",
        "report": {
            "esophagus": "Normal esophageal mucosa...",
            "gej": "Z-line is regular...",
            ...
        }
    }
    
    OR flat format:
    {
        "video_id": "...",
        "sections": [
            {"name": "esophagus", "text": "Normal..."},
            ...
        ]
    }
    """
    section_texts = defaultdict(list)
    
    for record in records:
        # Handle nested report dict format
        if 'report' in record and isinstance(record['report'], dict):
            for section_name, text in record['report'].items():
                if text and isinstance(text, str):
                    section_texts[section_name.lower()].append(text.strip())
        
        # Handle sections list format
        elif 'sections' in record and isinstance(record['sections'], list):
            for section in record['sections']:
                if isinstance(section, dict):
                    name = section.get('name', section.get('section_name', ''))
                    text = section.get('text', section.get('content', ''))
                    if name and text:
                        section_texts[name.lower()].append(text.strip())
        
        # Handle flat section fields (esophagus, gej, etc.)
        else:
            for key, value in record.items():
                if key in ['video_id', 'patient_id', 'date', 'metadata']:
                    continue
                if isinstance(value, str) and value.strip():
                    section_texts[key.lower()].append(value.strip())
    
    return dict(section_texts)


def is_normal_text(text: str) -> bool:
    """Check if text describes normal findings."""
    text_lower = text.lower()
    for keyword in ABNORMAL_KEYWORDS:
        if keyword in text_lower:
            return False
    return True


def find_common_templates(
    texts: List[str],
    top_k: int = 5,
    min_count: int = 2
) -> List[str]:
    """
    Find most common normal templates from text list.
    
    Uses exact matching for simplicity. More sophisticated approaches
    could use fuzzy matching or sentence embedding clustering.
    """
    # Filter to normal texts only
    normal_texts = [t for t in texts if is_normal_text(t)]
    
    if not normal_texts:
        return []
    
    # Count occurrences
    counter = Counter(normal_texts)
    
    # Get top-k that appear at least min_count times
    templates = []
    for text, count in counter.most_common():
        if count >= min_count:
            templates.append(text)
            if len(templates) >= top_k:
                break
    
    # If not enough templates meet min_count, relax constraint
    if len(templates) < top_k:
        for text, count in counter.most_common(top_k):
            if text not in templates:
                templates.append(text)
                if len(templates) >= top_k:
                    break
    
    return templates


def build_template_library(
    train_jsonl: str,
    top_k: int = 5,
    min_count: int = 2
) -> TemplateLibrary:
    """Build template library from training data."""
    
    print(f"Loading training data from {train_jsonl}...")
    records = load_jsonl(train_jsonl)
    print(f"  Loaded {len(records)} records")
    
    print("Extracting section texts...")
    section_texts = extract_section_texts(records)
    
    for section, texts in section_texts.items():
        print(f"  {section}: {len(texts)} texts")
    
    print(f"\nFinding top-{top_k} normal templates per section...")
    templates_dict = {}
    
    for section, texts in section_texts.items():
        templates = find_common_templates(texts, top_k=top_k, min_count=min_count)
        templates_dict[section] = templates
        print(f"  {section}: {len(templates)} templates found")
        if templates:
            print(f"    Example: {templates[0][:80]}...")
    
    # Create library
    library = TemplateLibrary()
    library.templates = templates_dict
    
    return library


def main():
    parser = argparse.ArgumentParser(
        description="Build template library from training JSONL"
    )
    parser.add_argument(
        '--train_jsonl',
        type=str,
        required=True,
        help='Path to training JSONL file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='data/templates.json',
        help='Output path for template library JSON'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of templates to keep per section'
    )
    parser.add_argument(
        '--min_count',
        type=int,
        default=2,
        help='Minimum occurrence count for a template'
    )
    
    args = parser.parse_args()
    
    # Build library
    library = build_template_library(
        train_jsonl=args.train_jsonl,
        top_k=args.top_k,
        min_count=args.min_count
    )
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    library.save(str(output_path))
    
    print(f"\nTemplate library saved to {output_path}")
    
    # Print summary
    total_templates = sum(len(t) for t in library.templates.values())
    print(f"Total: {len(library.templates)} sections, {total_templates} templates")


if __name__ == '__main__':
    main()
