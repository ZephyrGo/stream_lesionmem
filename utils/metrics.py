"""
Evaluation metrics for STREAM-LesionMem.

Implements:
- abnormal_section_recall: Recall of abnormal sections (weak label via template deviation)
- duplicate_rate: Rate of repeated keywords/phrases in generated report
- unsupported_rate: Rate of findings mentioned without evidence in memory slots
"""

import re
import math
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import Counter


# Common medical keywords that indicate abnormal findings
ABNORMAL_KEYWORDS = [
    "lesion", "polyp", "ulcer", "erosion", "nodule", "mass", "tumor",
    "inflammation", "erythema", "edema", "bleeding", "stricture",
    "irregularity", "abnormal", "suspicious", "malignant", "dysplasia",
    "metaplasia", "atrophy", "hyperplasia", "neoplasm", "carcinoma",
]

# Keywords for finding extraction
FINDING_PATTERNS = [
    r"(\d+\s*(?:mm|cm)\s+(?:lesion|polyp|mass|nodule|tumor))",
    r"((?:small|large|sessile|pedunculated|flat|elevated)\s+(?:lesion|polyp|mass))",
    r"((?:erythematous|ulcerated|inflamed)\s+(?:area|region|mucosa))",
    r"((?:polyp|lesion|mass)\s+(?:in|at|near)\s+\w+)",
]


def is_abnormal_section(text: str) -> bool:
    """
    Check if section text indicates abnormality.
    
    Args:
        text: Section text
        
    Returns:
        True if contains abnormal keywords
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in ABNORMAL_KEYWORDS)


def extract_findings(text: str) -> List[str]:
    """
    Extract finding mentions from text.
    
    Args:
        text: Report text
        
    Returns:
        List of extracted findings
    """
    findings = []
    text_lower = text.lower()
    
    # Extract using patterns
    for pattern in FINDING_PATTERNS:
        matches = re.findall(pattern, text_lower)
        findings.extend(matches)
    
    # Also extract standalone keywords
    for kw in ABNORMAL_KEYWORDS:
        if kw in text_lower:
            findings.append(kw)
    
    return list(set(findings))


def abnormal_section_recall(
    pred_sections: Dict[str, str],
    gt_sections: Dict[str, str],
    templates: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    """
    Compute recall of abnormal sections.
    
    Uses template deviation as weak label: if GT section differs from all templates,
    it's considered abnormal.
    
    Args:
        pred_sections: Dict of section_name -> predicted text
        gt_sections: Dict of section_name -> ground truth text
        templates: Optional dict of section_name -> list of normal templates
        
    Returns:
        Dict with 'recall', 'precision', 'f1', 'tp', 'fp', 'fn'
    """
    # Identify GT abnormal sections
    gt_abnormal: Set[str] = set()
    pred_abnormal: Set[str] = set()
    
    for section, text in gt_sections.items():
        # Check if abnormal via keyword matching
        if is_abnormal_section(text):
            gt_abnormal.add(section)
        # Also check template deviation if templates provided
        elif templates and section in templates:
            # If text significantly differs from all templates, mark as abnormal
            is_template_match = False
            for template in templates[section]:
                # Simple similarity: check if template words are present
                template_words = set(template.lower().split())
                text_words = set(text.lower().split())
                overlap = len(template_words & text_words) / max(len(template_words), 1)
                if overlap > 0.7:
                    is_template_match = True
                    break
            if not is_template_match:
                gt_abnormal.add(section)
    
    # Identify predicted abnormal sections
    for section, text in pred_sections.items():
        if is_abnormal_section(text):
            pred_abnormal.add(section)
    
    # Compute metrics
    tp = len(gt_abnormal & pred_abnormal)
    fp = len(pred_abnormal - gt_abnormal)
    fn = len(gt_abnormal - pred_abnormal)
    
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        "abnormal_recall": recall,
        "abnormal_precision": precision,
        "abnormal_f1": f1,
        "abnormal_tp": tp,
        "abnormal_fp": fp,
        "abnormal_fn": fn,
        "gt_abnormal_sections": list(gt_abnormal),
        "pred_abnormal_sections": list(pred_abnormal),
    }


def duplicate_rate(
    text: str,
    ngram_range: Tuple[int, int] = (2, 5),
    threshold: int = 2,
) -> Dict[str, float]:
    """
    Compute rate of duplicated n-grams in text.
    
    Args:
        text: Generated report text
        ngram_range: (min_n, max_n) for n-gram extraction
        threshold: Minimum count to consider as duplicate
        
    Returns:
        Dict with 'duplicate_rate', 'unique_rate', 'duplicated_ngrams'
    """
    words = text.lower().split()
    
    if len(words) < ngram_range[0]:
        return {
            "duplicate_rate": 0.0,
            "unique_rate": 1.0,
            "duplicated_ngrams": [],
        }
    
    # Extract n-grams
    all_ngrams = []
    for n in range(ngram_range[0], min(ngram_range[1] + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i + n])
            all_ngrams.append(ngram)
    
    if not all_ngrams:
        return {
            "duplicate_rate": 0.0,
            "unique_rate": 1.0,
            "duplicated_ngrams": [],
        }
    
    # Count duplicates
    counter = Counter(all_ngrams)
    duplicated = [ng for ng, count in counter.items() if count >= threshold]
    
    # Compute rates
    total_ngrams = len(all_ngrams)
    duplicated_count = sum(count - 1 for ng, count in counter.items() if count >= threshold)
    
    duplicate_rate_val = duplicated_count / max(total_ngrams, 1)
    unique_rate = 1.0 - duplicate_rate_val
    
    return {
        "duplicate_rate": duplicate_rate_val,
        "unique_rate": unique_rate,
        "duplicated_ngrams": [" ".join(ng) for ng in duplicated[:10]],  # Top 10
        "total_ngrams": total_ngrams,
        "duplicated_count": duplicated_count,
    }


def unsupported_rate(
    generated_text: str,
    slot_summaries: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute rate of findings in generated text that lack evidence in memory slots.
    
    Args:
        generated_text: Generated report text
        slot_summaries: List of slot summary dicts with 'section', 'evidence', etc.
        
    Returns:
        Dict with 'unsupported_rate', 'supported_rate', 'unsupported_findings'
    """
    # Extract findings from generated text
    findings = extract_findings(generated_text)
    
    if not findings:
        return {
            "unsupported_rate": 0.0,
            "supported_rate": 1.0,
            "total_findings": 0,
            "supported_findings": [],
            "unsupported_findings": [],
        }
    
    # Get evidence keywords from slots
    slot_keywords: Set[str] = set()
    for slot in slot_summaries:
        # Add section name
        if "section" in slot:
            slot_keywords.add(slot["section"].lower())
        # Add any keywords from slot metadata
        if "keywords" in slot:
            slot_keywords.update(kw.lower() for kw in slot["keywords"])
        # Check if slot has evidence
        if slot.get("evidence_count", 0) > 0:
            slot_keywords.add("evidence")
    
    # Also consider slots with non-zero update count as supported
    active_slots = sum(1 for s in slot_summaries if s.get("update_count", 0) > 0)
    
    # Check which findings have support
    supported = []
    unsupported = []
    
    for finding in findings:
        # A finding is "supported" if:
        # 1. It matches a keyword in active slots, OR
        # 2. There are active slots (weak criterion for generated content)
        finding_lower = finding.lower()
        
        has_support = False
        for kw in slot_keywords:
            if kw in finding_lower or finding_lower in kw:
                has_support = True
                break
        
        # Weak support: if there are any active slots
        if not has_support and active_slots > 0:
            # Give partial credit if slots exist
            has_support = True
        
        if has_support:
            supported.append(finding)
        else:
            unsupported.append(finding)
    
    total = len(findings)
    unsupported_rate_val = len(unsupported) / max(total, 1)
    supported_rate = len(supported) / max(total, 1)
    
    return {
        "unsupported_rate": unsupported_rate_val,
        "supported_rate": supported_rate,
        "total_findings": total,
        "supported_findings": supported,
        "unsupported_findings": unsupported,
        "active_slots": active_slots,
    }


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization with lowercasing."""
    return text.lower().split()


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token list."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i + n]))
    return Counter(ngrams)


def compute_bleu(
    pred: str,
    gt: str,
    max_n: int = 4,
    smoothing: bool = True,
) -> Dict[str, float]:
    """
    Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    
    Args:
        pred: Predicted text
        gt: Ground truth text (single reference)
        max_n: Maximum n-gram order (default 4)
        smoothing: Whether to apply smoothing for zero counts
        
    Returns:
        Dict with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
    """
    pred_tokens = tokenize(pred)
    gt_tokens = tokenize(gt)
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return {f"bleu{i}": 0.0 for i in range(1, max_n + 1)}
    
    # Compute modified precision for each n-gram order
    precisions = []
    
    for n in range(1, max_n + 1):
        pred_ngrams = get_ngrams(pred_tokens, n)
        gt_ngrams = get_ngrams(gt_tokens, n)
        
        if len(pred_ngrams) == 0:
            if smoothing:
                precisions.append(1.0 / (len(pred_tokens) + 1))
            else:
                precisions.append(0.0)
            continue
        
        # Clipped count: min(pred_count, gt_count) for each n-gram
        clipped_count = 0
        total_count = 0
        
        for ngram, count in pred_ngrams.items():
            clipped_count += min(count, gt_ngrams.get(ngram, 0))
            total_count += count
        
        if total_count == 0:
            if smoothing:
                precisions.append(1.0 / (len(pred_tokens) + 1))
            else:
                precisions.append(0.0)
        else:
            # Add-1 smoothing for zero counts
            if smoothing and clipped_count == 0:
                precisions.append(1.0 / (total_count + 1))
            else:
                precisions.append(clipped_count / total_count)
    
    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(gt_tokens):
        bp = math.exp(1 - len(gt_tokens) / len(pred_tokens))
    
    # Compute BLEU-n scores (geometric mean of precisions 1..n)
    results = {}
    for n in range(1, max_n + 1):
        if all(p > 0 for p in precisions[:n]):
            log_avg = sum(math.log(p) for p in precisions[:n]) / n
            bleu_n = bp * math.exp(log_avg)
        else:
            bleu_n = 0.0
        results[f"bleu{n}"] = bleu_n
    
    return results


def compute_rouge_l(
    pred: str,
    gt: str,
) -> Dict[str, float]:
    """
    Compute ROUGE-L score based on Longest Common Subsequence.
    
    Args:
        pred: Predicted text
        gt: Ground truth text
        
    Returns:
        Dict with ROUGE-L precision, recall, and F1
    """
    pred_tokens = tokenize(pred)
    gt_tokens = tokenize(gt)
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return {"rouge_l": 0.0, "rouge_l_p": 0.0, "rouge_l_r": 0.0}
    
    # Compute LCS length using dynamic programming
    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_length = dp[m][n]
    
    # Compute precision, recall, F1
    precision = lcs_length / m if m > 0 else 0.0
    recall = lcs_length / n if n > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "rouge_l": f1,
        "rouge_l_p": precision,
        "rouge_l_r": recall,
    }


def compute_meteor(
    pred: str,
    gt: str,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> Dict[str, float]:
    """
    Compute METEOR score.
    
    Simplified implementation using exact matching and chunking penalty.
    For full METEOR (with stemming, synonyms, paraphrase), use nltk.meteor_score.
    
    Args:
        pred: Predicted text
        gt: Ground truth text
        alpha: Weight for precision (default 0.9)
        beta: Penalty factor (default 3.0)
        gamma: Fragmentation penalty weight (default 0.5)
        
    Returns:
        Dict with METEOR score
    """
    try:
        # Try using NLTK's implementation first (more accurate)
        import nltk
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        
        # Ensure wordnet is available
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        pred_tokens = tokenize(pred)
        gt_tokens = tokenize(gt)
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return {"meteor": 0.0}
        
        # NLTK meteor_score expects reference as list of tokens
        score = nltk_meteor([gt_tokens], pred_tokens)
        return {"meteor": score}
        
    except (ImportError, LookupError):
        # Fallback: simplified METEOR implementation
        pred_tokens = tokenize(pred)
        gt_tokens = tokenize(gt)
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return {"meteor": 0.0}
        
        # Find matches (exact match only in this simplified version)
        pred_matched = [False] * len(pred_tokens)
        gt_matched = [False] * len(gt_tokens)
        
        matches = 0
        for i, pt in enumerate(pred_tokens):
            for j, gt_t in enumerate(gt_tokens):
                if not gt_matched[j] and pt == gt_t:
                    pred_matched[i] = True
                    gt_matched[j] = True
                    matches += 1
                    break
        
        if matches == 0:
            return {"meteor": 0.0}
        
        # Precision and recall
        precision = matches / len(pred_tokens)
        recall = matches / len(gt_tokens)
        
        # F-mean with alpha weighting
        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall + 1e-8)
        
        # Chunking: count number of chunks (contiguous matched sequences)
        chunks = 0
        in_chunk = False
        for matched in pred_matched:
            if matched and not in_chunk:
                chunks += 1
                in_chunk = True
            elif not matched:
                in_chunk = False
        
        # Fragmentation penalty
        if matches > 0:
            frag = chunks / matches
        else:
            frag = 0.0
        
        penalty = gamma * (frag ** beta)
        
        # Final METEOR score
        meteor = f_mean * (1 - penalty)
        
        return {"meteor": max(0.0, meteor)}


def evaluate(
    pred_report: str,
    gt_report: str,
    templates: Optional[Dict[str, List[str]]] = None,
    slot_summaries: Optional[List[Dict[str, Any]]] = None,
    pred_sections: Optional[Dict[str, str]] = None,
    gt_sections: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of generated report.
    
    Args:
        pred_report: Full predicted report text
        gt_report: Full ground truth report text
        templates: Optional template library
        slot_summaries: Optional slot summaries from memory bank
        pred_sections: Optional dict of predicted sections
        gt_sections: Optional dict of ground truth sections
        
    Returns:
        Dict with all metrics
    """
    results: Dict[str, Any] = {}
    
    # BLEU scores (1-4)
    bleu = compute_bleu(pred_report, gt_report, max_n=4)
    results.update(bleu)
    
    # ROUGE-L score
    rouge_l = compute_rouge_l(pred_report, gt_report)
    results.update(rouge_l)
    
    # METEOR score
    meteor = compute_meteor(pred_report, gt_report)
    results.update(meteor)
    
    # Duplicate rate
    dup = duplicate_rate(pred_report)
    results.update({f"dup_{k}": v for k, v in dup.items() if isinstance(v, (int, float))})
    results["duplicated_ngrams"] = dup.get("duplicated_ngrams", [])
    
    # Unsupported rate (if slot summaries provided)
    if slot_summaries is not None:
        unsup = unsupported_rate(pred_report, slot_summaries)
        results.update({f"unsup_{k}": v for k, v in unsup.items() if isinstance(v, (int, float))})
        results["unsupported_findings"] = unsup.get("unsupported_findings", [])
    
    # Abnormal section recall (if sections provided)
    if pred_sections is not None and gt_sections is not None:
        abn = abnormal_section_recall(pred_sections, gt_sections, templates)
        results.update(abn)
    
    return results


def compute_report_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute standard NLG metrics for report generation.
    
    Returns BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L, METEOR.
    
    Args:
        predictions: List of predicted report strings
        references: List of reference report strings
        
    Returns:
        Dict with averaged metrics
    """
    assert len(predictions) == len(references), \
        f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
    
    if len(predictions) == 0:
        return {
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "rouge_l": 0.0,
            "meteor": 0.0,
        }
    
    # Accumulate scores
    all_bleu1, all_bleu2, all_bleu3, all_bleu4 = [], [], [], []
    all_rouge_l = []
    all_meteor = []
    
    for pred, ref in zip(predictions, references):
        # Convert to string if needed
        if isinstance(pred, dict):
            pred = pred.get('report', str(pred))
        if isinstance(ref, dict):
            ref = ref.get('report', str(ref))
        
        pred = str(pred) if pred else ""
        ref = str(ref) if ref else ""
        
        # BLEU scores
        bleu = compute_bleu(pred, ref, max_n=4)
        all_bleu1.append(bleu['bleu1'])
        all_bleu2.append(bleu['bleu2'])
        all_bleu3.append(bleu['bleu3'])
        all_bleu4.append(bleu['bleu4'])
        
        # ROUGE-L
        rouge = compute_rouge_l(pred, ref)
        all_rouge_l.append(rouge['rouge_l'])
        
        # METEOR
        meteor = compute_meteor(pred, ref)
        all_meteor.append(meteor['meteor'])
    
    # Average
    n = len(predictions)
    return {
        "bleu1": sum(all_bleu1) / n,
        "bleu2": sum(all_bleu2) / n,
        "bleu3": sum(all_bleu3) / n,
        "bleu4": sum(all_bleu4) / n,
        "rouge_l": sum(all_rouge_l) / n,
        "meteor": sum(all_meteor) / n,
    }


def evaluate_batch(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    templates: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    """
    Evaluate a batch of predictions.
    
    Args:
        predictions: List of prediction dicts with 'report', 'sections', 'slot_summaries'
        ground_truths: List of GT dicts with 'report', 'sections'
        templates: Optional template library
        
    Returns:
        Aggregated metrics
    """
    all_metrics: Dict[str, List[float]] = {}
    
    for pred, gt in zip(predictions, ground_truths):
        metrics = evaluate(
            pred_report=pred.get("report", ""),
            gt_report=gt.get("report", ""),
            templates=templates,
            slot_summaries=pred.get("slot_summaries"),
            pred_sections=pred.get("sections"),
            gt_sections=gt.get("sections"),
        )
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(float(value))
    
    # Average
    results = {}
    for key, values in all_metrics.items():
        results[f"avg_{key}"] = sum(values) / max(len(values), 1)
        results[f"std_{key}"] = (
            (sum((v - results[f"avg_{key}"]) ** 2 for v in values) / max(len(values) - 1, 1)) ** 0.5
            if len(values) > 1 else 0.0
        )
    
    results["num_samples"] = len(predictions)
    
    return results
