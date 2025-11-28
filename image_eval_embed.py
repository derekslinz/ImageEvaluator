from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json  # Import json for parsing the response
import logging
import logging.handlers
import os
import pickle
import re
import subprocess
import sys
import time
import warnings
from bisect import bisect_left
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Suppress FutureWarning from timm (used by PyIQA)
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated")

import math
from functools import lru_cache
import numpy as np
import requests
from PIL import Image, ImageFilter, ImageStat
from colorama import Fore, Style
from pydantic import BaseModel, ConfigDict, field_validator
from tqdm import tqdm
from profile_config import (
    PROFILE_CONFIG,
    get_profile,
    get_profile_name,
    PYIQA_BASELINE_STATS,
    PROFILE_SCORE_CENTER,
    PROFILE_SCORE_STD_SCALE,
    get_default_pyiqa_shift,
)

try:
    import piexif  # type: ignore
    import piexif.helper  # type: ignore
    PIEXIF_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    piexif = None  # type: ignore
    PIEXIF_AVAILABLE = False

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False

try:
    import rawpy
except ImportError:
    rawpy = None  # type: ignore

RAWPY_AVAILABLE = rawpy is not None
RAWPY_IMPORT_WARNINGED = False
RAW_EXTENSIONS = {'.nef', '.cr2', '.cr3', '.arw', '.rw2', '.raf', '.orf', '.dng'}

# SSL certificate workaround for corporate proxies/firewalls with SSL inspection
# This allows PyIQA/timm to download model weights through SSL-inspecting proxies
import ssl
import certifi

def _configure_ssl_for_model_downloads():
    """Configure SSL context to handle self-signed certificates in proxy chains."""
    try:
        # Try to use certifi's certificate bundle
        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        # Fall back to unverified context if certifi fails
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    # Patch urllib to use our context
    import urllib.request
    urllib.request.install_opener(
        urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context)
        )
    )

# Apply SSL fix before importing torch/pyiqa
_configure_ssl_for_model_downloads()

try:
    import torch  # type: ignore
    import pyiqa  # type: ignore
    PYIQA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    pyiqa = None  # type: ignore
    PYIQA_AVAILABLE = False


def list_pyiqa_metrics() -> List[str]:
    if not PYIQA_AVAILABLE or pyiqa is None:
        return []
    try:
        if hasattr(pyiqa, "list_models"):
            models = pyiqa.list_models()
            if models:
                return sorted(models)
        return sorted(getattr(pyiqa, "AVAILABLE_METRICS", []))
    except Exception:
        return []


# Increase PIL decompression bomb limit for large legitimate images
Image.MAX_IMAGE_PIXELS = None  # Remove limit entirely (or set to a higher value like 500000000)


# =============================================================================
# Constants
# =============================================================================

# PyIQA configuration
PYIQA_MAX_LONG_EDGE = 1024
DEFAULT_CLIPIQ_MODEL = "clipiqa+_vitL14_512"
DEFAULT_IQA_CALIBRATION_PATH = Path(__file__).with_name("iqa_calibration.json")

# Stock evaluation thresholds
STOCK_MIN_RESOLUTION_MP = 4.0
STOCK_RECOMMENDED_MP = 12.0
STOCK_MIN_DPI = 300
STOCK_DPI_FIXABLE = 240

# Technical metric baseline statistics (mean, std) for outlier detection
# Thresholds are calculated as: mean ± 1.5*std
# These can be recalibrated based on your image corpus
# Based on empirical data from 2400-image GuruShots dataset (Nov 2025)
TECHNICAL_BASELINES = {
    # Sharpness: lower is worse, flag if < mean - 1.5*std
    "sharpness": {"mean": 94.40, "std": 14.25},  # warn < 73.0, critical < 52.0
    # Noise: higher is worse, flag if > mean + 1.5*std
    "noise_score": {"mean": 16.62, "std": 16.65},  # warn > 41.6, critical > 54.1
    # Highlight clipping %: higher is worse, flag if > mean + 1.5*std
    "histogram_clipping_highlights": {"mean": 2.81, "std": 4.86},  # warn > 10.1, critical > 13.8
    # Shadow clipping %: higher is worse, flag if > mean + 1.5*std
    "histogram_clipping_shadows": {"mean": 8.22, "std": 9.32},  # warn > 22.2, critical > 29.2
    # Color cast delta: higher is worse, calibrated to recent corpus (mean≈42, std≈32)
    "color_cast_delta": {"mean": 42.0, "std": 32.0},  # warn ~74, critical ~90
}

# Multiplier for outlier detection
OUTLIER_SIGMA_MULTIPLIER_WARN = 1.5    # Mild outliers (warning)
OUTLIER_SIGMA_MULTIPLIER_CRITICAL = 2.5  # Strong outliers (critical)


def get_technical_threshold(metric_name: str, direction: str = "high", severity: str = "warn") -> float:
    """Calculate threshold for a metric based on baseline statistics.
    
    Args:
        metric_name: Name of the metric in TECHNICAL_BASELINES
        direction: "high" for metrics where higher is worse (noise, clipping)
                   "low" for metrics where lower is worse (sharpness)
        severity: "warn" for mild outliers (1.5σ), "critical" for strong (2.5σ)
    
    Returns:
        Threshold value (mean ± multiplier*std)
    """
    baseline = TECHNICAL_BASELINES.get(metric_name, {"mean": 50.0, "std": 25.0})
    mean = baseline["mean"]
    std = baseline["std"]
    
    multiplier = OUTLIER_SIGMA_MULTIPLIER_CRITICAL if severity == "critical" else OUTLIER_SIGMA_MULTIPLIER_WARN
    
    if direction == "low":
        return mean - (multiplier * std)
    else:  # high
        return mean + (multiplier * std)


# Thresholds computed from empirical baselines
# Sharpness: lower is worse
STOCK_SHARPNESS_WARN = get_technical_threshold("sharpness", "low", "warn")      # ~73.0
STOCK_SHARPNESS_CRITICAL = get_technical_threshold("sharpness", "low", "critical")  # ~58.8
STOCK_SHARPNESS_OPTIMAL = TECHNICAL_BASELINES["sharpness"]["mean"]  # ~94.4

# Noise: higher is worse
STOCK_NOISE_WARN = get_technical_threshold("noise_score", "high", "warn")       # ~41.6
STOCK_NOISE_CRITICAL = get_technical_threshold("noise_score", "high", "critical")  # ~58.2

# Highlight clipping: higher is worse
STOCK_CLIPPING_HIGHLIGHTS_WARN = get_technical_threshold("histogram_clipping_highlights", "high", "warn")  # ~10.1
STOCK_CLIPPING_HIGHLIGHTS_CRITICAL = get_technical_threshold("histogram_clipping_highlights", "high", "critical")  # ~15.0

# Shadow clipping: higher is worse  
STOCK_CLIPPING_SHADOWS_WARN = get_technical_threshold("histogram_clipping_shadows", "high", "warn")  # ~22.2
STOCK_CLIPPING_SHADOWS_CRITICAL = get_technical_threshold("histogram_clipping_shadows", "high", "critical")  # ~31.5

# Color cast (use 1.0σ for warn, 1.5σ for critical to match observed distribution)
COLOR_CAST_WARN = TECHNICAL_BASELINES["color_cast_delta"]["mean"] + TECHNICAL_BASELINES["color_cast_delta"]["std"]
COLOR_CAST_CRITICAL = TECHNICAL_BASELINES["color_cast_delta"]["mean"] + 1.5 * TECHNICAL_BASELINES["color_cast_delta"]["std"]

# Legacy aliases for backward compatibility
STOCK_NOISE_HIGH = STOCK_NOISE_CRITICAL
STOCK_CLIPPING_THRESHOLD = STOCK_CLIPPING_HIGHLIGHTS_WARN

# Default detection threshold for labeling a color cast
COLOR_CAST_THRESHOLD = COLOR_CAST_WARN

# Score validation bounds
SCORE_MIN = 1
SCORE_MAX = 100

# Histogram analysis thresholds
HISTOGRAM_HIGHLIGHT_START = 250
HISTOGRAM_SHADOW_END = 6

STOCK_EVALUATION_PROMPT = """You are a senior stock photography reviewer for major agencies (Adobe Stock, Shutterstock, Getty).
Given the attached image and technical summary, output STRICT JSON with these keys:
{"COMMERCIAL_VIABILITY": int, "TECHNICAL_QUALITY": int, "COMPOSITION_CLARITY": int, "KEYWORD_POTENTIAL": int,
 "RELEASE_CONCERNS": int, "REJECTION_RISKS": int, "OVERALL_STOCK_SCORE": int,
 "RECOMMENDATION": "EXCELLENT|GOOD|MARGINAL|REJECT", "PRIMARY_CATEGORY": "Business|Lifestyle|Nature|Food|Technology|Travel|Other",
 "NOTES": "short sentence", "ISSUES": "comma list"}

Guidelines:
- Scores 0-100 integers. Be conservative; mediocre images cluster 50-65.
- COMMERCIAL_VIABILITY: demand, versatility, timelessness.
- TECHNICAL_QUALITY: exposure, sharpness, noise, DR.
- COMPOSITION_CLARITY: subject readability, copy space, framing.
- KEYWORD_POTENTIAL: count of accurate, high-demand concepts.
- RELEASE_CONCERNS: legal risk (people/property/logos). 100 = safe.
- REJECTION_RISKS: likelihood image passes review. 100 = very likely to pass (noise, focus, clichés).
- OVERALL_STOCK_SCORE = 0.4*TQ + 0.25*CV + 0.2*CC + 0.1*KP + 0.05*(RC+RR)/2, then subtract 5 points for highlight or shadow clipping >12% and 5 points for sharpness<30 (stackable). Clamp 0-100.
- RECOMMENDATION >=85=EXCELLENT, 70-84=GOOD, 50-69=MARGINAL, else REJECT.
- PRIMARY_CATEGORY: pick the best single commercial category.
- NOTES: <=120 chars mention decisive strengths/weaknesses. ISSUES: comma-separated short tags (e.g., "high noise, needs release").

TECHNICAL SUMMARY:
{technical_context}
"""

CONTEXT_PROFILE_MAP = {
    # Preferred contexts (1:1 mapping)
    "studio_photography": "studio_photography",
    "macro_food": "macro_food",
    "macro_nature": "macro_nature",
    "portrait_neutral": "portrait_neutral",
    "portrait_highkey": "portrait_highkey",
    "landscape": "landscape",
    "street_documentary": "street_documentary",
    "sports_action": "sports_action",
    "wildlife_animal": "wildlife_animal",
    "night_artificial_light": "night_artificial_light",
    "night_natural_light": "night_natural_light",
    "architecture_realestate": "architecture_realestate",
    "fineart_creative": "fineart_creative",
    # Legacy/alias contexts
    "concert_night": "night_artificial_light",
    "event_lowlight": "night_artificial_light",
    "travel_story": "street_documentary",
    "travel": "landscape",
    "travel_reportage": "street_documentary",
    "stock_product": "studio_photography",
    "product_catalog": "studio_photography",
    "wildlife": "wildlife_animal",
    "animal": "wildlife_animal",
    "pet": "wildlife_animal",
    "astro": "night_natural_light",
    "astrophotography": "night_natural_light",
    "milky_way": "night_natural_light",
    "aurora": "night_natural_light",
}

# Allowed classification labels (used for truncation handling)
ALLOWED_LABELS = [
    "landscape",
    "portrait_neutral",
    "portrait_highkey",
    "macro_food",
    "macro_nature",
    "street_documentary",
    "sports_action",
    "wildlife_animal",
    "night_artificial_light",
    "night_natural_light",
    "architecture_realestate",
    "studio_photography",
    "fineart_creative",
]


def parse_context_response(raw: str) -> Tuple[str, str]:
    """
    Parse model response to extract context label.
    
    Handles:
    - Exact matches
    - qwen style "category=..." format
    - Truncated responses (e.g., 'wildlife_an' -> 'wildlife_animal')
    
    Returns (label, source), where:
      - label is one of ALLOWED_LABELS or 'unknown'
      - source is 'exact', 'prefix', 'truncated', or 'unknown'
    """
    norm = raw.strip().lower()
    
    # Handle qwen style "category=…"
    if norm.startswith("category="):
        norm = norm.split("=", 1)[1].strip()
    
    # Strip quotes if present
    norm = norm.strip('"\'')
    
    # Exact match
    if norm in ALLOWED_LABELS:
        return norm, "exact"
    
    # Check CONTEXT_PROFILE_MAP for aliases
    if norm in CONTEXT_PROFILE_MAP:
        return CONTEXT_PROFILE_MAP[norm], "alias"
    
    # Truncation / partial match, e.g. 'wildlife_an' or 'wildlife'
    # Check if response is a prefix of a label OR label is a prefix of response
    candidates = [
        lab for lab in ALLOWED_LABELS
        if lab.startswith(norm) or norm.startswith(lab)
    ]
    if len(candidates) == 1:
        return candidates[0], "truncated"
    
    # Check for underscore-separated partial match (e.g., "night_art" -> "night_artificial_light")
    if '_' in norm:
        parts = norm.split('_')
        candidates = [
            lab for lab in ALLOWED_LABELS
            if all(part in lab for part in parts)
        ]
        if len(candidates) == 1:
            return candidates[0], "truncated"
    
    # Still unknown
    return "unknown", "unknown"


def map_context_to_profile(context_label: str) -> str:
    """Map classifier output to one of the 13 defined profile keys."""
    if not context_label:
        return "studio_photography"
    normalized = context_label.strip().lower()
    mapped = CONTEXT_PROFILE_MAP.get(normalized, normalized)
    if mapped not in PROFILE_CONFIG:
        return "studio_photography"
    return mapped


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weight dictionary so it sums to 1.0."""
    positive_items = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(positive_items.values())
    if total <= 0:
        return {k: 1.0 / len(positive_items) for k in positive_items} if positive_items else {}
    return {k: v / total for k, v in positive_items.items()}


# IQA calibration (loaded from JSON at startup)
IQA_CALIBRATION: Dict[str, Dict[str, Any]] = {}


def _baseline_fallback() -> Dict[str, Dict[str, Any]]:
    return {k: {"mu": v.get("mean", 0.0), "sigma": v.get("std", 0.0)} for k, v in PYIQA_BASELINE_STATS.items()}


def load_iqa_calibration(calibration_path: Optional[Union[str, Path]]) -> Dict[str, Dict[str, Any]]:
    """
    Load IQA calibration (mu, sigma, optional sorted values) from JSON.

    Expected JSON structure:
        {
          "iqa_models": {
            "model_name": {
              "mu": float,         # Mean value for z-score normalization
              "sigma": float,      # Standard deviation for z-score normalization
              "sorted_values": [float, ...]  # Optional: sorted calibration values for percentile calculation
            }
          }
        }

    If the file is missing or cannot be parsed, falls back to PYIQA_BASELINE_STATS.

    The optional 'sorted_values' field, if present, is used for percentile calculations.
    """

    def _normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        mu = entry.get("mu", entry.get("mean", 0.0))
        sigma = entry.get("sigma", entry.get("std", 0.0))
        normalized = {
            "mu": float(mu) if mu is not None else 0.0,
            "sigma": float(sigma) if sigma is not None else 0.0,
        }
        if isinstance(entry.get("sorted_values"), list):
            try:
                values = [float(x) for x in entry["sorted_values"]]
                if values == sorted(values):
                    normalized["sorted_values"] = values
                else:
                    logger.warning("IQA calibration: 'sorted_values' for entry %r are not sorted; ignoring.", entry)
            except (TypeError, ValueError):
                logger.warning("IQA calibration: 'sorted_values' for entry %r contain non-numeric values; ignoring.", entry)
        return normalized

    if calibration_path:
        path = Path(calibration_path).expanduser()
    else:
        path = DEFAULT_IQA_CALIBRATION_PATH

    if path.is_file():
        try:
            payload = json.loads(path.read_text())
            models = payload.get("iqa_models", payload)
            calibration = {k: _normalize_entry(v) for k, v in models.items()}
            logger.info("Loaded IQA calibration for %d models from %s", len(calibration), path)
            return calibration
        except Exception as exc:
            logger.warning("Failed to load IQA calibration from %s: %s", path, exc)

    logger.warning("Using fallback IQA calibration baselines; calibration file not found at %s", path)
    return _baseline_fallback()


def _compute_percentile(value: float, sorted_values: Optional[List[float]]) -> Optional[float]:
    if not sorted_values:
        return None
    idx = bisect_left(sorted_values, value)
    percentile = 100.0 * idx / len(sorted_values)
    return max(0.0, min(100.0, percentile))


def z_to_fused_score(z_score: float) -> float:
    z_clamped = max(-3.0, min(3.0, z_score))
    return (z_clamped + 3.0) / 6.0 * 100.0


def compute_metric_z_scores(metric_scores: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, Optional[float]], Dict[str, float]]:
    """Convert calibrated metric scores to z-scores using loaded calibration.

    Returns:
        z_scores: Clamped z-scores (±3)
        percentiles: Empirical percentile estimates when sorted calibration values exist
        fused_scores: Linear mapping of z to [0, 100]
    """

    z_scores: Dict[str, float] = {}
    percentiles: Dict[str, Optional[float]] = {}
    fused_scores: Dict[str, float] = {}
    baseline = _baseline_fallback()
    for key, value in metric_scores.items():
        stats = IQA_CALIBRATION.get(key) or baseline.get(key, {})
        sigma = stats.get("sigma") or 0.0
        if sigma <= 1e-6:
            z = 0.0
        else:
            mu = stats.get("mu", 0.0)
            z = (value - mu) / sigma
        z_clamped = max(-3.0, min(3.0, z))
        z_scores[key] = z_clamped
        fused_scores[key] = z_to_fused_score(z_clamped)
        percentiles[key] = _compute_percentile(value, stats.get("sorted_values"))
    return z_scores, percentiles, fused_scores


def compute_disagreement_z(z_scores: Dict[str, float]) -> float:
    """Return negative spread across model z-scores (used as stability penalty/bonus)."""
    if not z_scores:
        return 0.0
    values = list(z_scores.values())
    disagreement = max(values) - min(values)
    return -disagreement


def categorize_score(score: float) -> str:
    """Map composite score to a human-friendly bucket."""
    if score >= 90:
        return "elite"
    if score >= 85:
        return "award_ready"
    if score >= 75:
        return "portfolio"
    if score >= 65:
        return "solid"
    if score >= 55:
        return "needs_work"
    return "critical"


def compute_profile_composite(profile_key: str, z_scores: Dict[str, float], diff_z: float) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """Return (base_score, composite_z, contributions) before rule penalties."""
    profile_cfg = PROFILE_CONFIG.get(profile_key, PROFILE_CONFIG["studio_photography"])
    weights = normalize_weights(profile_cfg.get("model_weights", {}))
    composite_z = 0.0
    contributions: Dict[str, Dict[str, float]] = {}
    for metric_key, weight in weights.items():
        if weight <= 0:
            continue
        metric_value = diff_z if metric_key == "pyiqa_diff_z" else z_scores.get(metric_key, 0.0)
        contribution = metric_value * weight
        composite_z += contribution
        contributions[metric_key] = {
            "weight": weight,
            "value": metric_value,
            "contribution": contribution,
        }
    base_score = PROFILE_SCORE_CENTER + (composite_z * PROFILE_SCORE_STD_SCALE)
    return base_score, composite_z, contributions


def apply_profile_rules(profile_key: str, technical_metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Apply profile-specific rule penalties/bonuses based on technical metrics."""
    profile_cfg = PROFILE_CONFIG.get(profile_key, PROFILE_CONFIG["studio_photography"])
    rules = profile_cfg.get("rules", {})
    adjustments: List[str] = []
    penalty = 0.0

    sharpness = technical_metrics.get("sharpness")
    sharpness_rules = rules.get("sharpness")
    if isinstance(sharpness, (int, float)) and sharpness_rules:
        if sharpness < sharpness_rules.get("critical_threshold", -float("inf")):
            penalty += sharpness_rules.get("critical_penalty", 0.0)
            adjustments.append("sharpness_critical")
        elif sharpness < sharpness_rules.get("soft_threshold", -float("inf")):
            penalty += sharpness_rules.get("soft_penalty", 0.0)
            adjustments.append("sharpness_soft")

    clipping = technical_metrics.get("histogram_clipping_highlights")
    clipping_rules = rules.get("clipping")
    if isinstance(clipping, (int, float)) and clipping_rules:
        if clipping > clipping_rules.get("hard_pct", float("inf")):
            penalty += clipping_rules.get("hard_penalty", 0.0)
            adjustments.append("clipping_hard")
        elif clipping > clipping_rules.get("warn_pct", float("inf")):
            penalty += clipping_rules.get("warn_penalty", 0.0)
            adjustments.append("clipping_warn")
        elif clipping_rules.get("bonus_pct") is not None and clipping < clipping_rules.get("bonus_pct", 0.0):
            bonus = clipping_rules.get("bonus_points", 0.0)
            if bonus:
                penalty -= bonus
                adjustments.append("clipping_bonus")

    color_cast_label = technical_metrics.get("color_cast")
    color_cast_delta = technical_metrics.get("color_cast_delta", 0.0)
    color_rules = rules.get("color_cast") or {}
    color_threshold = color_rules.get("threshold", COLOR_CAST_WARN)
    color_penalty = color_rules.get("penalty", 0.0)
    if color_cast_label and color_cast_label != "neutral":
        if color_cast_delta >= color_threshold:
            penalty += color_penalty
            adjustments.append("color_cast")

    brightness = technical_metrics.get("brightness")
    brightness_rules = rules.get("brightness")
    if isinstance(brightness, (int, float)) and brightness_rules:
        min_ok = brightness_rules.get("min_ok")
        max_ok = brightness_rules.get("max_ok")
        mild = brightness_rules.get("mild_penalty", 0.0)
        strong = brightness_rules.get("strong_penalty", mild)
        tolerance = 20.0
        if min_ok is not None and brightness < min_ok:
            delta = min_ok - brightness
            apply = strong if delta > tolerance else mild
            if apply:
                penalty += apply
                adjustments.append("brightness_low_strong" if delta > tolerance else "brightness_low")
        elif max_ok is not None and brightness > max_ok:
            delta = brightness - max_ok
            apply = strong if delta > tolerance else mild
            if apply:
                penalty += apply
                adjustments.append("brightness_high_strong" if delta > tolerance else "brightness_high")

    return penalty, adjustments


def build_profile_metadata(
    profile_key: str,
    context_label: str,
    technical_metrics: Dict[str, Any],
    technical_warnings: List[str],
    metric_details: Dict[str, Dict[str, float]],
    z_scores: Dict[str, float],
    percentiles: Dict[str, Optional[float]],
    fused_scores: Dict[str, float],
    diff_z: float,
    composite_z: float,
    base_score: float,
    final_score: float,
    rule_penalty: float,
    rule_notes: List[str],
    contributions: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Assemble metadata payload for profiled PyIQA evaluation."""
    rounded = int(round(max(0.0, min(100.0, final_score))))
    base_rounded = int(round(max(0.0, min(100.0, base_score))))
    profile_name = PROFILE_CONFIG.get(profile_key, {}).get("name")
    context_profile_name = technical_metrics.get("context_profile", profile_key.title())
    display_name = profile_name or context_profile_name or profile_key.replace("_", " ").title()
    category = categorize_score(final_score)

    keyword_candidates = ["pyiqa", profile_key, context_label, category]
    ordered_unique = []
    for kw in keyword_candidates:
        if kw and kw not in ordered_unique:
            ordered_unique.append(kw)
    keywords = ",".join(ordered_unique)

    metric_summary_parts = []
    for metric, detail in metric_details.items():
        z_val = z_scores.get(metric, 0.0)
        percentile = percentiles.get(metric)
        summary = f"{metric}:{detail['calibrated']:.1f} (z={z_val:+.2f})"
        if percentile is not None:
            summary += f" p{percentile:.1f}"
        metric_summary_parts.append(summary)
    metric_summary = "; ".join(metric_summary_parts)

    description_lines = [
        f"Composite profile '{display_name}' score {rounded}/100 ({category}).",
        f"Base model blend before rules: {base_rounded}/100 (penalty {rule_penalty:+.1f}).",
        f"Model breakdown: {metric_summary}",
        f"Disagreement penalty (pyiqa_diff_z): {diff_z:+.2f}",
    ]
    if rule_notes:
        description_lines.append(f"Rule triggers: {', '.join(rule_notes)}")

    extended_metrics = dict(technical_metrics)
    extended_metrics.update({
        "pyiqa_profile": profile_key,
        "pyiqa_metric_details": metric_details,
        "pyiqa_z_scores": z_scores,
        "pyiqa_percentiles": percentiles,
        "pyiqa_fused_scores": fused_scores,
        "pyiqa_composite_z": composite_z,
        "pyiqa_composite_fused_score": z_to_fused_score(composite_z),
        "pyiqa_disagreement_z": diff_z,
        "pyiqa_contributions": contributions,
        "pyiqa_rule_penalty": rule_penalty,
        "pyiqa_rule_notes": rule_notes,
        "pyiqa_category": category,
    })

    metadata = {
        'overall_score': str(rounded),
        'score': str(rounded),
        'technical_score': str(base_rounded),
        'composition_score': str(rounded),
        'lighting_score': str(rounded),
        'creativity_score': str(rounded),
        'title': f"{display_name} IQA composite {rounded}/100",
        'description': " ".join(description_lines),
        'keywords': keywords,
        'technical_metrics': extended_metrics,
        'technical_warnings': technical_warnings,
        'post_process_potential': extended_metrics.get('post_process_potential'),
        'pyiqa_profile': profile_key,
        'pyiqa_category': category,
        'context_label': context_label,
    }
    return metadata


def compute_stock_notes(technical_metrics: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Derive stock-specific notes/fixable issues from technical metrics."""
    notes: List[str] = []
    fixable: List[str] = []

    mp = technical_metrics.get('megapixels')
    if isinstance(mp, (int, float)):
        if mp < STOCK_MIN_RESOLUTION_MP:
            notes.append(f"low resolution ({mp:.1f}MP < {STOCK_MIN_RESOLUTION_MP}MP)")
        elif mp < STOCK_RECOMMENDED_MP:
            notes.append(f"below recommended resolution ({mp:.1f}MP)")

    sharpness = technical_metrics.get('sharpness')
    if isinstance(sharpness, (int, float)):
        if sharpness < STOCK_SHARPNESS_CRITICAL:
            notes.append(f"critically soft (sharpness {sharpness:.1f})")
        elif sharpness < STOCK_SHARPNESS_WARN:
            notes.append(f"soft focus (sharpness {sharpness:.1f})")

    noise_score = technical_metrics.get('noise_score')
    if isinstance(noise_score, (int, float)):
        if noise_score > STOCK_NOISE_CRITICAL:
            notes.append(f"high noise ({noise_score:.1f})")
        elif noise_score > STOCK_NOISE_WARN:
            notes.append(f"moderate noise ({noise_score:.1f})")

    highlights = technical_metrics.get('histogram_clipping_highlights')
    if isinstance(highlights, (int, float)):
        if highlights > STOCK_CLIPPING_HIGHLIGHTS_CRITICAL:
            notes.append(f"severe highlight clipping {highlights:.1f}%")
        elif highlights > STOCK_CLIPPING_HIGHLIGHTS_WARN:
            notes.append(f"highlight clipping {highlights:.1f}%")

    shadows = technical_metrics.get('histogram_clipping_shadows')
    if isinstance(shadows, (int, float)):
        if shadows > STOCK_CLIPPING_SHADOWS_CRITICAL:
            notes.append(f"severe shadow clipping {shadows:.1f}%")
        elif shadows > STOCK_CLIPPING_SHADOWS_WARN:
            notes.append(f"shadow clipping {shadows:.1f}%")

    dpi_x = technical_metrics.get('dpi_x')
    if isinstance(dpi_x, (int, float)) and dpi_x < STOCK_MIN_DPI:
        notes.append(f"low DPI ({dpi_x:.0f})")
        if dpi_x >= STOCK_DPI_FIXABLE:
            fixable.append(f"DPI metadata: {dpi_x:.0f} → {STOCK_MIN_DPI}")

    return notes, fixable

    return notes, fixable


def generate_ollama_metadata(
    image_path: str,
    ollama_host_url: str,
    model: str,
    profile_key: str,
    final_score: float,
    technical_metrics: Dict[str, Any],
    technical_warnings: List[str],
) -> Optional[Dict[str, str]]:
    """Use Ollama to refine title/description/keywords."""
    try:
        encoded_image = encode_image_for_classification(image_path)
    except Exception as exc:
        logger.error(f"Failed to encode image for metadata generation ({image_path}): {exc}")
        return None

    warnings_str = ", ".join(technical_warnings) if technical_warnings else "none"
    summary_parts = [
        f"profile: {profile_key}",
        f"score: {final_score:.1f}/100",
        f"sharpness: {technical_metrics.get('sharpness')}",
        f"brightness: {technical_metrics.get('brightness')}",
        f"clipping_highlights: {technical_metrics.get('histogram_clipping_highlights')}",
        f"clipping_shadows: {technical_metrics.get('histogram_clipping_shadows')}",
        f"noise_score: {technical_metrics.get('noise_score')}",
        f"color_cast: {technical_metrics.get('color_cast')}",
        f"warnings: {warnings_str}",
    ]
    technical_summary = "\n".join(f"- {part}" for part in summary_parts)

    prompt = f"""You craft concise metadata for professional photo libraries.

Use the info below plus the attached image to output STRICT JSON with exactly these keys:
{{"title": "...", "description": "...", "keywords": "..."}}

Rules:
- title <= 60 chars, objective, no sensational language.
- description <= 200 chars, mention key visual elements/lighting without guessing hidden context.
- keywords: lowercase comma-separated list (<=12 unique items), no hashtags.

TECH SUMMARY:
{technical_summary}
"""

    payload = {
        "model": model,
        "stream": False,
        "images": [encoded_image],
        "prompt": prompt,
        "format": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "keywords": {"type": "string"},
            },
            "required": ["title", "description", "keywords"],
        },
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }

    def make_request():
        response = requests.post(ollama_host_url, json=payload, timeout=120)
        response.raise_for_status()
        return response

    try:
        response = retry_with_backoff(make_request)
    except Exception as exc:
        logger.error(f"Ollama metadata generation failed for {image_path}: {exc}")
        return None

    data = response.json()
    raw_json = data.get("response") or data.get("message") or data.get("text")
    if not raw_json:
        logger.error(f"Ollama returned no metadata payload for {image_path}")
        return None

    try:
        metadata_obj = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse Ollama metadata JSON for {image_path}: {exc}")
        return None

    title = str(metadata_obj.get("title", "")).strip()
    description = str(metadata_obj.get("description", "")).strip()
    keywords_raw = str(metadata_obj.get("keywords", "")).lower()
    keywords = []
    for token in keywords_raw.split(','):
        token = token.strip()
        if token and token not in keywords:
            keywords.append(token)
        if len(keywords) >= 12:
            break
    keywords_str = ",".join(keywords)

    result: Dict[str, str] = {}
    if title:
        result["title"] = title[:60]
    if description:
        result["description"] = description[:200]
    if keywords_str:
        result["keywords"] = keywords_str
    return result or None


def build_stock_technical_context(
    technical_metrics: Dict[str, Any],
    technical_warnings: List[str],
) -> str:
    parts = []
    mp = technical_metrics.get('megapixels')
    dims = technical_metrics.get('dimensions')
    if isinstance(mp, (int, float)) and dims:
        parts.append(f"resolution: {mp:.1f}MP ({dims})")
    sharpness = technical_metrics.get('sharpness')
    if isinstance(sharpness, (int, float)):
        parts.append(f"sharpness metric: {sharpness:.1f}")
    noise = technical_metrics.get('noise_score')
    if isinstance(noise, (int, float)):
        parts.append(f"noise score: {noise:.1f}")
    highlights = technical_metrics.get('histogram_clipping_highlights')
    if isinstance(highlights, (int, float)):
        parts.append(f"highlight clipping: {highlights:.1f}%")
    shadows = technical_metrics.get('histogram_clipping_shadows')
    if isinstance(shadows, (int, float)):
        parts.append(f"shadow clipping: {shadows:.1f}%")
    dpi = technical_metrics.get('dpi_x')
    if isinstance(dpi, (int, float)):
        parts.append(f"dpi: {dpi:.0f}")
    stock_notes = technical_metrics.get('stock_notes') or []
    if stock_notes:
        parts.append("notes: " + "; ".join(stock_notes))
    if technical_warnings:
        parts.append("warnings: " + "; ".join(technical_warnings))
    return "\n".join(parts)


def generate_stock_assessment(
    image_path: str,
    ollama_host_url: str,
    model: str,
    technical_metrics: Dict[str, Any],
    technical_warnings: List[str],
) -> Optional[Dict[str, Any]]:
    """Call Ollama to get stock suitability scores."""
    try:
        encoded_image = encode_image_for_classification(image_path)
    except Exception as exc:
        logger.error(f"Failed to encode image for stock assessment ({image_path}): {exc}")
        return None

    technical_context = build_stock_technical_context(technical_metrics, technical_warnings)
    prompt = STOCK_EVALUATION_PROMPT.format(technical_context=technical_context)
    payload = {
        "model": model,
        "stream": False,
        "images": [encoded_image],
        "prompt": prompt,
        "format": {
            "type": "object",
            "properties": {
                "COMMERCIAL_VIABILITY": {"type": ["integer", "string"]},
                "TECHNICAL_QUALITY": {"type": ["integer", "string"]},
                "COMPOSITION_CLARITY": {"type": ["integer", "string"]},
                "KEYWORD_POTENTIAL": {"type": ["integer", "string"]},
                "RELEASE_CONCERNS": {"type": ["integer", "string"]},
                "REJECTION_RISKS": {"type": ["integer", "string"]},
                "OVERALL_STOCK_SCORE": {"type": ["integer", "string"]},
                "RECOMMENDATION": {"type": "string"},
                "PRIMARY_CATEGORY": {"type": "string"},
                "NOTES": {"type": "string"},
                "ISSUES": {"type": "string"},
            },
            "required": [
                "COMMERCIAL_VIABILITY",
                "TECHNICAL_QUALITY",
                "COMPOSITION_CLARITY",
                "KEYWORD_POTENTIAL",
                "RELEASE_CONCERNS",
                "REJECTION_RISKS",
                "OVERALL_STOCK_SCORE",
                "RECOMMENDATION",
                "PRIMARY_CATEGORY",
            ],
        },
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }

    def make_request():
        response = requests.post(ollama_host_url, json=payload, timeout=180)
        response.raise_for_status()
        return response

    try:
        response = retry_with_backoff(make_request)
    except Exception as exc:
        logger.error(f"Ollama stock evaluation failed for {image_path}: {exc}")
        return None

    data = response.json()
    raw_json = data.get("response") or data.get("message") or data.get("text")
    if not raw_json:
        logger.error(f"Ollama returned no stock payload for {image_path}")
        return None

    try:
        stock_obj = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse stock JSON for {image_path}: {exc}")
        return None

    result: Dict[str, Any] = {}
    field_map = {
        "COMMERCIAL_VIABILITY": "stock_commercial_viability",
        "TECHNICAL_QUALITY": "stock_technical_quality",
        "COMPOSITION_CLARITY": "stock_composition_clarity",
        "KEYWORD_POTENTIAL": "stock_keyword_potential",
        "RELEASE_CONCERNS": "stock_release_concerns",
        "REJECTION_RISKS": "stock_rejection_risks",
        "OVERALL_STOCK_SCORE": "stock_overall_score",
    }
    for field, out_key in field_map.items():
        val = stock_obj.get(field)
        if val is None:
            val = stock_obj.get(field.lower())
        try:
            score = float(val)
        except (TypeError, ValueError):
            logger.error(f"Failed to parse {field}='{val}' as float for {image_path}")
            return None
        result[out_key] = int(max(0, min(100, round(score))))

    result["stock_recommendation"] = str(stock_obj.get("RECOMMENDATION", "")).upper()
    result["stock_primary_category"] = str(stock_obj.get("PRIMARY_CATEGORY", "")).title()
    result["stock_llm_notes"] = str(stock_obj.get("NOTES", "")).strip()
    result["stock_llm_issues"] = str(stock_obj.get("ISSUES", "")).strip()
    return result


def resolve_metric_model_name(metric_key: str, args) -> Optional[str]:
    metric_key = metric_key.lower()
    if metric_key == "clipiqa_z":
        return args.pyiqa_model
    if metric_key == "laion_aes_z":
        return "laion_aes"
    if metric_key == "musiq_ava_z":
        return "musiq-ava"
    if metric_key == "maniqa_z":
        return "maniqa"
    if metric_key == "musiq_paq2piq_z":
        return "musiq-paq2piq"
    if metric_key == "pyiqa_diff_z":
        return None
    return None


# Configure basic logging (will be updated based on verbose flag)
# Default to WARNING for quiet operation; setup_logging() adjusts based on CLI flags
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default to suppressing PyIQA chatter unless explicitly enabled
_pyiqa_root_logger = logging.getLogger('pyiqa')
if not any(isinstance(handler, logging.NullHandler) for handler in _pyiqa_root_logger.handlers):
    _pyiqa_root_logger.addHandler(logging.NullHandler())
_pyiqa_root_logger.setLevel(logging.WARNING)
_pyiqa_root_logger.propagate = False





def load_image_tensor_with_max_edge(image_path: str, max_long_edge: int) -> Optional["torch.Tensor"]:
    if torch is None:
        return None
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            long_edge = max(w, h)
            if long_edge > max_long_edge and max_long_edge > 0:
                scale = max_long_edge / float(long_edge)
                new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                img = img.resize(new_size, Image.Resampling.BILINEAR)  # BILINEAR for speed
            np_img = np.asarray(img).astype('float32') / 255.0
    except Exception as exc:
        logger.error(f'Failed to load image for PyIQA preprocessing {image_path}: {exc}')
        return None
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
    return tensor


class TensorCache:
    """LRU cache for image tensors to avoid reloading for each model."""
    
    def __init__(self, max_size: int = 32, device: Optional[str] = None):
        self.max_size = max_size
        self.device = device or ('cuda' if torch and torch.cuda.is_available() else 'cpu')
        self._cache: Dict[str, "torch.Tensor"] = {}
        self._order: List[str] = []  # LRU order
    
    def get(self, image_path: str, max_long_edge: int) -> Optional["torch.Tensor"]:
        """Get tensor from cache or load from disk."""
        cache_key = f"{image_path}:{max_long_edge}"
        
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._order.remove(cache_key)
            self._order.append(cache_key)
            return self._cache[cache_key]
        
        # Load and cache
        tensor = load_image_tensor_with_max_edge(image_path, max_long_edge)
        if tensor is None:
            return None
        
        # Move to device
        tensor = tensor.to(self.device)
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = self._order.pop(0)
            old_tensor = self._cache.pop(oldest_key, None)
            if old_tensor is not None:
                del old_tensor
        
        self._cache[cache_key] = tensor
        self._order.append(cache_key)
        return tensor
    
    def clear(self):
        """Clear all cached tensors."""
        for tensor in self._cache.values():
            del tensor
        self._cache.clear()
        self._order.clear()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def preload(self, image_paths: List[str], max_long_edge: int):
        """Preload multiple images into cache."""
        for path in image_paths:
            self.get(path, max_long_edge)


class PyIqaScorer:
    """Wrapper around PyIQA metrics for local inference."""

    def __init__(self, model_name: str = 'musiq', device: Optional[str] = None,
                 score_shift: float = 0.0, scale_factor: Optional[float] = None,
                 max_long_edge: int = PYIQA_MAX_LONG_EDGE):
        if not PYIQA_AVAILABLE:
            raise RuntimeError("PyIQA backend is unavailable. Install torch and pyiqa to use this scoring engine.")
        available = list_pyiqa_metrics()
        canonical_name = model_name
        if available:
            lower_lookup = {name.lower(): name for name in available}
            match = lower_lookup.get(model_name.lower())
            if not match:
                raise RuntimeError(
                    f"PyIQA metric '{model_name}' not available. Installed metrics: {', '.join(available)}"
                )
            canonical_name = match
        self.model_name = canonical_name
        desired_device = device or ('cuda' if torch and torch.cuda.is_available() else 'cpu')
        if torch is None:
            raise RuntimeError("PyIQA requires PyTorch, but it could not be imported.")
        if desired_device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("Requested CUDA for PyIQA but no GPU detected; falling back to CPU.")
            desired_device = 'cpu'
        self.device = desired_device
        self.metric = pyiqa.create_metric(canonical_name, device=self.device)
        self.score_shift = score_shift
        self.scale_factor = scale_factor
        self.max_long_edge = max_long_edge
        self.use_amp = bool(torch) and self.device.startswith('cuda')

    def score_batch(self, image_paths: List[str], tensor_cache: Optional[TensorCache] = None) -> Dict[str, float]:
        """Score a batch of images, optionally using cached tensors."""
        results: Dict[str, float] = {}
        if self.use_amp and torch is not None:
            def autocast_ctx():
                return torch.amp.autocast('cuda')
        else:
            def autocast_ctx():
                return nullcontext()
        assert torch is not None, "PyTorch is required for PyIQA scoring but was not imported."
        with torch.no_grad():
            for image_path in image_paths:
                image_tensor = None
                score = None
                try:
                    # Use tensor cache if provided
                    if tensor_cache is not None:
                        image_tensor = tensor_cache.get(image_path, self.max_long_edge)
                    else:
                        image_tensor = load_image_tensor_with_max_edge(image_path, self.max_long_edge)
                        if image_tensor is not None:
                            image_tensor = image_tensor.to(self.device)
                    
                    if image_tensor is not None:
                        with autocast_ctx():
                            score = self.metric(image_tensor)
                    else:
                        with autocast_ctx():
                            score = self.metric(image_path)
                    value = float(score.item()) if hasattr(score, "item") else float(score)
                    results[image_path] = value
                except Exception as exc:
                    logger.error(f"PyIQA scoring failed for {image_path}: {exc}")
                finally:
                    # Only delete if we loaded it ourselves (not from cache)
                    if tensor_cache is None and image_tensor is not None:
                        del image_tensor
                    if score is not None:
                        del score
        return results

    def score_tensors_stacked(self, tensors: List["torch.Tensor"], image_paths: List[str]) -> Dict[str, float]:
        """Score pre-loaded tensors using true batch stacking for parallel inference.
        
        Note: Only works if all tensors have the same spatial dimensions.
        Falls back to sequential if shapes differ.
        """
        if not tensors or torch is None:
            return {}
        
        results: Dict[str, float] = {}
        
        if self.use_amp:
            def autocast_ctx():
                return torch.amp.autocast('cuda')
        else:
            def autocast_ctx():
                return nullcontext()
        
        # Check if all tensors have same shape for true batching
        shapes = [t.shape[1:] for t in tensors]  # Ignore batch dim
        can_stack = len(set(shapes)) == 1
        
        with torch.no_grad():
            if can_stack and len(tensors) > 1:
                # True batch inference - stack all tensors
                try:
                    stacked = torch.cat(tensors, dim=0)  # [N, C, H, W]
                    with autocast_ctx():
                        scores = self.metric(stacked)
                    # Handle different output formats
                    if hasattr(scores, 'shape') and len(scores.shape) > 0:
                        for i, path in enumerate(image_paths):
                            results[path] = float(scores[i].item() if hasattr(scores[i], 'item') else scores[i])
                    else:
                        # Single score returned - shouldn't happen but handle it
                        results[image_paths[0]] = float(scores.item() if hasattr(scores, 'item') else scores)
                    del stacked
                except Exception as exc:
                    logger.debug(f"Stacked batch failed, falling back to sequential: {exc}")
                    # Fall back to sequential
                    for tensor, path in zip(tensors, image_paths):
                        try:
                            with autocast_ctx():
                                score = self.metric(tensor)
                            results[path] = float(score.item() if hasattr(score, 'item') else score)
                        except Exception as inner_exc:
                            logger.error(f"PyIQA scoring failed for {path}: {inner_exc}")
            else:
                # Sequential inference (different shapes or single image)
                for tensor, path in zip(tensors, image_paths):
                    try:
                        with autocast_ctx():
                            score = self.metric(tensor)
                        results[path] = float(score.item() if hasattr(score, 'item') else score)
                    except Exception as exc:
                        logger.error(f"PyIQA scoring failed for {path}: {exc}")
        
        return results

    def score_single(self, image_path: str) -> float:
        _, _, calibrated = self.score_image(image_path)
        return calibrated

    def convert_score(self, raw_score: float) -> float:
        if self.scale_factor is not None:
            return raw_score * self.scale_factor
        if 0.0 <= raw_score <= 1.0:
            return raw_score * 100.0
        if 0.0 <= raw_score <= 10.0:
            return raw_score * 10.0
        return raw_score

    def _apply_shift(self, scaled_score: float) -> float:
        shifted = scaled_score + self.score_shift
        return max(0.0, min(100.0, shifted))

    def score_image(self, image_path: str) -> Tuple[float, float, float]:
        """Return (raw, scaled, calibrated) scores for a single image."""
        scores = self.score_batch([image_path])
        if image_path not in scores:
            raise RuntimeError(f"PyIQA returned no score for {image_path}")
        raw_score = scores[image_path]
        scaled_score = self.convert_score(raw_score)
        calibrated = self._apply_shift(scaled_score)
        return raw_score, scaled_score, calibrated


class PyiqaManager:
    """Lazily instantiate and reuse PyIQA scorers for multiple metrics."""

    def __init__(
        self,
        device: Optional[str],
        scale_factor: Optional[float],
        shift_overrides: Optional[Dict[str, float]] = None,
        max_cached_models: int = 1,
        tensor_cache_size: int = 32,
    ):
        self.device = device or ('cuda' if torch and torch.cuda.is_available() else 'cpu')
        self.scale_factor = scale_factor
        self.shift_overrides = {k.lower(): v for k, v in (shift_overrides or {}).items()}
        self.max_cached_models = max(1, int(max_cached_models))
        self.scorers: Dict[str, PyIqaScorer] = {}
        self.scorer_usage: List[str] = []
        # Tensor cache to avoid reloading images for each model
        self.tensor_cache = TensorCache(max_size=tensor_cache_size, device=self.device)

    def _mark_used(self, model_name: str):
        if model_name in self.scorer_usage:
            self.scorer_usage.remove(model_name)
        self.scorer_usage.append(model_name)

    def _evict_if_needed(self):
        while len(self.scorers) >= self.max_cached_models:
            oldest = self.scorer_usage.pop(0)
            scorer = self.scorers.pop(oldest, None)
            if scorer:
                del scorer
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_scorer(self, model_name: str) -> PyIqaScorer:
        if model_name not in self.scorers:
            self._evict_if_needed()
            shift = self.shift_overrides.get(model_name.lower(), get_default_pyiqa_shift(model_name))
            self.scorers[model_name] = PyIqaScorer(
                model_name=model_name,
                device=self.device,
                score_shift=shift,
                scale_factor=self.scale_factor,
                max_long_edge=PYIQA_MAX_LONG_EDGE,
            )
        self._mark_used(model_name)
        return self.scorers[model_name]

    def score_metrics(self, image_path: str, metric_keys: List[str], args) -> Dict[str, Dict[str, float]]:
        """Score a single image across all requested metrics, reusing cached tensor."""
        scores: Dict[str, Dict[str, float]] = {}
        
        # Pre-load tensor into cache once for all models
        tensor = self.tensor_cache.get(image_path, PYIQA_MAX_LONG_EDGE)
        
        for metric_key in metric_keys:
            if metric_key == "pyiqa_diff_z":
                continue
            model_name = resolve_metric_model_name(metric_key, args)
            if not model_name:
                continue
            scorer = self.get_scorer(model_name)
            try:
                # Use cached tensor via score_batch with tensor_cache
                batch_scores = scorer.score_batch([image_path], tensor_cache=self.tensor_cache)
                raw = batch_scores.get(image_path)
                if raw is None:
                    continue
                scaled = scorer.convert_score(raw)
                calibrated = scorer._apply_shift(scaled)
            except RuntimeError as exc:
                logger.error(f"PyIQA metric {model_name} failed for {image_path}: {exc}")
                if "out of memory" in str(exc).lower() and torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.tensor_cache.clear()
                continue
            scores[metric_key] = {
                "raw": float(raw),
                "scaled": float(scaled),
                "calibrated": float(calibrated),
                "model_name": model_name,
            }
        return scores

    def score_batch(
        self,
        image_paths: List[str],
        metric_keys: List[str],
        args,
        batch_size: int = 8
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Score multiple images in batches for improved efficiency.
        
        Uses tensor caching (load once, score with all models) and true batch
        stacking for parallel GPU inference when possible.
        
        Args:
            image_paths: List of image paths to score
            metric_keys: List of metric keys to compute (e.g., 'clipiqa_z', 'maniqa_z')
            args: CLI arguments containing model configuration
            batch_size: Number of images to process per batch
            
        Returns:
            Dict mapping image_path -> metric_key -> {raw, scaled, calibrated, model_name}
        """
        # Initialize results dict for all images
        all_results: Dict[str, Dict[str, Dict[str, float]]] = {
            path: {} for path in image_paths
        }
        
        # Group by model to minimize model switching
        model_to_metrics: Dict[str, List[str]] = {}
        for metric_key in metric_keys:
            if metric_key == "pyiqa_diff_z":
                continue
            model_name = resolve_metric_model_name(metric_key, args)
            if model_name:
                if model_name not in model_to_metrics:
                    model_to_metrics[model_name] = []
                model_to_metrics[model_name].append(metric_key)
        
        # Pre-load all tensors into cache for this batch
        for path in image_paths:
            self.tensor_cache.get(path, PYIQA_MAX_LONG_EDGE)
        
        # Process each model's metrics
        for model_name, metrics in model_to_metrics.items():
            scorer = self.get_scorer(model_name)
            
            # Process images in batches with true stacking
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                # Collect pre-loaded tensors for stacked batch inference
                batch_tensors = []
                valid_paths = []
                for path in batch_paths:
                    tensor = self.tensor_cache.get(path, PYIQA_MAX_LONG_EDGE)
                    if tensor is not None:
                        batch_tensors.append(tensor)
                        valid_paths.append(path)
                
                if not batch_tensors:
                    continue
                
                try:
                    # Use true batch stacking for parallel inference
                    batch_scores = scorer.score_tensors_stacked(batch_tensors, valid_paths)
                except RuntimeError as exc:
                    logger.error(f"PyIQA batch scoring failed for model {model_name}: {exc}")
                    if "out of memory" in str(exc).lower() and torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.tensor_cache.clear()
                        # Retry with tensor cache (sequential)
                        logger.info(f"Retrying sequentially after OOM")
                        for path in batch_paths:
                            try:
                                single_scores = scorer.score_batch([path], tensor_cache=self.tensor_cache)
                                raw = single_scores.get(path)
                                if raw is not None:
                                    scaled = scorer.convert_score(raw)
                                    calibrated = scorer._apply_shift(scaled)
                                    for metric_key in metrics:
                                        all_results[path][metric_key] = {
                                            "raw": float(raw),
                                            "scaled": float(scaled),
                                            "calibrated": float(calibrated),
                                            "model_name": model_name,
                                        }
                            except Exception as inner_exc:
                                logger.error(f"PyIQA single-image scoring failed for {path}: {inner_exc}")
                    continue
                
                # Process batch results
                for path in valid_paths:
                    raw = batch_scores.get(path)
                    if raw is None:
                        continue
                    scaled = scorer.convert_score(raw)
                    calibrated = scorer._apply_shift(scaled)
                    for metric_key in metrics:
                        all_results[path][metric_key] = {
                            "raw": float(raw),
                            "scaled": float(scaled),
                            "calibrated": float(calibrated),
                            "model_name": model_name,
                        }
        
        # Clear tensor cache after batch to free memory
        self.tensor_cache.clear()
        
        return all_results


def _is_raw_image(image_path: str) -> bool:
    """Check if file is a RAW image based on extension."""
    return Path(image_path).suffix.lower() in RAW_EXTENSIONS

def _warn_rawpy_missing():
    """Log a one-time warning when rawpy is needed but not installed."""
    global RAWPY_IMPORT_WARNINGED
    if not RAWPY_IMPORT_WARNINGED:
        logger.warning(
            "rawpy is not installed; RAW files (e.g., NEF, CR2, ARW) will be skipped unless rawpy is available."
        )
        RAWPY_IMPORT_WARNINGED = True

def _get_int_env(var_name: str, fallback: int) -> int:
    """Read an integer environment variable, falling back if unset/invalid."""
    value = os.environ.get(var_name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Ignoring invalid value for {var_name}: {value!r}. Using {fallback}.")
        return fallback

# Configuration constants
DEFAULT_IMAGE_FOLDER = os.environ.get("IMAGE_EVAL_DEFAULT_FOLDER", str(Path.cwd()))
DEFAULT_OLLAMA_URL = os.environ.get("IMAGE_EVAL_OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_WORKER_COUNT = _get_int_env("IMAGE_EVAL_WORKERS", 4)
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds
DEFAULT_MODEL = "qwen3-vl:8b"
CACHE_DIR = ".image_eval_cache"
CACHE_VERSION = "v1"
# Technical analysis constants - now using statistical thresholds from TECHNICAL_BASELINES
# COLOR_CAST_WARN and COLOR_CAST_CRITICAL are defined at module level with other thresholds
HISTOGRAM_HIGHLIGHT_RANGE = (250, 256)  # Histogram bins considered as highlights
HISTOGRAM_SHADOW_RANGE = (0, 6)  # Histogram bins considered as shadows
MIN_LONG_EDGE = 850  # Minimum pixels required on the long edge


class ImageResolutionTooSmallError(RuntimeError):
    """Raised when an image fails the minimum long-edge requirement."""


def _extract_pil_exif_metadata(image_path: str) -> Dict[str, Union[str, int, None]]:
    """Extract EXIF metadata using PIL - works for both RAW (NEF/CR2) and JPEG/PNG.
    
    Note: Lens information may not be available for all RAW files as it's often stored in
    manufacturer-specific MakerNote tags that PIL doesn't parse. Use exiftool for complete
    lens metadata extraction if needed.
    """
    metadata: Dict[str, Union[str, int, None]] = {}
    
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return metadata
            
            # For RAW files, EXIF data is in a separate IFD (tag 0x8769)
            # Try to get the EXIF IFD for more complete metadata
            exif_ifd = None
            try:
                if hasattr(exif, 'get_ifd'):
                    exif_ifd = exif.get_ifd(0x8769)
            except Exception:
                pass
            
            # Use EXIF IFD if available, otherwise fall back to main EXIF
            data_source = exif_ifd if exif_ifd else exif
            
            # ISO - tag 34855
            iso = data_source.get(34855)
            if iso:
                metadata['iso'] = int(iso)
            
            # Aperture (FNumber) - tag 33437
            f_number = data_source.get(33437)
            if f_number:
                if isinstance(f_number, tuple) and len(f_number) == 2:
                    metadata['aperture'] = f"f/{f_number[0]/f_number[1]:.1f}"
                elif hasattr(f_number, '__float__'):
                    metadata['aperture'] = f"f/{float(f_number):.1f}"
            
            # Shutter speed (ExposureTime) - tag 33434
            exposure_time = data_source.get(33434)
            if exposure_time:
                if isinstance(exposure_time, tuple) and len(exposure_time) == 2:
                    if exposure_time[0] == 1:
                        metadata['shutter_speed'] = f"1/{exposure_time[1]}s"
                    else:
                        metadata['shutter_speed'] = f"{exposure_time[0]/exposure_time[1]:.3f}s"
                elif hasattr(exposure_time, '__float__'):
                    exp_val = float(exposure_time)
                    if exp_val < 1:
                        metadata['shutter_speed'] = f"1/{int(1/exp_val)}s"
                    else:
                        metadata['shutter_speed'] = f"{exp_val:.3f}s"
            
            # Focal length - tag 37386
            focal_length = data_source.get(37386)
            if focal_length:
                if isinstance(focal_length, tuple) and len(focal_length) == 2:
                    metadata['focal_length'] = f"{focal_length[0]/focal_length[1]:.0f}mm"
                elif hasattr(focal_length, '__float__'):
                    metadata['focal_length'] = f"{float(focal_length):.0f}mm"
            
            # Camera make - tag 271 (from main EXIF, not IFD)
            camera_make = exif.get(271)
            if camera_make:
                metadata['camera_make'] = str(camera_make).strip()
            
            # Camera model - tag 272 (from main EXIF, not IFD)
            camera_model = exif.get(272)
            if camera_model:
                metadata['camera_model'] = str(camera_model).strip()
            
            # Lens model - tag 42036
            lens_model = data_source.get(42036)
            if lens_model:
                metadata['lens_model'] = str(lens_model).strip()
            
            if metadata:
                logger.debug(f"Extracted {len(metadata)} EXIF fields from {image_path}")
            else:
                logger.debug(f"No EXIF metadata found in {image_path}")
            
            return metadata
            
    except Exception as e:
        logger.debug(f"PIL EXIF extraction failed for {image_path}: {e}")
        return {}



def encode_image_for_classification(image_path: str) -> str:
    """Encode image as base64 JPEG for classification. Converts RAW to preview first."""
    ext = Path(image_path).suffix.lower()
    
    # For RAW files, extract embedded JPEG preview or render small version
    if ext in RAW_EXTENSIONS:
        if not RAWPY_AVAILABLE:
            _warn_rawpy_missing()
            # Fallback: try to read as regular image (may work for some RAW formats)
            try:
                with Image.open(image_path) as img:
                    buffer = io.BytesIO()
                    # Resize to reasonable size for classification
                    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    img.convert('RGB').save(buffer, format='JPEG', quality=85)
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as e:
                logger.warning(f"Could not encode RAW file {image_path} for classification: {e}")
                raise
        
        try:
            # Use rawpy to get embedded JPEG preview or render small version
            raw = rawpy.imread(image_path)  # type: ignore
            try:
                # Try to extract embedded JPEG preview (fast)
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:  # type: ignore
                        return base64.b64encode(thumb.data).decode('utf-8')
                except Exception:
                    pass  # No thumbnail, will render below
                
                # Render a small preview (slower but works)
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    half_size=True  # Render at half resolution for speed
                )
                img = Image.fromarray(rgb.astype('uint8'))
                # Further resize if still large
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            finally:
                raw.close()
        except Exception as e:
            logger.warning(f"Could not process RAW file {image_path} for classification: {e}")
            raise
    else:
        # For standard formats, resize and encode as JPEG
        try:
            with Image.open(image_path) as img:
                buffer = io.BytesIO()
                # Resize to reasonable size for classification
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                img.convert('RGB').save(buffer, format='JPEG', quality=85)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.warning(f"Could not encode image {image_path} for classification: {e}")
            raise


def _downsample_image(img: Image.Image, max_long_edge: int) -> Image.Image:
    """Downsample image so longest edge is at most max_long_edge pixels."""
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return img
    
    scale = max_long_edge / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


# Max resolution for technical analysis (long edge in pixels)
# 2048 is a good balance between speed and accuracy for sharpness/noise detection.
ANALYSIS_MAX_LONG_EDGE = 2048


@contextmanager
def open_image_for_analysis(image_path: str, max_long_edge: int = ANALYSIS_MAX_LONG_EDGE):
    """Context manager to open image files (including RAW) for analysis.
    
    Yields a PIL Image object downsampled to max_long_edge for faster processing.
    For RAW files, uses rawpy for full-quality demosaic then downsamples.
    
    Args:
        image_path: Path to image file
        max_long_edge: Maximum pixels for longest edge (default 2048)
    """
    ext = Path(image_path).suffix.lower()
    if ext in RAW_EXTENSIONS:
        if not RAWPY_AVAILABLE:
            _warn_rawpy_missing()
            raise RuntimeError("rawpy is required to process RAW files")
        raw = rawpy.imread(image_path)  # type: ignore
        try:
            # Full demosaic for accurate sharpness detection
            # Use sRGB gamma for proper tonal distribution (critical for Laplacian sharpness)
            # Linear gamma (1.0, 1.0) produces very dark images with no edge contrast
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                gamma=(2.222, 4.5)  # sRGB-like gamma for proper contrast
            )
        finally:
            raw.close()

        img = Image.fromarray(rgb.astype('uint8'))
        # Downsample to target resolution
        img = _downsample_image(img, max_long_edge)
        try:
            yield img
        finally:
            img.close()
    else:
        with Image.open(image_path) as orig_img:
            img = _downsample_image(orig_img, max_long_edge)
            try:
                yield img
            finally:
                if img is not orig_img:
                    img.close()


@lru_cache(maxsize=16)
def _center_weight_mask(height: int, width: int, sigma_ratio: float = 0.35) -> np.ndarray:
    """Return a normalized center-weighted mask favoring likely subject areas."""

    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    sigma = sigma_ratio
    gaussian = np.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
    normalized = gaussian / max(gaussian.sum(), 1e-8)
    return normalized


def _compute_perceptual_sharpness(laplacian_map: np.ndarray) -> float:
    """Estimate perceptual sharpness with subject weighting and DOF tolerance."""

    if laplacian_map.size == 0:
        return 0.0

    lap_float = laplacian_map.astype(np.float32)
    focus_abs = np.abs(lap_float)

    height, width = focus_abs.shape
    megapixels = (height * width) / 1_000_000

    laplacian_var = float(focus_abs.var())
    scale_factor = math.sqrt(max(megapixels, 0.1))
    base_sharpness = math.sqrt(max(0.0, laplacian_var / scale_factor)) * 5.0

    edge_95 = float(np.percentile(focus_abs, 95.0))
    strength_norm = focus_abs / max(edge_95, focus_abs.mean() + 1e-6)
    strength_norm = np.clip(strength_norm, 0.0, 2.0)

    center_mask = _center_weight_mask(height, width)
    subject_focus = float(np.sum(strength_norm * center_mask))

    edge_threshold = edge_95 * 0.5 if edge_95 > 0 else focus_abs.mean()
    coverage_ratio = float(np.mean(focus_abs > max(edge_threshold, 1e-6)))
    coverage_score = min(1.0, coverage_ratio / 0.2)

    perceptual = (
        0.6 * base_sharpness
        + 25.0 * min(subject_focus, 1.5) / 1.5
        + 15.0 * coverage_score
    )

    return float(max(0.0, min(100.0, perceptual)))


def _compute_sharpness_noise_fallback(gray_array: np.ndarray) -> Tuple[float, float, float]:
    """Estimate sharpness and noise without relying on OpenCV."""

    gray_float = gray_array.astype(np.float32)

    # Discrete Laplacian to approximate focus/edge strength
    padded = np.pad(gray_float, 1, mode='edge')
    laplacian = (
        -4 * gray_float
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
    )
    sharpness = _compute_perceptual_sharpness(laplacian)

    # Basic noise estimate using blurred residuals in flat regions
    h, w = gray_array.shape
    target_long = 2048
    scale = target_long / float(max(h, w))
    if scale < 1.0:
        gray_small = np.array(
            Image.fromarray(gray_array).resize(
                (int(w * scale), int(h * scale)), resample=Image.BILINEAR
            )
        )
    else:
        gray_small = gray_array

    gray_f = gray_small.astype(np.float32) / 255.0
    blurred = np.array(
        Image.fromarray(gray_small).filter(ImageFilter.GaussianBlur(radius=1.0)),
        dtype=np.float32,
    ) / 255.0
    residual = gray_f - blurred

    gx = np.zeros_like(gray_f)
    gy = np.zeros_like(gray_f)
    gx[:, 1:] = np.diff(gray_f, axis=1)
    gy[1:, :] = np.diff(gray_f, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)

    edge_thresh = np.percentile(grad_mag, 75.0)
    flat_mask = grad_mag < edge_thresh

    p_low, p_high = np.percentile(gray_f, [5.0, 95.0])
    luminance_mask = (gray_f > p_low) & (gray_f < p_high)

    final_mask = flat_mask & luminance_mask
    flat_residuals = residual[final_mask]
    min_pixels = int(0.01 * residual.size)
    if flat_residuals.size < min_pixels:
        flat_residuals = residual.flatten()
        logger.debug("Noise estimation using full image (insufficient flat regions)")

    med = float(np.median(flat_residuals))
    mad = float(np.median(np.abs(flat_residuals - med)))
    if mad < 1e-6:
        sigma_noise = 0.0
    else:
        sigma_noise = 1.4826 * mad

    p1, p99 = np.percentile(gray_f, [1.0, 99.0])
    dynamic_range = max(float(p99 - p1), 1e-6)
    relative_noise = sigma_noise / dynamic_range

    REL_NOISE_MIN = 0.002
    REL_NOISE_MAX = 0.04
    rn_clipped = min(max(relative_noise, REL_NOISE_MIN), REL_NOISE_MAX)
    noise_score = (rn_clipped - REL_NOISE_MIN) / (REL_NOISE_MAX - REL_NOISE_MIN) * 100.0

    return sharpness, float(sigma_noise), float(noise_score)


def validate_image(image_path: str) -> bool:
    """Validate if file is a valid, non-corrupted image."""
    ext = Path(image_path).suffix.lower()
    if ext in RAW_EXTENSIONS:
        if not RAWPY_AVAILABLE:
            _warn_rawpy_missing()
            return False
        try:
            raw = rawpy.imread(image_path)  # type: ignore
            raw.close()
            return True
        except Exception as e:
            logger.warning(f"Invalid or corrupted RAW image {image_path}: {e}")
            return False

    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        # Re-open for actual check (verify closes the file)
        with Image.open(image_path) as img:
            img.load()  # Try to load the image data
        return True
    except Exception as e:
        logger.warning(f"Invalid or corrupted image {image_path}: {e}")
        return False


def validate_score(score_input) -> Optional[int]:
    """Extract and validate score is between SCORE_MIN-SCORE_MAX. Accepts int, str, or other types."""
    # Handle integer input directly
    if isinstance(score_input, int):
        if SCORE_MIN <= score_input <= SCORE_MAX:
            return score_input
        else:
            logger.warning(f"Integer score out of range: {score_input}")
            return None
    
    # Handle string or other input
    try:
        # Try direct conversion first
        score = int(score_input)
        if SCORE_MIN <= score <= SCORE_MAX:
            return score
    except (ValueError, TypeError):
        pass
    
    # Try to extract first two-digit or single-digit number from string
    # Look for standalone numbers (with word boundaries)
    matches = re.findall(r'\b(\d{1,3})\b', str(score_input))
    for match in matches:
        score = int(match)
        if SCORE_MIN <= score <= SCORE_MAX:
            return score
    
    # Fallback: try to find any number
    match = re.search(r'\d+', str(score_input))
    if match:
        score = int(match.group())
        if SCORE_MIN <= score <= SCORE_MAX:
            return score
    
    # Truncate long error messages
    score_preview = str(score_input)[:100] + '...' if len(str(score_input)) > 100 else str(score_input)
    logger.warning(f"Invalid score (no valid number {SCORE_MIN}-{SCORE_MAX} found): {score_preview}")
    return None


def retry_with_backoff(func, max_retries=MAX_RETRIES, base_delay=RETRY_DELAY_BASE):
    """Retry function with exponential backoff, skipping permanent errors."""
    # Errors that should not be retried (permanent failures)
    permanent_error_types = (FileNotFoundError, PermissionError, ValueError, TypeError, KeyError)
    
    for attempt in range(max_retries):
        try:
            return func()
        except permanent_error_types as e:
            # Don't retry permanent errors
            logger.error(f"Permanent error, not retrying: {e}")
            raise
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with current delay calculation
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
            time.sleep(delay)
    return None


def extract_exif_metadata(image_path: str) -> Dict[str, Union[str, int, None]]:
    """Extract technical metadata from EXIF data using PIL (works for RAW and standard formats)."""
    metadata: Dict[str, Union[str, int, None]] = {
        'iso': None,
        'aperture': None,
        'shutter_speed': None,
        'focal_length': None,
        'camera_make': None,
        'camera_model': None,
        'lens_model': None
    }
    
    # Use PIL for EXIF extraction - it works for both RAW and standard formats
    metadata.update(_extract_pil_exif_metadata(image_path))
    return metadata


# Lightweight classification prompt for 12 contexts
IMAGE_CONTEXT_CLASSIFIER_PROMPT = """You are NOT a chat assistant.
You MUST NOT explain, greet, apologize, justify, describe, or respond conversationally.

Your ONLY job is to classify the image into EXACTLY ONE photography category from the list below.
You are a strict image classifier. Choose EXACTLY ONE category from the list.

Global portrait rule:
- Portraits may be of humans OR animals.
- Only call an image a portrait if the main subject is framed between:
  - most zoomed-in: the whole head (not just an eye or lips), and
  - least zoomed-in: the full body with reasonable surrounding context.
- If the subject is tiny in the frame (small figure in a big scene), do NOT use a portrait class.
  Use landscape, wildlife_animal, street_documentary, etc. instead.
- If only a small part of the subject (eye, mouth, fur patch, feathers detail) is shown, treat it as macro/close-up, not portrait.

Categories:
- landscape: wide outdoor scenes, natural environments, scenery, horizons, seascapes, skies
- wildlife_animal: animals as the main subject (birds, mammals, etc.), not extreme close-up of a small detail
- portrait_neutral: human or animal portraits with neutral/normal lighting; subject framed from head/shoulders up to full body
- portrait_highkey: human or animal portraits with very bright, mostly white background and high-key lighting
- macro_nature: extreme close-ups of flowers, insects, leaves, feathers, fur, etc.
- macro_food: close-ups of food, dishes, drinks, ingredients where food is clearly the main subject
- street_documentary: candid street scenes, people in public spaces, documentary-style urban life
- sports_action: sports or fast action (players, running, jumping, flying, strong motion)
- architecture_realestate: buildings, interiors, rooms, city structures as the main subject
- studio_photography: controlled studio shots of products, objects, or people with deliberate lighting and backgrounds (often plain, seamless, or styled backdrops)
- night_natural_light: night scenes lit mainly by moonlight or natural low light (e.g. stars, moonlit landscapes)
- night_artificial_light: night scenes dominated by artificial lights (neon, city lights, concerts, street lamps)
- fineart_creative: abstract, surreal, heavily stylized or concept-driven images that don’t clearly fit the other categories

Answer with JUST the category name, exactly as written above:

STRICT RULES:

1. If the main subject is a human OR non-human animal and fills most of the frame
   AND is framed like a portrait (head/torso or full body with context), choose a portrait_* category.
2. If the main subject is a non-human animal but framed more as part of a larger scene
   (e.g. more environmental, not portrait-style), choose wildlife_animal.
3. If the subject is a tiny part of a larger outdoor scene → landscape.
4. Macro categories are ONLY for extreme close-ups.
5. If uncertain, choose the closest general category (NOT wildlife or macro).

FORBIDDEN OUTPUT:

- No explanations
- No reasoning
- No greetings
- No multi-sentence answers
- No markdown
- No justification
- No extra words

Return ONLY valid JSON, exactly like:
{"category": "<one category from the list>"}

Examples:
{"category": "fineart_creative"}
or
{"category": "night_natural_light"}
"""


class ClassificationResult:
    """Result of image context classification with confidence tracking."""
    __slots__ = ('context', 'confidence', 'method', 'raw_response', 'retries')
    
    def __init__(self, context: str, confidence: str = 'low', method: str = 'fallback',
                 raw_response: str = '', retries: int = 0):
        self.context = context
        self.confidence = confidence  # 'high', 'medium', 'low'
        self.method = method  # 'exact', 'number', 'category', 'weighted', 'partial', 'fallback'
        self.raw_response = raw_response
        self.retries = retries


def classify_image_context(image_path: str, ollama_host_url: str, model: str,   
                           max_retries: int = MAX_RETRIES) -> str:
    """Quickly classify image context using vision model with retry logic.
    
    Returns the context string. For detailed results, use classify_image_context_detailed().
    """
    result = classify_image_context_detailed(image_path, ollama_host_url, model, max_retries)
    return result.context


def classify_image_context_detailed(image_path: str, ollama_host_url: str, model: str,
                                     max_retries: int = MAX_RETRIES) -> ClassificationResult:
    """Classify image context with retry logic and confidence tracking.
    
    Retries on:
    - Network errors (timeout, connection)
    - Unrecognized responses
    """
    last_result = None
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = _classify_image_context_once(image_path, ollama_host_url, model)
            result.retries = attempt
            
            # If we got a valid classification (not a fallback), return it
            if result.method != 'fallback' and result.confidence != 'low':
                return result
            
            # Got a fallback/unrecognized response - retry if we have attempts left
            last_result = result
            if attempt < max_retries - 1:
                logger.warning(f"Classification returned unrecognized response for {image_path} "
                              f"(attempt {attempt + 1}/{max_retries}): '{result.raw_response[:100]}...' - retrying")
                time.sleep(RETRY_DELAY_BASE * (attempt + 1))
            else:
                # Last attempt - return whatever we got
                logger.warning(f"Classification exhausted retries for {image_path}, using fallback")
                return result
                
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(f"Classification timeout for {image_path} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY_BASE * (attempt + 1))
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f"Classification connection error for {image_path} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY_BASE * (attempt + 1))
        except Exception as e:
            # Don't retry for other exceptions
            logger.warning(f"Classification failed for {image_path}: {e}")
            return ClassificationResult('unknown', 'low', 'error', str(e), attempt)
    
    # If we have a last result from fallback attempts, return it
    if last_result:
        return last_result
    
    logger.warning(f"Classification exhausted retries for {image_path}: {last_error}")
    return ClassificationResult('unknown', 'low', 'retry_exhausted', str(last_error), max_retries)


def _classify_image_context_once(image_path: str, ollama_host_url: str, model: str) -> ClassificationResult:
    """Single attempt at image context classification."""
    # Encode image appropriately (handles RAW files by creating JPEG preview)
    logger.debug(f"Encoding image for classification: {image_path}")
    encoded_image = encode_image_for_classification(image_path)
    image_size_kb = len(encoded_image) * 3 / 4 / 1024  # Approximate decoded size in KB
    logger.debug(f"Encoded image size: {image_size_kb:.1f}KB")
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "images": [encoded_image],
        "prompt": IMAGE_CONTEXT_CLASSIFIER_PROMPT,
        "options": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "num_predict": 16,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0
        }
    }
    
    logger.debug(f"Sending classification request for {image_path}")
    response = requests.post(ollama_host_url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    result = response.json()
    
    # Log full result for debugging
    logger.debug(f"Classification API response keys: {result.keys()}")
    
    raw_response = result.get('response', '')
    context_label = raw_response.strip().lower()
    # 1) Direct "category=<label>" extraction
    candidate = None  # Initialize before conditional assignment
    m = re.search(r'category\s*=\s*([a-z_]+)', context_label)
    if m:
        candidate = m.group(1)

    # Known labels from your prompt (canonical names)
    allowed_labels = {
        "landscape",
        "wildlife_animal",
        "portrait_neutral",
        "portrait_highkey",
        "macro_nature",
        "macro_food",
        "street_documentary",
        "sports_action",
        "architecture_realestate",
        "studio_photography",
        "night_natural_light",
        "night_artificial_light",
        "fineart_creative",
    }

    # Handle truncated/internal variants like "wildlife_an"
    # Find exact or unique-prefix match
    matched = None
    if candidate:
        if candidate in allowed_labels:
            matched = candidate
        else:
            matches = [lbl for lbl in allowed_labels if lbl.startswith(candidate)]
            if len(matches) == 1:
                matched = matches[0]

    if matched:
        logger.info(f"Context classification: {matched} (category=) for {image_path}")
        return ClassificationResult(matched, 'high', 'category_eq', raw_response, 0)
    # Log raw response for debugging
    logger.debug(f"Context classification raw response: '{raw_response}' for {image_path}")
    
    # Check for empty response - this indicates the model doesn't support vision or has an issue
    if not raw_response or not context_label:
        # try alternate fields
        fallback_response = (
            result.get('text') or
            result.get('message') or
            (result.get('choices', [{}])[0].get('text') if isinstance(result.get('choices'), list) and result.get('choices') else '') or
            result.get('thinking')
        )
        if fallback_response:
            raw_response = fallback_response
            context_label = raw_response.strip().lower()
            logger.debug(f"Context classification fallback response: '{raw_response}' for {image_path}")
        if not raw_response or not context_label:
            logger.warning(f"Context classification returned empty response for {image_path}. Model payload: {result}")
            return ClassificationResult('unknown', 'low', 'empty_response', '', 0)
    
    # Try parse_context_response first (handles truncation and category= format)
    parsed_label, parse_source = parse_context_response(raw_response)
    if parsed_label != "unknown":
        confidence = 'high' if parse_source in ('exact', 'alias') else 'medium'
        logger.info(f"Context classification: {parsed_label} ({parse_source}) for {image_path}")
        return ClassificationResult(parsed_label, confidence, parse_source, raw_response, 0)
    
    # Handle numbered responses (e.g., "1", "1.", "5. landscape")
    number_to_context = {
        '1': 'landscape',
        '2': 'portrait_neutral',
        '3': 'portrait_highkey',
        '4': 'macro_food',
        '5': 'macro_nature',
        '6': 'street_documentary',
        '7': 'sports_action',
        '8': 'wildlife_animal',
        '9': 'night_artificial_light',
        '10': 'night_natural_light',
        '11': 'architecture_realestate',
        '12': 'studio_photography',
        '13': 'fineart_creative'
    }
    
    # Check if response starts with a number
    first_word = context_label.split()[0] if context_label.split() else ''
    clean_number = first_word.rstrip('.').strip()
    if clean_number in number_to_context:
        matched_context = number_to_context[clean_number]
        logger.info(f"Context classification: {matched_context} (from number '{clean_number}') for {image_path}")
        return ClassificationResult(matched_context, 'high', 'number', raw_response, 0)
    
    # Explicit "category X" reference takes priority
    category_matches = re.findall(r'category\s*(\d+)', context_label)
    if category_matches:
        last_match = category_matches[-1]
        if last_match in number_to_context:
            matched_context = number_to_context[last_match]
            logger.info(f"Context classification: {matched_context} (from 'category {last_match}') for {image_path}")
            return ClassificationResult(matched_context, 'high', 'category', raw_response, 0)

    # Weighted sentence analysis to avoid false positives
    sentences = re.split(r'[.!?\n]+', context_label)
    positive_markers = ["include", "includes", "fits", "fit", "belongs to", "classified as", "this is", "is a", "matches", "best fits", "category is", "would be", "should be", "falls under"]
    negative_markers = ["don't fit", "doesn't fit", "does not fit", "not ", "no ", "without", "isn't", "aren't"]
    candidate_scores = {}
    
    # First pass: check for explicit category mentions with context
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        sentiment = 0
        lowered = s
        if any(marker in lowered for marker in negative_markers):
            sentiment = -1
        elif any(marker in lowered for marker in positive_markers):
            sentiment = 1
        for known_context in PROFILE_CONFIG.keys():
            # Check for the context name with word boundaries
            pattern = r'\b' + re.escape(known_context) + r'\b'
            if re.search(pattern, lowered):
                score = 2 if sentiment == 1 else (-2 if sentiment == -1 else 1)
                candidate_scores[known_context] = max(score, candidate_scores.get(known_context, -10))
    
    # Second pass: look for category name anywhere in response (common in verbose responses)
    if not candidate_scores:
        for known_context in PROFILE_CONFIG.keys():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(known_context) + r'\b'
            if re.search(pattern, context_label):
                candidate_scores[known_context] = 1
    
    if candidate_scores:
        best_context, best_score = max(candidate_scores.items(), key=lambda kv: kv[1])
        if best_score > 0:
            confidence = 'high' if best_score >= 2 else 'medium'
            logger.info(f"Context classification: {best_context} (sentence-weighted, score={best_score}) for {image_path}")
            return ClassificationResult(best_context, confidence, 'weighted', raw_response, 0)

    # Validate against known contexts (exact match)
    if context_label in PROFILE_CONFIG:
        logger.info(f"Context classification: {context_label} (exact match) for {image_path}")
        return ClassificationResult(context_label, 'high', 'exact', raw_response, 0)
    
    # Try to extract valid context from response (partial match)
    for known_context in PROFILE_CONFIG.keys():
        if known_context in context_label:
            logger.info(f"Context classification: {known_context} (extracted from '{context_label}') for {image_path}")
            return ClassificationResult(known_context, 'medium', 'partial', raw_response, 0)
    
    # Log the failure with the actual response
    logger.warning(f"Context classification failed: unknown response '{raw_response}' (normalized: '{context_label}') for {image_path}. "
                  f"Defaulting to 'unknown'. Consider manual context override.")
    return ClassificationResult('unknown', 'low', 'fallback', raw_response, 0)


def analyze_image_technical(image_path: str, iso_value: Optional[int] = None, context: str = 'studio_photography') -> Dict:
    """Analyze image for technical quality metrics with camera/ISO-agnostic noise estimation."""
    metrics = {
        'sharpness': 0.0,
        'brightness': 0.0,
        'contrast': 0.0,
        'histogram_clipping_highlights': 0.0,
        'histogram_clipping_shadows': 0.0,
        'color_cast': 'neutral',
        'context': context,
        'noise_sigma': 0.0,   # high-frequency noise in flat areas, normalized [0,1]
        'noise_score': 0.0,   # 0–100 severity
        'noise': 0.0,         # alias of noise_score for backward compatibility
        'status': 'success',  # 'success' or 'error' to indicate analysis status
    }
    
    # Get context profile
    profile = get_profile(context)
    metrics['context_profile'] = profile['name']
    
    try:
        # Read image with PIL (or rawpy) for stats
        with open_image_for_analysis(image_path) as img:
            img_rgb = img if img.mode == 'RGB' else img.convert('RGB')
            width, height = img_rgb.size
            long_edge = max(width, height)
            if long_edge < MIN_LONG_EDGE:
                raise ImageResolutionTooSmallError(
                    f"Skipping {image_path}: long edge {long_edge}px is below the minimum {MIN_LONG_EDGE}px requirement"
                )
            metrics['dimensions'] = f"{width}x{height}"
            metrics['megapixels'] = float((width * height) / 1_000_000)
            metrics['aspect_ratio'] = float(width / max(height, 1))
            dpi_info = img.info.get('dpi', (72, 72))
            if isinstance(dpi_info, tuple):
                metrics['dpi_x'] = float(dpi_info[0])
                metrics['dpi_y'] = float(dpi_info[1] if len(dpi_info) > 1 else dpi_info[0])
            elif isinstance(dpi_info, (int, float)):
                metrics['dpi_x'] = metrics['dpi_y'] = float(dpi_info)
            
            # --- Brightness / contrast ---
            stat = ImageStat.Stat(img_rgb)
            metrics['brightness'] = float(sum(stat.mean) / len(stat.mean))
            metrics['contrast'] = float(sum(stat.stddev) / len(stat.stddev))

            # --- Histogram clipping (highlights/shadows) ---
            # Use max per channel because a single clipped channel is problematic
            histogram = img_rgb.histogram()
            total_pixels = img_rgb.size[0] * img_rgb.size[1]

            highlights_per_channel = []
            shadows_per_channel = []

            for channel_idx in range(3):  # R, G, B
                channel_hist = histogram[channel_idx * 256:(channel_idx + 1) * 256]
                highlight_pixels = sum(channel_hist[HISTOGRAM_HIGHLIGHT_RANGE[0]:HISTOGRAM_HIGHLIGHT_RANGE[1]])
                shadow_pixels = sum(channel_hist[HISTOGRAM_SHADOW_RANGE[0]:HISTOGRAM_SHADOW_RANGE[1]])

                highlights_per_channel.append((highlight_pixels / total_pixels) * 100.0)
                shadows_per_channel.append((shadow_pixels / total_pixels) * 100.0)

            # Use max to catch worst-case single-channel clipping
            metrics['histogram_clipping_highlights'] = max(highlights_per_channel)
            metrics['histogram_clipping_shadows'] = max(shadows_per_channel)

            # --- Color cast detection (global) ---
            r_mean, g_mean, b_mean = stat.mean[:3]
            max_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
            metrics['color_cast_delta'] = float(max_diff)
            # We only set the label here; the profile decides how/if to penalize.
            if max_diff > COLOR_CAST_THRESHOLD:
                # Identify dominant channel with sufficient margin (>5 units)
                if r_mean > g_mean + 5 and r_mean > b_mean + 5:
                    metrics['color_cast'] = 'warm/red'
                elif b_mean > r_mean + 5 and b_mean > g_mean + 5:
                    metrics['color_cast'] = 'cool/blue'
                elif g_mean > r_mean + 5 and g_mean > b_mean + 5:
                    metrics['color_cast'] = 'green'
                else:
                    # max_diff > threshold but no clear dominant channel
                    metrics['color_cast'] = 'mixed'

            # --- Sharpness via Laplacian variance (scale-normalized) ---
            img_gray = img_rgb.convert('L')
            gray_array = np.array(img_gray)

            sharpness: Optional[float] = None
            sigma_noise: Optional[float] = None
            noise_score: Optional[float] = None

            if CV2_AVAILABLE and cv2 is not None:
                try:
                    laplacian_map = cv2.Laplacian(gray_array, cv2.CV_32F)
                    sharpness = _compute_perceptual_sharpness(laplacian_map)

                    h, w = gray_array.shape
                    target_long = 2048
                    scale = target_long / float(max(h, w))
                    if scale < 1.0:
                        gray_small = cv2.resize(
                            gray_array,
                            dsize=None,
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        gray_small = gray_array

                    gray_f = gray_small.astype(np.float32)
                    max_val = gray_f.max()
                    if max_val > 1.5:
                        gray_f /= 255.0
                    elif max_val > 0:
                        gray_f /= max_val

                    blurred = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.0, sigmaY=1.0)
                    residual = gray_f - blurred

                    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
                    grad_mag = np.sqrt(gx**2 + gy**2)

                    edge_thresh = np.percentile(grad_mag, 75.0)
                    flat_mask = grad_mag < edge_thresh

                    p_low, p_high = np.percentile(gray_f, [5.0, 95.0])
                    luminance_mask = (gray_f > p_low) & (gray_f < p_high)

                    final_mask = flat_mask & luminance_mask
                    flat_residuals = residual[final_mask]

                    min_pixels = int(0.01 * residual.size)
                    if flat_residuals.size < min_pixels:
                        flat_residuals = residual.flatten()
                        logger.debug(
                            f"Noise estimation using full image for {image_path} (insufficient flat regions)"
                        )

                    med = float(np.median(flat_residuals))
                    mad = float(np.median(np.abs(flat_residuals - med)))
                    if mad < 1e-6:
                        sigma_noise = 0.0
                    else:
                        sigma_noise = 1.4826 * mad

                    p1, p99 = np.percentile(gray_f, [1.0, 99.0])
                    dynamic_range = max(float(p99 - p1), 1e-6)
                    relative_noise = sigma_noise / dynamic_range

                    REL_NOISE_MIN = 0.002
                    REL_NOISE_MAX = 0.04
                    rn_clipped = min(max(relative_noise, REL_NOISE_MIN), REL_NOISE_MAX)
                    noise_score = (rn_clipped - REL_NOISE_MIN) / (REL_NOISE_MAX - REL_NOISE_MIN) * 100.0
                except Exception as err:
                    logger.debug("OpenCV-based sharpness/noise failed for %s: %s", image_path, err)

            if (
                sharpness is None
                or sigma_noise is None
                or noise_score is None
                or not math.isfinite(sharpness)
                or not math.isfinite(noise_score)
                or sharpness <= 0.0
                or noise_score <= 0.0
            ):
                logger.debug(
                    "Falling back to PIL/numpy sharpness/noise estimation for %s due to degenerate CV2 result",
                    image_path,
                )
                sharpness, sigma_noise, noise_score = _compute_sharpness_noise_fallback(gray_array)

            metrics['sharpness'] = float(sharpness)
            metrics['noise_sigma'] = float(sigma_noise) if sigma_noise is not None else 0.0
            metrics['noise_score'] = float(noise_score)
            metrics['noise'] = float(noise_score)
    
    except ImageResolutionTooSmallError:
        raise
    except Exception as e:
        logger.debug(f"Could not analyze technical metrics for {image_path}: {e}")
        metrics['status'] = 'error'
    
    return metrics


def count_technical_flags(technical_metrics: Dict) -> Tuple[int, int]:
    """Count critical and warn-level technical flags for an image.
    
    Returns:
        Tuple of (critical_count, warn_count) where:
        - critical_count: number of metrics exceeding critical threshold
        - warn_count: number of metrics exceeding warn but not critical threshold
    """
    critical_count = 0
    warn_count = 0
    
    # Sharpness: lower is worse
    sharpness = technical_metrics.get('sharpness')
    if isinstance(sharpness, (int, float)):
        if sharpness < STOCK_SHARPNESS_CRITICAL:
            critical_count += 1
        elif sharpness < STOCK_SHARPNESS_WARN:
            warn_count += 1
    
    # Noise: higher is worse
    noise_score = technical_metrics.get('noise_score')
    if isinstance(noise_score, (int, float)):
        if noise_score > STOCK_NOISE_CRITICAL:
            critical_count += 1
        elif noise_score > STOCK_NOISE_WARN:
            warn_count += 1
    
    # Highlight clipping: higher is worse
    highlights = technical_metrics.get('histogram_clipping_highlights')
    if isinstance(highlights, (int, float)):
        if highlights > STOCK_CLIPPING_HIGHLIGHTS_CRITICAL:
            critical_count += 1
        elif highlights > STOCK_CLIPPING_HIGHLIGHTS_WARN:
            warn_count += 1
    
    # Shadow clipping: higher is worse
    shadows = technical_metrics.get('histogram_clipping_shadows')
    if isinstance(shadows, (int, float)):
        if shadows > STOCK_CLIPPING_SHADOWS_CRITICAL:
            critical_count += 1
        elif shadows > STOCK_CLIPPING_SHADOWS_WARN:
            warn_count += 1
    
    # Color cast: higher delta is worse
    color_cast_delta = technical_metrics.get('color_cast_delta')
    if isinstance(color_cast_delta, (int, float)):
        if color_cast_delta > COLOR_CAST_CRITICAL:
            critical_count += 1
        elif color_cast_delta > COLOR_CAST_WARN:
            warn_count += 1
    
    return critical_count, warn_count


def is_technically_warned(technical_metrics: Dict) -> bool:
    """Determine if an image should be counted as 'technically warned'.
    
    An image is technically warned if it has:
    - At least one critical flag, OR
    - At least two different metrics with warn flags
    """
    critical_count, warn_count = count_technical_flags(technical_metrics)
    return critical_count >= 1 or warn_count >= 2


def assess_technical_metrics(technical_metrics: Dict, context: str = "studio_photography") -> List[str]:
    """Generate human-readable warnings based on measured metrics and context."""
    profile = get_profile(context)
    rules = profile.get("rules", {})
    warnings: List[str] = []

    # Sharpness
    sharpness = technical_metrics.get('sharpness')
    sharpness_rules = rules.get("sharpness", {})
    if sharpness is not None:
        if sharpness < sharpness_rules.get("critical_threshold", 30.0):
            warnings.append(f"Sharpness critically low ({sharpness:.1f})")
        elif sharpness < sharpness_rules.get("soft_threshold", 60.0):
            warnings.append(f"Lower sharpness ({sharpness:.1f}) may impact detail")

    # Highlight and shadow clipping
    highlights = float(technical_metrics.get('histogram_clipping_highlights', 0.0))
    shadows = float(technical_metrics.get('histogram_clipping_shadows', 0.0))
    clipping_rules = rules.get("clipping", {})
    clip_warn = clipping_rules.get("warn_pct", 5.0)

    if highlights > clip_warn:
        warnings.append(f"Highlight clipping {highlights:.1f}% reduces tonal range")

    if shadows > clip_warn:
        warnings.append(f"Shadow clipping {shadows:.1f}% removes shadow detail")

    # Color cast (respect profile penalties)
    color_cast = technical_metrics.get('color_cast', 'neutral')
    color_rules = rules.get("color_cast", {})
    if color_cast != 'neutral' and color_rules.get("penalty", 0) > 0:
        warnings.append(f"Color cast detected: {color_cast}")

    # Noise (0–100 severity)
    noise_score = float(technical_metrics.get('noise_score', 0.0))
    noise_rules = rules.get("noise", {})
    if noise_score > noise_rules.get("high", 60.0):
        warnings.append(f"High noise level (score {noise_score:.1f}/100)")
    elif noise_score > noise_rules.get("warn", 30.0):
        warnings.append(f"Elevated noise (score {noise_score:.1f}/100)")

    return warnings


def compute_post_process_potential(technical_metrics: Dict, context: str = "studio_photography") -> int:
    """Estimate how much post-processing can improve this image (0–100)."""
    profile = get_profile(context)
    post_process = profile.get("post_process", {})
    rules = profile.get("rules", {})
    
    base_score = float(post_process.get("base", 70))

    # Sharpness contribution
    sharpness = technical_metrics.get('sharpness')
    if sharpness is not None:
        sharpness_heavy = post_process.get("sharpness_heavy", 35.0)
        sharpness_soft = post_process.get("sharpness_soft", 55.0)
        if sharpness < sharpness_heavy:
            base_score -= 25
        elif sharpness < sharpness_soft:
            base_score -= 10
        else:
            base_score += 5

    # Clipping contribution
    highlights = float(technical_metrics.get('histogram_clipping_highlights', 0.0))
    shadows = float(technical_metrics.get('histogram_clipping_shadows', 0.0))
    clipping = max(highlights, shadows)
    
    clip_high = post_process.get("clip_high", 15.0)
    clip_mid = post_process.get("clip_mid", 5.0)
    clip_bonus = post_process.get("clip_bonus", 2.0)

    if clipping > clip_high:
        base_score -= 20
    elif clipping > clip_mid:
        base_score -= 10
    elif clipping < clip_bonus:
        base_score += 5
    # else: neutral zone (between clip_bonus and clip_mid) - no adjustment

    # Noise contribution
    noise_score = float(technical_metrics.get('noise_score', 0.0))
    noise_rules = rules.get("noise", {})
    noise_penalty_high = post_process.get("noise_penalty_high", 15)
    noise_penalty_mid = post_process.get("noise_penalty_mid", 5)
    
    if noise_score > noise_rules.get("high", 60.0):
        base_score -= noise_penalty_high
    elif noise_score > noise_rules.get("warn", 30.0):
        base_score -= noise_penalty_mid

    # Color cast contribution
    color_cast = technical_metrics.get('color_cast', 'neutral')
    if color_cast != 'neutral':
        color_penalty = rules.get("color_cast", {}).get("penalty", 5)
        base_score -= color_penalty

    post_score = max(0, min(100, int(round(base_score))))
    return post_score


def analyze_image_with_context(image_path: str, ollama_host_url: str, model: str,
                               context_override: Optional[str], skip_context_classification: bool,
                               stock_eval: bool = False
                               ) -> Tuple[str, Dict, Dict, List[str]]:
    """Determine context, extract EXIF, and compute technical metrics for an image."""
    if context_override:
        image_context = context_override if context_override in PROFILE_CONFIG else 'studio_photography'
        logger.info(f"Using manual context override: {image_context}")
    else:
        cached_context = read_cached_context(image_path)
        if cached_context:
            image_context = cached_context
            logger.info(f"Using cached context from EXIF: {image_context}")
        elif skip_context_classification:
            image_context = 'studio_photography'
            logger.debug(f"Context classification disabled, using default: {image_context}")
        else:
            try:
                image_context = classify_image_context(image_path, ollama_host_url, model)
            except Exception as e:
                logger.warning(f"Context classification failed for {image_path}: {e}, using default")
                image_context = 'studio_photography'

    logger.info(f"Image context for {image_path}: {image_context} ({PROFILE_CONFIG[image_context]['name']})")

    exif_data = extract_exif_metadata(image_path)

    iso_value = exif_data.get('iso')
    if isinstance(iso_value, str):
        try:
            iso_str = iso_value.replace('ISO', '').replace(',', '').strip()
            if '/' in iso_str:
                iso_str = iso_str.split('/')[0]
            iso_value = int(float(iso_str))
        except (ValueError, AttributeError):
            logger.debug(f"Could not parse ISO value: {iso_value}")
            iso_value = None
    elif isinstance(iso_value, float):
        iso_value = int(iso_value)

    technical_metrics = analyze_image_technical(image_path, iso_value, context=image_context)
    if technical_metrics.get('status') == 'error':
        logger.warning(f"Technical analysis failed for {image_path}, using default metrics")

    technical_warnings = assess_technical_metrics(technical_metrics, context=image_context)
    technical_metrics['warnings'] = technical_warnings
    technical_metrics['post_process_potential'] = compute_post_process_potential(technical_metrics, context=image_context)
    stock_notes, stock_fixable = compute_stock_notes(technical_metrics)
    technical_metrics['stock_notes'] = stock_notes
    technical_metrics['stock_fixable'] = stock_fixable
    if stock_eval and stock_notes:
        technical_warnings.extend(stock_notes)

    return image_context, exif_data, technical_metrics, technical_warnings


def configure_pyiqa_logging(verbose: bool = False, debug: bool = False):
    """Control PyIQA log verbosity based on CLI flags."""
    pyiqa_logger = logging.getLogger('pyiqa')
    if not any(isinstance(handler, logging.NullHandler) for handler in pyiqa_logger.handlers):
        pyiqa_logger.addHandler(logging.NullHandler())

    if debug:
        pyiqa_level = logging.DEBUG
        pyiqa_logger.propagate = True
    elif verbose:
        pyiqa_level = logging.INFO
        pyiqa_logger.propagate = True
    else:
        pyiqa_level = logging.WARNING
        pyiqa_logger.propagate = False

    pyiqa_logger.setLevel(pyiqa_level)


def setup_logging(log_file: Optional[str] = None, verbose: bool = False, debug: bool = False):
    """Configure logging with file handler and appropriate level."""
    # Set level based on verbose/debug flags
    # Default to WARNING for quiet operation; --verbose enables INFO, --debug enables DEBUG
    if debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    
    # Clear root logger handlers (from basicConfig at module load)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()
    logger.setLevel(log_level)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if log file specified
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB per file, 5 backups
        )
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    configure_pyiqa_logging(verbose=verbose, debug=debug)


def get_image_hash(image_path: str) -> str:
    """Calculate hash of image file for caching."""
    hash_md5 = hashlib.md5()
    with open(image_path, 'rb') as f:
        # Read first 1MB and last 1KB for quick hash
        chunk = f.read(1024*1024)
        hash_md5.update(chunk)
        try:
            f.seek(-1024, 2)  # Seek to 1KB from end
            hash_md5.update(f.read())
        except:
            pass
    return hash_md5.hexdigest()


def get_cache_path(image_path: str, model: str, cache_dir: str) -> str:
    """Get cache file path for image and model combination."""
    image_hash = get_image_hash(image_path)
    cache_key = f"{image_hash}_{model}_{CACHE_VERSION}"
    return os.path.join(cache_dir, f"{cache_key}.cache")


def load_from_cache(image_path: str, model: str, cache_dir: str) -> Optional[Dict]:
    """Load cached API response if available."""
    if not cache_dir:
        return None
    
    try:
        cache_path = get_cache_path(image_path, model, cache_dir)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                logger.debug(f"Cache hit for {image_path}")
                return cached_data
    except Exception as e:
        logger.debug(f"Cache read failed for {image_path}: {e}")
    
    return None


def save_to_cache(image_path: str, model: str, metadata: Dict, cache_dir: str):
    """Save evaluation metadata to disk cache.
    
    Creates cache directories as needed. Cache key is derived from
    image path, model name, and file modification time.
    """
    """Save API response to cache."""
    if not cache_dir:
        return
    
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = get_cache_path(image_path, model, cache_dir)
        with open(cache_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.debug(f"Cached response for {image_path}")
    except Exception as e:
        logger.warning(f"Cache write failed for {image_path}: {e}")


def get_backup_path(image_path: str, backup_dir: Optional[str] = None) -> str:
    """Get backup path for an image file."""
    if backup_dir:
        # Create parallel directory structure in backup dir
        rel_path = os.path.relpath(image_path)
        backup_path = os.path.join(backup_dir, rel_path + '.original')
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        return backup_path
    else:
        # Original behavior - same directory
        return f"{os.path.splitext(image_path)[0]}.original{os.path.splitext(image_path)[1]}"


def sanitize_string(s: str) -> str:
    """Sanitize string for Exif compatibility."""
    return s.replace('\x00', '').replace('\n', ' ').replace('\r', ' ')  # Remove null bytes and newlines


def _decode_exif_string(value: Union[bytes, str, int, None]) -> str:
    """Decode EXIF string values safely to plain text."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore').strip('\x00').strip()
    if value is not None:
        return str(value).strip('\x00').strip()
    return ''


def read_cached_context(image_path: str) -> Optional[str]:
    """Read cached context from EXIF ImageDescription if it maps to a known profile."""
    try:
        file_ext = Path(image_path).suffix.lower()

        context_value: Optional[str] = None
        if file_ext in RAW_EXTENSIONS:
            result = subprocess.run(
                ['exiftool', '-s3', '-ImageDescription', image_path],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                context_value = result.stdout.strip()
        else:
            if not PIEXIF_AVAILABLE or piexif is None:
                logger.debug("piexif not available; cannot inspect ImageDescription for %s", image_path)
                return None

            with Image.open(image_path) as img:
                exif_data = img.info.get("exif", b"")
                if not exif_data:
                    return None

                exif_dict = piexif.load(exif_data)
                raw_value = exif_dict["0th"].get(piexif.ImageIFD.ImageDescription)
                context_value = _decode_exif_string(raw_value)

        if not context_value:
            return None

        normalized = context_value.strip().lower()
        if normalized in PROFILE_CONFIG:
            return normalized

        mapped = CONTEXT_PROFILE_MAP.get(normalized)
        if mapped and mapped in PROFILE_CONFIG:
            return mapped

        logger.debug(
            "Cached context '%s' in %s did not map to a known profile; ignoring",
            context_value,
            image_path,
        )
        return None
    except Exception as e:
        logger.debug(f"Failed to read cached context from EXIF for {image_path}: {e}")
        return None


class Metadata(BaseModel):
    model_config = ConfigDict(extra='allow')  # Allow extra fields without raising an error
    
    score: int  # Changed from str to int
    title: str
    description: str
    keywords: str

    @field_validator('keywords')
    def validate_keywords(cls, v):
        keyword_list = v.split(',')
        # Allow up to and including 12 keywords
        if len(keyword_list) > 12:
            keyword_list = keyword_list[:12]
        return ','.join(keyword_list).strip()  # Join back to a string and strip whitespace


def has_user_comment(image_path: str) -> bool:
    """Check if the UserComment metadata exists and is not empty."""
    try:
        file_ext = Path(image_path).suffix.lower()
        if file_ext in RAW_EXTENSIONS:
            result = subprocess.run(
                ['exiftool', '-UserComment', '-s3', image_path],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
            return False

        if not PIEXIF_AVAILABLE or piexif is None:
            logger.debug("piexif not available; cannot inspect UserComment for %s", image_path)
            return False

        with Image.open(image_path) as img:
            exif_data = img.info.get("exif", b"")
            if not exif_data:  # Check if EXIF data is empty
                return False

            exif_dict = piexif.load(exif_data)
            user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
            return user_comment is not None and user_comment != b''
    except Exception as e:
        logger.error(f"Error reading metadata for {image_path}: {e}")
        return False


def verify_metadata(image_path: str, expected_metadata: Dict) -> bool:
    """Verify that metadata was correctly embedded in the image."""
    try:
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # For RAW/TIFF files, use exiftool to verify
        if file_ext in ['.dng', '.nef', '.tif', '.tiff']:
            result = subprocess.run(
                ['exiftool', '-UserComment', '-XPTitle', '-XPComment', '-XPKeywords', '-s', '-s', '-s', image_path],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                output = result.stdout
                # Basic check - see if score appears in output
                return str(expected_metadata.get('score', '')) in output
            return False
        
        # For JPEG/PNG, use PIL
        if not PIEXIF_AVAILABLE or piexif is None:
            logger.debug("piexif not available; cannot verify metadata in %s", image_path)
            return False

        with Image.open(image_path) as img:
            exif_data = img.info.get("exif", b"")
            if not exif_data:
                return False
            
            exif_dict = piexif.load(exif_data)
            user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
            
            if user_comment:
                # Check if score is in UserComment
                comment_str = str(user_comment)
                return str(expected_metadata.get('score', '')) in comment_str
            
        return False
    except Exception as e:
        logger.warning(f"Could not verify metadata for {image_path}: {e}")
        return False


def embed_metadata_exiftool(image_path: str, metadata: Dict, backup_dir: Optional[str] = None, verify: bool = False) -> bool:
    """Embed metadata into RAW/TIFF files using exiftool.
    Returns True if successful, False otherwise.
    """
    try:
        # Create backup first
        backup_image_path = get_backup_path(image_path, backup_dir)
        
        # Build exiftool command
        # UserComment for score, XPTitle, XPComment (description), XPKeywords
        technical_str = metadata.get("technical_score", "")

        cmd = [
            'exiftool',
            '-overwrite_original',  # Don't create _original files
            f'-UserComment={metadata.get("score", "")}',
            f'-XPTitle={metadata.get("title", "")}',
            f'-XPComment={metadata.get("description", "")}',
            f'-XPKeywords={metadata.get("keywords", "")}',
            f'-XPSubject={technical_str}',
        ]
        context_label = sanitize_string(metadata.get("context_label", ""))
        if context_label:
            cmd.append(f'-ImageDescription={context_label}')
        cmd.append(image_path)
        
        # Create manual backup before modifying
        if not os.path.exists(backup_image_path):
            import shutil
            shutil.copy2(image_path, backup_image_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{Fore.BLUE}Embedding score:{Fore.RESET} {Fore.GREEN}{metadata.get('score', '')}{Fore.RESET}")
            if technical_str:
                print(f"{Fore.BLUE}Embedding technical score:{Fore.RESET} {Fore.GREEN}{technical_str}{Fore.RESET}")
            print(f"{Fore.BLUE}Embedding Title:{Fore.RESET} {Fore.GREEN}{metadata.get('title', '')}{Fore.RESET}")
            print(f"{Fore.BLUE}Embedding Description:{Fore.RESET} {Fore.GREEN}{metadata.get('description', '')}{Fore.RESET}")
            print(f"{Fore.BLUE}Embedding Keywords:{Fore.RESET} {Fore.GREEN}{metadata.get('keywords', '')}{Fore.RESET}")
            
            # Verify if requested
            if verify:
                if verify_metadata(image_path, metadata):
                    print(Fore.GREEN + f"✓ Metadata verified in {Style.BRIGHT}{image_path}{Style.RESET_ALL}" + Fore.RESET)
                else:
                    logger.warning(f"⚠ Metadata verification failed for {image_path}")
            else:
                print(Fore.GREEN + f"Metadata successfully embedded in {Style.BRIGHT}{image_path}{Style.RESET_ALL} using exiftool" + Fore.RESET)
            return True
        else:
            logger.error(f"exiftool failed for {image_path}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error embedding metadata with exiftool in {image_path}: {e}")
        return False


def embed_metadata(image_path: str, metadata: Dict, backup_dir: Optional[str] = None, verify: bool = False) -> bool:
    """
    Embed metadata into image file. Returns True if successful, False if skipped.
    Uses exiftool for RAW/TIFF files, PIL for JPEG/PNG.
    """
    try:
        # Check if file format is RAW/TIFF
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.dng', '.nef', '.tif', '.tiff']:
            return embed_metadata_exiftool(image_path, metadata, backup_dir, verify)
        
        if not PIEXIF_AVAILABLE or piexif is None:
            logger.error("piexif is required to embed metadata into JPEG/PNG files. Please install piexif.")
            return False

        # Open the image to access its EXIF data
        with Image.open(image_path) as img:
            exif_data = img.info.get("exif", b"")
            if exif_data:
                exif_dict = piexif.load(exif_data)
            else:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}  # Create a new EXIF structure

            user_comment = piexif.helper.UserComment.dump(str(metadata.get('score', '')))
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
            print(f"{Fore.BLUE}Embedding score:{Fore.RESET} {Fore.GREEN}{metadata.get('score', '')}{Fore.RESET}")

            # Embedding Title (UTF-16LE with BOM for Windows compatibility)
            title = metadata.get('title', '').encode('utf-16le') + b'\x00\x00'
            exif_dict["0th"][piexif.ImageIFD.XPTitle] = title
            print(f"{Fore.BLUE}Embedding Title:{Fore.RESET} {Fore.GREEN}{metadata.get('title', '')}{Fore.RESET}")

            # Embedding Description
            description = metadata.get('description', '').encode('utf-16le') + b'\x00\x00'
            exif_dict["0th"][piexif.ImageIFD.XPComment] = description
            print(f"{Fore.BLUE}Embedding Description:{Fore.RESET} {Fore.GREEN}{metadata.get('description', '')}{Fore.RESET}")

            # Embedding Keywords
            keywords = metadata.get('keywords', '').encode('utf-16le') + b'\x00\x00'
            exif_dict["0th"][piexif.ImageIFD.XPKeywords] = keywords
            print(f"{Fore.BLUE}Embedding Keywords:{Fore.RESET} {Fore.GREEN}{metadata.get('keywords', '')}{Fore.RESET}")

            technical_value = metadata.get('technical_score', '')
            exif_dict["0th"][piexif.ImageIFD.XPSubject] = (str(technical_value).encode('utf-16le') + b'\x00\x00')
            if technical_value:
                print(f"{Fore.BLUE}Embedding technical score:{Fore.RESET} {Fore.GREEN}{technical_value}{Fore.RESET}")

            context_value = sanitize_string(metadata.get('context_label', ''))
            if context_value:
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = context_value.encode('utf-8')
                print(f"{Fore.BLUE}Embedding Context:{Fore.RESET} {Fore.GREEN}{context_value}{Fore.RESET}")

            # Prepare Exif data with sanitized strings
            exif_bytes = piexif.dump(exif_dict)

            # Backup original image
            backup_image_path = get_backup_path(image_path, backup_dir)
            if os.path.exists(image_path):  # Ensure the file exists before renaming
                import shutil
                shutil.copy2(image_path, backup_image_path)

            # Save with new metadata
            img.save(image_path, exif=exif_bytes)
            
            # Verify if requested
            if verify:
                if verify_metadata(image_path, metadata):
                    print(Fore.GREEN + f"✓ Metadata verified in {Style.BRIGHT}{image_path}{Style.RESET_ALL}" + Fore.RESET)
                else:
                    logger.warning(f"⚠ Metadata verification failed for {image_path}")
            else:
                print(Fore.GREEN + f"Metadata successfully embedded in {Style.BRIGHT}{image_path}{Style.RESET_ALL}" + Fore.RESET)
            return True

    except Exception as e:
        logger.error(f"Error embedding metadata in {image_path}: {e}")
        return False


def embed_context_only(image_path: str, context_label: str, backup_dir: Optional[str] = None) -> bool:
    """Embed only the context label into image EXIF without scoring.
    
    This is a lightweight operation for pre-classifying image sets before full evaluation.
    The context is stored in ImageDescription field.
    """
    try:
        file_ext = os.path.splitext(image_path)[1].lower()
        
        # For RAW/TIFF files, use exiftool
        if file_ext in ['.dng', '.nef', '.tif', '.tiff', '.cr2', '.cr3', '.arw', '.rw2', '.raf', '.orf']:
            cmd = ['exiftool', '-overwrite_original']
            if backup_dir:
                os.makedirs(backup_dir, exist_ok=True)
                cmd.extend(['-b', '-o', backup_dir])
            cmd.extend([
                f'-ImageDescription={context_label}',
                image_path
            ])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"exiftool failed for {image_path}: {result.stderr}")
                return False
            print(f"{Fore.CYAN}[Context]{Style.RESET_ALL} {os.path.basename(image_path)}: {Fore.GREEN}{context_label}{Style.RESET_ALL}")
            return True
        
        # For JPEG/PNG, use piexif
        if not PIEXIF_AVAILABLE or piexif is None:
            logger.error("piexif required for JPEG/PNG context embedding")
            return False
        
        with Image.open(image_path) as img:
            exif_data = img.info.get("exif", b"")
            if exif_data:
                exif_dict = piexif.load(exif_data)
            else:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
            
            # Store context in ImageDescription
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = context_label.encode('utf-8')
            
            exif_bytes = piexif.dump(exif_dict)
            
            # Backup if requested
            if backup_dir:
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, os.path.basename(image_path) + '.original')
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(image_path, backup_path)
            
            # Save with new EXIF
            img.save(image_path, exif=exif_bytes, quality=95)
        
        print(f"{Fore.CYAN}[Context]{Style.RESET_ALL} {os.path.basename(image_path)}: {Fore.GREEN}{context_label}{Style.RESET_ALL}")
        return True
        
    except Exception as e:
        logger.error(f"Error embedding context in {image_path}: {e}")
        return False


def process_context_only(
    folder_path: str,
    ollama_host_url: str,
    model: str,
    context_override: Optional[str] = None,
    csv_output: Optional[str] = None,
    backup_dir: Optional[str] = None,
    dry_run: bool = False,
    workers: int = 4,
    force: bool = False,
    retry_contexts: Optional[List[str]] = None
) -> List[Tuple[str, str, str]]:
    """Classify images and embed context only (no scoring).
    
    By default, processes images that:
    - Have no context embedded, OR
    - Have 'studio_photography' context (the fallback)
    
    Use --retry-contexts to also re-classify specific contexts.
    Use --force to re-classify ALL images regardless of existing context.
    
    Returns list of (image_path, context, confidence) tuples.
    """
    results: List[Tuple[str, str, str]] = []
    
    # Collect images
    image_paths = []
    file_types = ['.jpg', '.jpeg', '.png', '.nef', '.tif', '.tiff', '.dng', '.cr2', '.cr3', '.arw']
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if '.original.' in filename:
                continue
            if any(filename.lower().endswith(ext) for ext in file_types):
                image_path = os.path.join(root, filename)
                
                # Check existing context
                cached = read_cached_context(image_path)
                
                if force:
                    # Force mode: process everything
                    image_paths.append(image_path)
                elif cached and cached != 'studio_photography':
                    # Check if this context should be retried
                    if retry_contexts and cached in retry_contexts:
                        print(f"{Fore.MAGENTA}[Retry]{Style.RESET_ALL} {os.path.basename(image_path)}: was {cached}")
                        image_paths.append(image_path)
                    else:
                        # Has real classification - skip and record
                        results.append((image_path, cached, 'cached'))
                        print(f"{Fore.YELLOW}[Cached]{Style.RESET_ALL} {os.path.basename(image_path)}: {cached}")
                else:
                    # No context OR studio_photography fallback - needs (re)classification
                    if cached == 'studio_photography':
                        print(f"{Fore.MAGENTA}[Retry]{Style.RESET_ALL} {os.path.basename(image_path)}: was fallback")
                    image_paths.append(image_path)
    
    if not image_paths:
        print(f"{Fore.YELLOW}No images to process (all may have cached context){Style.RESET_ALL}")
        return results
    
    print(f"\n{Fore.CYAN}Processing {len(image_paths)} images for context classification...{Style.RESET_ALL}\n")
    
    # Process images
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def classify_single(image_path: str) -> Tuple[str, str, str]:
        if context_override:
            context = context_override if context_override in PROFILE_CONFIG else 'studio_photography'
            confidence = 'override'
        else:
            try:
                result = classify_image_context_detailed(image_path, ollama_host_url, model)
                context = result.context
                confidence = result.confidence
            except Exception as e:
                logger.warning(f"Classification failed for {image_path}: {e}")
                context = 'studio_photography'
                confidence = 'error'
        
        if not dry_run:
            embed_context_only(image_path, context, backup_dir)
        else:
            profile_name = get_profile_name(context)
            print(f"{Fore.MAGENTA}[Dry-run]{Style.RESET_ALL} {os.path.basename(image_path)}: {context} ({profile_name})")
        
        return (image_path, context, confidence)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_single, path): path for path in image_paths}
        
        with tqdm(total=len(image_paths), desc="Classifying", unit="img") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    path = futures[future]
                    logger.error(f"Failed to process {path}: {e}")
                    results.append((path, 'studio_photography', 'error'))
                pbar.update(1)
    
    # Write CSV if requested
    if csv_output:
        with open(csv_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'filename', 'context', 'profile_name', 'confidence'])
            for path, context, confidence in results:
                profile_name = get_profile_name(context)
                writer.writerow([path, os.path.basename(path), context, profile_name, confidence])
        print(f"\n{Fore.GREEN}Context assignments saved to: {csv_output}{Style.RESET_ALL}")
    
    # Print summary
    context_counts: Dict[str, int] = {}
    for _, context, _ in results:
        context_counts[context] = context_counts.get(context, 0) + 1
    
    print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Context Classification Summary{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    for context, count in sorted(context_counts.items(), key=lambda x: -x[1]):
        profile_name = get_profile_name(context)
        pct = 100 * count / len(results)
        print(f"  {context:25} ({profile_name:30}): {count:4} ({pct:5.1f}%)")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    print(f"  {'Total':25} {' ':30}  {len(results):4}")
    
    return results


def collect_images(folder_path: str, file_types: Optional[List[str]] = None, skip_existing: bool = True) -> List[str]:
    """Collect all image paths to process."""
    image_paths = []
    
    # Default file types if not specified
    if file_types is None:
        file_types = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.nef', '.NEF', 
                     '.tif', '.tiff', '.TIF', '.TIFF', '.dng', '.DNG']
    else:
        # Normalize extensions to include dot
        file_types = [ext if ext.startswith('.') else f'.{ext}' for ext in file_types]
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Skip files with '.original' in the name (before the extension)
            if '.original.' in filename:
                continue

            if any(filename.endswith(ext) for ext in file_types):
                image_path = os.path.join(root, filename)
                
                # Check if UserComment exists (skip if flag is set)
                if skip_existing and has_user_comment(image_path):
                    continue
                
                # Validate image is not corrupted
                if not validate_image(image_path):
                    logger.warning(f"Skipping corrupted/invalid image: {image_path}")
                    continue
                    
                image_paths.append(image_path)
    
    return image_paths


def _handle_cached_or_dry_run(image_path: str, cache_dir: Optional[str], model: str,
                              backup_dir: Optional[str], verify: bool,
                              dry_run: bool) -> Optional[Tuple[str, Optional[Dict]]]:
    """Return immediate result for dry-run or cached entries."""
    if dry_run:
        logger.debug(f"Dry run - skipping processing for {image_path}")
        return (image_path, {'score': '0', 'title': '[DRY RUN]', 'description': 'Would process this image', 'keywords': 'dry-run'})

    if cache_dir:
        cached_metadata = load_from_cache(image_path, model, cache_dir)
        if cached_metadata:
            logger.info(f"Using cached result for {image_path}")
            try:
                embed_metadata(image_path, cached_metadata, backup_dir, verify)
                return (image_path, cached_metadata)
            except Exception as e:
                logger.error(f"Error embedding cached metadata for {image_path}: {e}")
                return (image_path, None)
    return None


def process_images_with_pyiqa(
    image_paths: List[str],
    cache_model_label: str,
    classification_model: str,
    context_host_url: str,
    metadata_host_url: Optional[str],
    cache_dir: Optional[str],
    backup_dir: Optional[str],
    verify: bool,
    pyiqa_manager: PyiqaManager,
    min_score: Optional[int],
    dry_run: bool,
    context_override: Optional[str],
    skip_context_classification: bool,
    args,
    use_ollama_metadata: bool = False,
    stock_eval: bool = False
) -> List[Tuple[str, Optional[Dict]]]:
    """Profile-aware PyIQA pipeline with context detection and rule weighting."""
    results: List[Tuple[str, Optional[Dict]]] = []
    
    # Timing accumulators for performance analysis
    timing_stats = {"context": 0.0, "pyiqa": 0.0, "metadata": 0.0, "embed": 0.0, "count": 0}
    
    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
        for image_path in image_paths:
            immediate = _handle_cached_or_dry_run(
                image_path, cache_dir, cache_model_label, backup_dir, verify, dry_run=dry_run
            )
            if immediate:
                path, metadata = immediate
                if metadata and min_score is not None:
                    try:
                        score = int(validate_score(metadata.get('score', '0')) or 0)
                        if score < min_score:
                            pbar.update(1)
                            continue
                    except (ValueError, TypeError):
                        pass
                results.append((path, metadata))
                pbar.update(1)
                continue

            t0 = time.time()
            try:
                image_context, _, technical_metrics, technical_warnings = analyze_image_with_context(
                    image_path,
                    context_host_url,
                    classification_model,
                    context_override,
                    skip_context_classification,
                    stock_eval
                )
            except ImageResolutionTooSmallError as exc:
                logger.warning(str(exc))
                results.append((image_path, None))
                pbar.update(1)
                continue
            except Exception as exc:
                logger.error(f"Context/technical analysis failed for {image_path}: {exc}")
                results.append((image_path, None))
                pbar.update(1)
                continue
            t1 = time.time()
            timing_stats["context"] += t1 - t0

            profile_key = map_context_to_profile(image_context)
            profile_cfg = PROFILE_CONFIG.get(profile_key, PROFILE_CONFIG["studio_photography"])
            metric_keys = list(profile_cfg.get("model_weights", {}).keys())
            try:
                metric_details = pyiqa_manager.score_metrics(image_path, metric_keys, args)
            except Exception as exc:
                logger.error(f"PyIQA scoring failed for {image_path}: {exc}")
                results.append((image_path, None))
                pbar.update(1)
                continue
            t2 = time.time()
            timing_stats["pyiqa"] += t2 - t1

            if not metric_details:
                logger.error(f"No PyIQA metrics computed for {image_path}")
                results.append((image_path, None))
                pbar.update(1)
                continue

            calibrated_scores = {k: v["calibrated"] for k, v in metric_details.items()}
            z_scores, percentiles, fused_scores = compute_metric_z_scores(calibrated_scores)
            for metric_key, detail in metric_details.items():
                detail["z"] = z_scores.get(metric_key, 0.0)
                detail["percentile"] = percentiles.get(metric_key)
                detail["fused_score"] = fused_scores.get(metric_key)
            diff_z = compute_disagreement_z(z_scores)
            base_score, composite_z, contributions = compute_profile_composite(profile_key, z_scores, diff_z)
            rule_penalty, rule_notes = apply_profile_rules(profile_key, technical_metrics)
            final_score = max(0.0, min(100.0, base_score - rule_penalty))

            metadata = build_profile_metadata(
                profile_key=profile_key,
                context_label=image_context,
                technical_metrics=technical_metrics,
                technical_warnings=technical_warnings,
                metric_details=metric_details,
                z_scores=z_scores,
                percentiles=percentiles,
                fused_scores=fused_scores,
                diff_z=diff_z,
                composite_z=composite_z,
                base_score=base_score,
                final_score=final_score,
                rule_penalty=rule_penalty,
                rule_notes=rule_notes,
                contributions=contributions,
            )

            if use_ollama_metadata and metadata_host_url:
                llm_fields = generate_ollama_metadata(
                    image_path=image_path,
                    ollama_host_url=metadata_host_url,
                    model=classification_model,
                    profile_key=profile_key,
                    final_score=final_score,
                    technical_metrics=technical_metrics,
                    technical_warnings=technical_warnings,
                )
                if llm_fields:
                    metadata.update(llm_fields)

            if stock_eval and metadata_host_url:
                stock_fields = generate_stock_assessment(
                    image_path=image_path,
                    ollama_host_url=metadata_host_url,
                    model=classification_model,
                    technical_metrics=technical_metrics,
                    technical_warnings=technical_warnings,
                )
                if stock_fields:
                    metadata.update(stock_fields)

            if cache_dir:
                save_to_cache(image_path, cache_model_label, metadata, cache_dir)
            t3 = time.time()
            timing_stats["metadata"] += t3 - t2

            try:
                embed_metadata(image_path, metadata, backup_dir, verify)
            except Exception as exc:
                logger.error(f"Failed to embed metadata for {image_path}: {exc}")
                results.append((image_path, None))
                pbar.update(1)
                continue
            t4 = time.time()
            timing_stats["embed"] += t4 - t3
            timing_stats["count"] += 1

            logger.info(f"Composite PyIQA score for {image_path}: {metadata.get('score')}")

            if min_score is not None:
                try:
                    score_val = int(validate_score(metadata.get('score', '0')) or 0)
                    if score_val < min_score:
                        pbar.update(1)
                        continue
                except (ValueError, TypeError):
                    pass

            results.append((image_path, metadata))
            pbar.update(1)

    # Print timing summary
    if timing_stats["count"] > 0:
        n = timing_stats["count"]
        print(f"\n{Fore.CYAN}Timing Summary ({n} images):{Style.RESET_ALL}")
        print(f"  Context/Technical: {timing_stats['context']:.1f}s total, {timing_stats['context']/n:.2f}s/img")
        print(f"  PyIQA Scoring:     {timing_stats['pyiqa']:.1f}s total, {timing_stats['pyiqa']/n:.2f}s/img")
        print(f"  Metadata Build:    {timing_stats['metadata']:.1f}s total, {timing_stats['metadata']/n:.2f}s/img")
        print(f"  Embed to EXIF:     {timing_stats['embed']:.1f}s total, {timing_stats['embed']/n:.2f}s/img")
        total = timing_stats['context'] + timing_stats['pyiqa'] + timing_stats['metadata'] + timing_stats['embed']
        print(f"  {Fore.GREEN}Total:               {total:.1f}s, {total/n:.2f}s/img{Style.RESET_ALL}")

    return results


def process_images_in_folder(folder_path: str, ollama_host_url: str, context_host_url: Optional[str] = None,
                            model: str = DEFAULT_MODEL,
                            file_types: Optional[List[str]] = None, skip_existing: bool = True,
                            dry_run: bool = False, min_score: Optional[int] = None,
                            backup_dir: Optional[str] = None, verify: bool = False,
                            cache_dir: Optional[str] = None,
                            context_override: Optional[str] = None,
                            skip_context_classification: bool = False,
                            pyiqa_manager: Optional[PyiqaManager] = None,
                            pyiqa_cache_label: Optional[str] = None,
                            cli_args: Optional[argparse.Namespace] = None,
                            use_ollama_metadata: bool = False,
                            stock_eval: bool = False) -> List[Tuple[str, Optional[Dict]]]:
    """Process images end-to-end using the PyIQA composite pipeline."""
    if pyiqa_manager is None:
        raise ValueError("PyIQA manager must be initialized before processing images.")

    image_paths = collect_images(folder_path, file_types=file_types, skip_existing=skip_existing)
    if not image_paths:
        logger.warning("No images found to process")
        return []

    cache_label = pyiqa_cache_label or "pyiqa_profiles"
    effective_stock_eval = stock_eval or bool(cli_args and getattr(cli_args, 'stock_eval', False))
    metadata_host = None
    if use_ollama_metadata or effective_stock_eval:
        metadata_host = ollama_host_url
    return process_images_with_pyiqa(
        image_paths=image_paths,
        cache_model_label=cache_label,
        classification_model=model,
        context_host_url=context_host_url or ollama_host_url,
        metadata_host_url=metadata_host,
        cache_dir=cache_dir,
        backup_dir=backup_dir,
        verify=verify,
        pyiqa_manager=pyiqa_manager,
        min_score=min_score,
        dry_run=dry_run,
        context_override=context_override,
        skip_context_classification=skip_context_classification,
        args=cli_args,
        use_ollama_metadata=use_ollama_metadata,
        stock_eval=effective_stock_eval
    )


def save_results_to_csv(results: List[Tuple[str, Optional[Dict]]], output_path: str):
    """Export evaluation results to a CSV file.
    
    Flattens nested metadata dictionaries and writes all fields as columns.
    """
    """Save processing results to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        # Collect all model keys from actual results
        model_keys = set()
        for _, metadata in results:
            if metadata:
                pyiqa_details = metadata.get('technical_metrics', {}).get('pyiqa_metric_details', {})
                model_keys.update(pyiqa_details.keys())
        model_fieldnames: List[str] = []
        for model_key in sorted(model_keys):
            model_fieldnames.extend([
                f"{model_key}_calibrated",
                f"{model_key}_z",
                f"{model_key}_percentile",
                f"{model_key}_fused_score",
            ])
        fieldnames = [
            'file_path', 'overall_score', 'technical_score', 'composition_score',
            'lighting_score', 'creativity_score', 'title', 'description',
            'keywords', 'status', 'context', 'context_profile', 'sharpness', 'brightness', 'contrast',
            'histogram_clipping_highlights', 'histogram_clipping_shadows',
            'color_cast', 'color_cast_delta', 'noise_sigma', 'noise_score', 'technical_warnings', 'post_process_potential',
            'resolution_mp', 'dpi', 'stock_notes', 'stock_fixable',
            'stock_overall_score', 'stock_recommendation', 'stock_primary_category',
            'stock_commercial_viability', 'stock_technical_quality', 'stock_composition_clarity',
            'stock_keyword_potential', 'stock_release_concerns', 'stock_rejection_risks',
            'stock_llm_notes', 'stock_llm_issues',
            'pyiqa_composite_z', 'pyiqa_composite_fused_score', 'pyiqa_disagreement_z'
        ] + model_fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for file_path, metadata in results:
            if metadata:
                technical_metrics = metadata.get('technical_metrics', {})
                warnings = metadata.get('technical_warnings', [])
                warnings_str = '; '.join(warnings) if warnings else ''
                pyiqa_details = technical_metrics.get('pyiqa_metric_details', {})
                row: Dict[str, Any] = {
                    'file_path': file_path,
                    'overall_score': metadata.get('overall_score', metadata.get('score', '')),
                    'technical_score': metadata.get('technical_score', ''),
                    'composition_score': metadata.get('composition_score', ''),
                    'lighting_score': metadata.get('lighting_score', ''),
                    'creativity_score': metadata.get('creativity_score', ''),
                    'title': metadata.get('title', ''),
                    'description': metadata.get('description', ''),
                    'keywords': metadata.get('keywords', ''),
                    'context': technical_metrics.get('context', ''),
                    'context_profile': technical_metrics.get('context_profile', ''),
                    'sharpness': technical_metrics.get('sharpness', ''),
                    'brightness': technical_metrics.get('brightness', ''),
                    'contrast': technical_metrics.get('contrast', ''),
                    'histogram_clipping_highlights': technical_metrics.get('histogram_clipping_highlights', ''),
                    'histogram_clipping_shadows': technical_metrics.get('histogram_clipping_shadows', ''),
                    'color_cast': technical_metrics.get('color_cast', ''),
                    'color_cast_delta': technical_metrics.get('color_cast_delta', ''),
                    'noise_sigma': technical_metrics.get('noise_sigma', ''),
                    'noise_score': technical_metrics.get('noise_score', ''),
                    'technical_warnings': warnings_str,
                    'post_process_potential': metadata.get('post_process_potential', ''),
                    'resolution_mp': technical_metrics.get('megapixels', ''),
                    'dpi': technical_metrics.get('dpi_x', ''),
                    'stock_notes': '; '.join(technical_metrics.get('stock_notes', [])),
                    'stock_fixable': '; '.join(technical_metrics.get('stock_fixable', [])),
                    'stock_overall_score': metadata.get('stock_overall_score', ''),
                    'stock_recommendation': metadata.get('stock_recommendation', ''),
                    'stock_primary_category': metadata.get('stock_primary_category', ''),
                    'stock_commercial_viability': metadata.get('stock_commercial_viability', ''),
                    'stock_technical_quality': metadata.get('stock_technical_quality', ''),
                    'stock_composition_clarity': metadata.get('stock_composition_clarity', ''),
                    'stock_keyword_potential': metadata.get('stock_keyword_potential', ''),
                    'stock_release_concerns': metadata.get('stock_release_concerns', ''),
                    'stock_rejection_risks': metadata.get('stock_rejection_risks', ''),
                    'stock_llm_notes': metadata.get('stock_llm_notes', ''),
                    'stock_llm_issues': metadata.get('stock_llm_issues', ''),
                    'pyiqa_composite_z': technical_metrics.get('pyiqa_composite_z', ''),
                    'pyiqa_composite_fused_score': technical_metrics.get('pyiqa_composite_fused_score', ''),
                    'pyiqa_disagreement_z': technical_metrics.get('pyiqa_disagreement_z', ''),
                    'status': 'success'
                }

                for model_key in sorted(IQA_CALIBRATION.keys()):
                    detail = pyiqa_details.get(model_key, {}) if isinstance(pyiqa_details, dict) else {}
                    row[f"{model_key}_calibrated"] = detail.get('calibrated', '')
                    row[f"{model_key}_z"] = detail.get('z', '')
                    row[f"{model_key}_percentile"] = detail.get('percentile', '')
                    row[f"{model_key}_fused_score"] = detail.get('fused_score', '')

                writer.writerow(row)
            else:
                empty_model_fields = {
                    f"{model_key}_{suffix}": ''
                    for model_key in sorted(IQA_CALIBRATION.keys())
                    for suffix in ["calibrated", "z", "percentile", "fused_score"]
                }
                failure_row = {
                    'file_path': file_path,
                    'overall_score': '',
                    'technical_score': '',
                    'composition_score': '',
                    'lighting_score': '',
                    'creativity_score': '',
                    'title': '',
                    'description': '',
                    'keywords': '',
                    'status': 'failed',
                    'context': '',
                    'context_profile': '',
                    'sharpness': '',
                    'brightness': '',
                    'contrast': '',
                    'histogram_clipping_highlights': '',
                    'histogram_clipping_shadows': '',
                    'color_cast': '',
                    'color_cast_delta': '',
                    'noise_sigma': '',
                    'noise_score': '',
                    'technical_warnings': '',
                    'post_process_potential': '',
                    'resolution_mp': '',
                    'dpi': '',
                    'stock_notes': '',
                    'stock_fixable': '',
                    'stock_overall_score': '',
                    'stock_recommendation': '',
                    'stock_primary_category': '',
                    'stock_commercial_viability': '',
                    'stock_technical_quality': '',
                    'stock_composition_clarity': '',
                    'stock_keyword_potential': '',
                    'stock_release_concerns': '',
                    'stock_rejection_risks': '',
                    'stock_llm_notes': '',
                    'stock_llm_issues': '',
                    'pyiqa_composite_z': '',
                    'pyiqa_composite_fused_score': '',
                    'pyiqa_disagreement_z': '',
                }
                failure_row.update(empty_model_fields)
                writer.writerow(failure_row)


def load_results_from_csv(csv_path: str) -> List[Tuple[str, Optional[Dict]]]:
    """Load previously saved results so statistics can be recomputed without reprocessing."""
    results: List[Tuple[str, Optional[Dict]]] = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata: Optional[Dict[str, Any]]
            if row.get('status') == 'failed':
                metadata = None
            else:
                # Build technical_metrics dict from CSV columns
                technical_metrics: Dict[str, Any] = {}
                
                # Parse numeric technical metrics
                for metric_key in ['sharpness', 'brightness', 'contrast', 
                                   'histogram_clipping_highlights', 'histogram_clipping_shadows',
                                   'color_cast_delta', 'noise_score', 'noise_sigma', 'resolution_mp']:
                    val_str = row.get(metric_key, '')
                    if val_str:
                        try:
                            technical_metrics[metric_key] = float(val_str)
                        except ValueError:
                            pass
                
                # Rename resolution_mp to megapixels for consistency
                if 'resolution_mp' in technical_metrics:
                    technical_metrics['megapixels'] = technical_metrics.pop('resolution_mp')
                
                # Parse color_cast_delta from color_cast field if present
                color_cast = row.get('color_cast', '')
                if color_cast:
                    technical_metrics['color_cast'] = color_cast
                
                # Parse IQA model scores from description
                # Format: "clipiqa_z:56.6 (z=-1.48); laion_aes_z:46.8 (z=-2.91); ..."
                pyiqa_details: Dict[str, Dict[str, float]] = {}
                description = row.get('description', '')
                if 'Model breakdown:' in description:
                    breakdown_part = description.split('Model breakdown:')[1]
                    # Find where breakdown ends (usually at "Disagreement" or end of string)
                    if 'Disagreement' in breakdown_part:
                        breakdown_part = breakdown_part.split('Disagreement')[0]
                    
                    # Parse each model: "clipiqa_z:56.6 (z=-1.48)"
                    import re
                    pattern = r'(\w+):([0-9.]+)\s*\(z=([+-]?[0-9.]+)\)'
                    for match in re.finditer(pattern, breakdown_part):
                        model_name = match.group(1)
                        calibrated = float(match.group(2))
                        z_score = float(match.group(3))
                        pyiqa_details[model_name] = {'calibrated': calibrated, 'z': z_score}
                
                if pyiqa_details:
                    technical_metrics['pyiqa_metric_details'] = pyiqa_details
                
                metadata = {
                    'overall_score': row.get('overall_score', ''),
                    'score': row.get('overall_score', ''),
                    'technical_score': row.get('technical_score', ''),
                    'composition_score': row.get('composition_score', ''),
                    'lighting_score': row.get('lighting_score', ''),
                    'creativity_score': row.get('creativity_score', ''),
                    'title': row.get('title', ''),
                    'description': row.get('description', ''),
                    'keywords': row.get('keywords', ''),
                    'technical_warnings': row.get('technical_warnings', '').split('; ') if row.get('technical_warnings') else [],
                    'post_process_potential': row.get('post_process_potential', ''),
                    'technical_metrics': technical_metrics,
                }
            results.append((row.get('file_path', ''), metadata))
    return results


def get_cache_stats(cache_dir: str) -> Dict:
    """Get cache statistics."""
    if not cache_dir or not os.path.exists(cache_dir):
        return {'enabled': False, 'entries': 0, 'size_mb': 0}
    
    try:
        cache_files = list(Path(cache_dir).glob('*.cache'))
        total_size = sum(f.stat().st_size for f in cache_files)
        return {
            'enabled': True,
            'entries': len(cache_files),
            'size_mb': total_size / (1024 * 1024)
        }
    except Exception as e:
        logger.warning(f"Error getting cache stats: {e}")
        return {'enabled': False, 'entries': 0, 'size_mb': 0}


def compute_metric_stats(values: List[float]) -> Dict[str, Any]:
    """Compute comprehensive statistics for a list of values."""
    if not values:
        return {}
    n = len(values)
    sorted_vals = sorted(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0
    std_dev = variance ** 0.5
    
    # Quartiles
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    q1 = sorted_vals[n // 4] if n >= 4 else sorted_vals[0]
    q3 = sorted_vals[(3 * n) // 4] if n >= 4 else sorted_vals[-1]
    
    return {
        'count': n,
        'mean': mean,
        'std': std_dev,
        'min': min(values),
        'max': max(values),
        'median': median,
        'q1': q1,
        'q3': q3,
        'values': values,  # Keep raw values for histogram
    }


def build_histogram(values: List[float], bins: int = 10, max_width: int = 25) -> List[Tuple[str, int, str]]:
    """Build ASCII histogram bins from values.
    
    Returns list of (bin_label, count, bar_string) tuples.
    """
    if not values:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    # Handle edge case where all values are the same
    if min_val == max_val:
        return [(f"{min_val:.1f}", len(values), '█' * max_width)]
    
    bin_width = (max_val - min_val) / bins
    histogram: List[Tuple[str, int]] = []
    
    for i in range(bins):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        if i == bins - 1:
            # Last bin includes max value
            count = sum(1 for v in values if bin_start <= v <= bin_end)
        else:
            count = sum(1 for v in values if bin_start <= v < bin_end)
        label = f"{bin_start:.1f}-{bin_end:.1f}"
        histogram.append((label, count))
    
    # Build bar strings
    max_count = max(c for _, c in histogram) if histogram else 1
    scale = max_count / max_width if max_count > max_width else 1
    
    result = []
    for label, count in histogram:
        bar_len = max(1, int(count / scale)) if count > 0 else 0
        bar = '█' * bar_len
        result.append((label, count, bar))
    
    return result


def calculate_statistics(results: List[Tuple[str, Optional[Dict]]]) -> Dict:
    """Calculate statistics from processing results."""
    scores = []
    tech_scores: List[int] = []
    raw_count = 0
    pil_count = 0
    potentials = []
    
    # Collect all technical metrics
    technical_metrics_collected: Dict[str, List[float]] = {
        'sharpness': [],
        'noise_score': [],
        'histogram_clipping_highlights': [],
        'histogram_clipping_shadows': [],
        'color_cast_delta': [],
        'brightness': [],
        'contrast': [],
        'megapixels': [],
    }
    
    # Collect all IQA metric scores
    iqa_metrics_collected: Dict[str, List[float]] = {}
    
    for image_path, metadata in results:
        # Track format types
        if image_path:
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext in ['.dng', '.nef', '.tif', '.tiff']:
                raw_count += 1
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                pil_count += 1
        
        if metadata and 'score' in metadata:
            try:
                # Try to extract numeric score from string
                score_str = str(metadata['score'])
                # Extract first number found
                match = re.search(r'\d+', score_str)
                if match:
                    score = int(match.group())
                    if SCORE_MIN <= score <= SCORE_MAX:
                        scores.append(score)
            except (ValueError, AttributeError):
                continue
        if metadata and metadata.get('technical_score'):
            try:
                tech_val = int(str(metadata['technical_score']))
                if SCORE_MIN <= tech_val <= SCORE_MAX:
                    tech_scores.append(tech_val)
            except (ValueError, TypeError):
                pass
        if metadata and 'post_process_potential' in metadata:
            try:
                potentials.append(int(metadata['post_process_potential']))
            except (ValueError, TypeError):
                pass
        
        # Collect technical metrics from metadata
        if metadata:
            tech_metrics = metadata.get('technical_metrics', {})
            for metric_name in technical_metrics_collected.keys():
                val = tech_metrics.get(metric_name)
                if isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):  # Exclude NaN
                    technical_metrics_collected[metric_name].append(float(val))
            
            # Collect IQA metric details
            pyiqa_details = tech_metrics.get('pyiqa_metric_details', {})
            for model_name, detail in pyiqa_details.items():
                if isinstance(detail, dict) and 'calibrated' in detail:
                    cal_val = detail['calibrated']
                    if isinstance(cal_val, (int, float)):
                        if model_name not in iqa_metrics_collected:
                            iqa_metrics_collected[model_name] = []
                        iqa_metrics_collected[model_name].append(float(cal_val))
    
    # Count images with significant technical issues:
    # - At least one critical flag, OR
    # - At least two different metrics with warn flags
    warning_images = sum(
        1 for _, md in results 
        if md and is_technically_warned(md.get('technical_metrics', {}))
    )

    if not scores:
        return {
            'total_processed': len(results),
            'successful': 0,
            'failed': len(results),
            'avg_score': 0,
            'min_score': 0,
            'max_score': 0,
            'score_distribution': {},
            'technical_score_distribution': {},
        'raw_count': raw_count,
        'pil_count': pil_count,
        'warning_images': warning_images,
        'avg_post_process_potential': sum(potentials)/len(potentials) if potentials else 0
        }
    
    # Calculate score distribution (bins of 5)
    distribution = {}
    for i in range(0, 100, 5):
        bin_label = f"{i}-{i+4}"
        distribution[bin_label] = sum(1 for s in scores if i <= s < i+5)
    distribution["95-100"] = sum(1 for s in scores if 95 <= s <= 100)

    tech_distribution: Dict[str, int] = {}
    if tech_scores:
        for i in range(0, 100, 5):
            bin_label = f"{i}-{i+4}"
            tech_distribution[bin_label] = sum(1 for s in tech_scores if i <= s < i+5)
        tech_distribution["95-100"] = sum(1 for s in tech_scores if 95 <= s <= 100)
    
    # Calculate statistical measures
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # Standard deviation
    if len(scores) > 1:
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
    else:
        std_dev = 0
    
    # Median and quartiles
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    median = sorted_scores[n // 2] if n % 2 == 1 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    q1 = sorted_scores[n // 4] if n >= 4 else sorted_scores[0]
    q3 = sorted_scores[(3 * n) // 4] if n >= 4 else sorted_scores[-1]
    
    tech_avg = (sum(tech_scores) / len(tech_scores)) if tech_scores else None
    tech_min = min(tech_scores) if tech_scores else None
    tech_max = max(tech_scores) if tech_scores else None
    if tech_scores:
        sorted_tech = sorted(tech_scores)
        tn = len(sorted_tech)
        tech_median = sorted_tech[tn // 2] if tn % 2 == 1 else (sorted_tech[tn // 2 - 1] + sorted_tech[tn // 2]) / 2
        tech_q1 = sorted_tech[tn // 4] if tn >= 4 else sorted_tech[0]
        tech_q3 = sorted_tech[(3 * tn) // 4] if tn >= 4 else sorted_tech[-1]
        if tn > 1:
            tech_var = sum((s - tech_avg) ** 2 for s in tech_scores) / tn  # type: ignore
            tech_std = tech_var ** 0.5
        else:
            tech_std = 0
    else:
        tech_median = tech_q1 = tech_q3 = tech_std = None

    return {
        'total_processed': len(results),
        'successful': len(scores),
        'failed': len(results) - len(scores),
        'avg_score': avg_score,
        'median_score': median,
        'std_dev': std_dev,
        'q1': q1,
        'q3': q3,
        'min_score': min(scores) if scores else 0,
        'max_score': max(scores) if scores else 0,
        'score_distribution': distribution,
        'raw_count': raw_count,
        'pil_count': pil_count,
        'warning_images': warning_images,
        'avg_post_process_potential': sum(potentials)/len(potentials) if potentials else 0,
        'technical_score_distribution': tech_distribution,
        'avg_technical_score': tech_avg,
        'technical_min_score': tech_min,
        'technical_max_score': tech_max,
        'technical_median_score': tech_median,
        'technical_q1': tech_q1,
        'technical_q3': tech_q3,
        'technical_std_dev': tech_std,
        # New: comprehensive metric statistics
        'technical_metrics_stats': {k: compute_metric_stats(v) for k, v in technical_metrics_collected.items() if v},
        'iqa_metrics_stats': {k: compute_metric_stats(v) for k, v in iqa_metrics_collected.items() if v},
    }


# Threshold annotations for histogram display
# Format: metric_name -> list of (threshold_value, label, direction)
# direction: "below" means values below threshold are flagged, "above" means values above are flagged
METRIC_THRESHOLDS: Dict[str, List[Tuple[float, str, str]]] = {
    'sharpness': [
        (STOCK_SHARPNESS_CRITICAL, "CRITICAL", "below"),
        (STOCK_SHARPNESS_WARN, "warn", "below"),
    ],
    'noise_score': [
        (STOCK_NOISE_WARN, "warn", "above"),
        (STOCK_NOISE_CRITICAL, "CRITICAL", "above"),
    ],
    'histogram_clipping_highlights': [
        (STOCK_CLIPPING_HIGHLIGHTS_WARN, "warn", "above"),
        (STOCK_CLIPPING_HIGHLIGHTS_CRITICAL, "CRITICAL", "above"),
    ],
    'histogram_clipping_shadows': [
        (STOCK_CLIPPING_SHADOWS_WARN, "warn", "above"),
        (STOCK_CLIPPING_SHADOWS_CRITICAL, "CRITICAL", "above"),
    ],
    'color_cast_delta': [
        (COLOR_CAST_WARN, "warn", "above"),
        (COLOR_CAST_CRITICAL, "CRITICAL", "above"),
    ],
}


def print_metric_stats(name: str, stat: Dict[str, Any], show_histogram: bool = True):
    """Print statistics and histogram for a single metric with threshold annotations."""
    if not stat:
        return
    
    # Get thresholds for this metric
    thresholds = METRIC_THRESHOLDS.get(name, [])
    
    # Build threshold info string
    threshold_info = ""
    if thresholds:
        parts = []
        for thresh_val, label, direction in thresholds:
            symbol = "<" if direction == "below" else ">"
            parts.append(f"{label}{symbol}{thresh_val:.1f}")
        threshold_info = f"  [Thresholds: {', '.join(parts)}]"
    
    print(f"\n  {name}:{threshold_info}")
    print(f"    Count: {stat['count']}")
    print(f"    Mean: {stat['mean']:.2f}  Std: {stat['std']:.2f}")
    print(f"    Range: {stat['min']:.2f} - {stat['max']:.2f}")
    print(f"    Quartiles: Q1={stat['q1']:.2f}  Median={stat['median']:.2f}  Q3={stat['q3']:.2f}")
    
    # Count values that exceed thresholds
    if thresholds and stat.get('values'):
        flagged_counts = []
        for thresh_val, label, direction in thresholds:
            if direction == "below":
                count = sum(1 for v in stat['values'] if v < thresh_val)
            else:
                count = sum(1 for v in stat['values'] if v > thresh_val)
            if count > 0:
                pct = 100 * count / len(stat['values'])
                flagged_counts.append(f"{label}: {count} ({pct:.1f}%)")
        if flagged_counts:
            print(f"    Flagged: {', '.join(flagged_counts)}")
    
    if show_histogram and stat.get('values'):
        histogram = build_histogram(stat['values'], bins=10, max_width=25)
        if histogram:
            print(f"    Distribution:")
            for label, count, bar in histogram:
                if count > 0:
                    # Check if this bin crosses any threshold
                    bin_parts = label.split('-')
                    try:
                        bin_start = float(bin_parts[0])
                        bin_end = float(bin_parts[1])
                        
                        markers = []
                        for thresh_val, thresh_label, direction in thresholds:
                            if bin_start <= thresh_val <= bin_end:
                                markers.append(f"◄{thresh_label}")
                        marker_str = ' '.join(markers)
                        if marker_str:
                            print(f"      {label:>15}: {bar} ({count}) {marker_str}")
                        else:
                            print(f"      {label:>15}: {bar} ({count})")
                    except (ValueError, IndexError):
                        print(f"      {label:>15}: {bar} ({count})")


def print_statistics(stats: Dict):
    """Print formatted summary statistics to console with color coding."""
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {stats['total_processed']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    if stats.get('warning_images'):
        print(f"Images with technical warnings: {stats['warning_images']}")
    print(f"RAW formats (exiftool): {stats.get('raw_count', 0)}")
    print(f"Standard formats (PIL): {stats.get('pil_count', 0)}")
    
    # Add timing information
    if 'elapsed_time' in stats:
        elapsed = stats['elapsed_time']
        print(f"\n{'='*60}")
        print(f"TIMING STATISTICS")
        print(f"{'='*60}")
        print(f"Total processing time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        if stats['total_processed'] > 0:
            print(f"Time per image: {elapsed/stats['total_processed']:.2f} seconds")
        if stats['successful'] > 0:
            print(f"Time per successful image: {elapsed/stats['successful']:.2f} seconds")
    
    if stats['successful'] > 0:
        print(f"\n{'='*60}")
        print(f"SCORE STATISTICS")
        print(f"{'='*60}")
        print(f"Average score: {stats['avg_score']:.2f}")
        print(f"Median score: {stats.get('median_score', 0):.2f}")
        print(f"Standard deviation: {stats.get('std_dev', 0):.2f}")
        print(f"Range: {stats['min_score']} - {stats['max_score']}")
        print(f"Quartiles (Q1/Q3): {stats.get('q1', 0):.0f} / {stats.get('q3', 0):.0f}")
        if stats.get('technical_score_distribution') and stats.get('avg_technical_score') is not None:
            print(f"\n{'-'*60}")
            print(f"TECHNICAL SCORE SUMMARY")
            print(f"{'-'*60}")
            print(f"Average technical score: {stats.get('avg_technical_score', 0):.2f}")
            print(f"Median technical score: {stats.get('technical_median_score', 0):.2f}")
            print(f"Technical std dev: {stats.get('technical_std_dev', 0):.2f}")
            print(f"Technical range: {stats.get('technical_min_score', 0)} - {stats.get('technical_max_score', 0)}")
            print(f"Technical quartiles (Q1/Q3): {stats.get('technical_q1', 0):.0f} / {stats.get('technical_q3', 0):.0f}")
        avg_post = stats.get('avg_post_process_potential')
        if avg_post is not None:
            print(f"Average post-process potential: {avg_post:.1f}/100")
        
        print(f"\n{'='*60}")
        print(f"SCORE DISTRIBUTION")
        print(f"{'='*60}")
        score_bins_sorted = sorted(
            stats['score_distribution'].items(),
            key=lambda item: int(item[0].split('-')[0])
        )
        max_overall = max((count for _, count in score_bins_sorted), default=1)
        scale_overall = max_overall / 80 if max_overall > 80 else 1
        for bin_range, count in score_bins_sorted:
            bar = '█' * max(1, int(count / scale_overall))
            print(f"{bin_range:>8}: {bar} ({count})")
        tech_dist = stats.get('technical_score_distribution')
        if tech_dist and stats.get('avg_technical_score') is not None:
            print(f"\n{'='*60}")
            print(f"TECHNICAL SCORE DISTRIBUTION")
            print(f"{'='*60}")
            tech_bins_sorted = sorted(
                tech_dist.items(),
                key=lambda item: int(item[0].split('-')[0])
            )
            max_tech = max((count for _, count in tech_bins_sorted), default=1)
            scale_tech = max_tech / 80 if max_tech > 80 else 1
            for bin_range, count in tech_bins_sorted:
                bar = '█' * max(1, int(count / scale_tech))
                print(f"{bin_range:>8}: {bar} ({count})")
        
        # Print detailed technical metrics statistics
        tech_metrics_stats = stats.get('technical_metrics_stats', {})
        if tech_metrics_stats:
            print(f"\n{'='*60}")
            print(f"TECHNICAL METRICS DETAILS")
            print(f"{'='*60}")
            # Order metrics for consistent display
            metric_order = ['sharpness', 'noise_score', 'histogram_clipping_highlights', 
                          'histogram_clipping_shadows', 'color_cast_delta', 'brightness', 
                          'contrast', 'megapixels']
            for metric_name in metric_order:
                if metric_name in tech_metrics_stats:
                    print_metric_stats(metric_name, tech_metrics_stats[metric_name])
            # Print any additional metrics not in the order list
            for metric_name, stat in tech_metrics_stats.items():
                if metric_name not in metric_order:
                    print_metric_stats(metric_name, stat)
        
        # Print IQA model statistics
        iqa_metrics_stats = stats.get('iqa_metrics_stats', {})
        if iqa_metrics_stats:
            print(f"\n{'='*60}")
            print(f"IQA MODEL SCORE DETAILS")
            print(f"{'='*60}")
            for model_name in sorted(iqa_metrics_stats.keys()):
                print_metric_stats(model_name, iqa_metrics_stats[model_name])
    
    print(f"\n{'='*60}")
    print(f"Note: JPEG/PNG use PIL, RAW/TIFF use exiftool for metadata embedding")
    print(f"{'='*60}\n")


def format_metric_stats_markdown(name: str, stat: Dict[str, Any], show_histogram: bool = True) -> List[str]:
    """Format statistics and histogram for a single metric as markdown lines."""
    if not stat:
        return []
    
    lines = []
    thresholds = METRIC_THRESHOLDS.get(name, [])
    
    # Header with thresholds
    threshold_info = ""
    if thresholds:
        parts = []
        for thresh_val, label, direction in thresholds:
            symbol = "<" if direction == "below" else ">"
            parts.append(f"`{label}{symbol}{thresh_val:.1f}`")
        threshold_info = f" — Thresholds: {', '.join(parts)}"
    
    lines.append(f"\n#### {name}{threshold_info}")
    lines.append(f"- **Count:** {stat['count']}")
    lines.append(f"- **Mean:** {stat['mean']:.2f} | **Std:** {stat['std']:.2f}")
    lines.append(f"- **Range:** {stat['min']:.2f} – {stat['max']:.2f}")
    lines.append(f"- **Quartiles:** Q1={stat['q1']:.2f} | Median={stat['median']:.2f} | Q3={stat['q3']:.2f}")
    
    # Count values that exceed thresholds
    if thresholds and stat.get('values'):
        flagged_parts = []
        for thresh_val, label, direction in thresholds:
            if direction == "below":
                count = sum(1 for v in stat['values'] if v < thresh_val)
            else:
                count = sum(1 for v in stat['values'] if v > thresh_val)
            if count > 0:
                pct = 100 * count / len(stat['values'])
                flagged_parts.append(f"**{label}:** {count} ({pct:.1f}%)")
        if flagged_parts:
            lines.append(f"- **Flagged:** {', '.join(flagged_parts)}")
    
    if show_histogram and stat.get('values'):
        histogram = build_histogram(stat['values'], bins=10, max_width=25)
        if histogram:
            lines.append("")
            lines.append("```")
            for label, count, bar in histogram:
                if count > 0:
                    bin_parts = label.split('-')
                    try:
                        bin_start = float(bin_parts[0])
                        bin_end = float(bin_parts[1])
                        markers = []
                        for thresh_val, thresh_label, direction in thresholds:
                            if bin_start <= thresh_val <= bin_end:
                                markers.append(f"◄{thresh_label}")
                        marker_str = ' '.join(markers)
                        if marker_str:
                            lines.append(f"{label:>15}: {bar} ({count}) {marker_str}")
                        else:
                            lines.append(f"{label:>15}: {bar} ({count})")
                    except (ValueError, IndexError):
                        lines.append(f"{label:>15}: {bar} ({count})")
            lines.append("```")
    
    return lines


def generate_statistics_markdown(stats: Dict, csv_path: str = "") -> str:
    """Generate markdown-formatted statistics report."""
    lines = []
    
    # Title
    if csv_path:
        lines.append(f"# Image Evaluation Statistics")
        lines.append(f"**Source:** `{csv_path}`\n")
    else:
        lines.append("# Image Evaluation Statistics\n")
    
    # Processing Summary
    lines.append("## Processing Summary")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Processed | {stats['total_processed']} |")
    lines.append(f"| Successful | {stats['successful']} |")
    lines.append(f"| Failed | {stats['failed']} |")
    if stats.get('warning_images'):
        lines.append(f"| Technical Warnings | {stats['warning_images']} |")
    lines.append(f"| RAW Formats | {stats.get('raw_count', 0)} |")
    lines.append(f"| Standard Formats | {stats.get('pil_count', 0)} |")
    
    # Timing
    if 'elapsed_time' in stats:
        elapsed = stats['elapsed_time']
        lines.append("")
        lines.append("## Timing")
        lines.append(f"- **Total Time:** {elapsed:.2f}s ({elapsed/60:.2f} min)")
        if stats['total_processed'] > 0:
            lines.append(f"- **Per Image:** {elapsed/stats['total_processed']:.2f}s")
    
    if stats['successful'] > 0:
        # Score Statistics
        lines.append("")
        lines.append("## Score Statistics")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Average | {stats['avg_score']:.2f} |")
        lines.append(f"| Median | {stats.get('median_score', 0):.2f} |")
        lines.append(f"| Std Dev | {stats.get('std_dev', 0):.2f} |")
        lines.append(f"| Range | {stats['min_score']} – {stats['max_score']} |")
        lines.append(f"| Q1 / Q3 | {stats.get('q1', 0):.0f} / {stats.get('q3', 0):.0f} |")
        
        # Technical Score Summary
        if stats.get('technical_score_distribution') and stats.get('avg_technical_score') is not None:
            lines.append("")
            lines.append("### Technical Score Summary")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Average | {stats.get('avg_technical_score', 0):.2f} |")
            lines.append(f"| Median | {stats.get('technical_median_score', 0):.2f} |")
            lines.append(f"| Std Dev | {stats.get('technical_std_dev', 0):.2f} |")
            lines.append(f"| Range | {stats.get('technical_min_score', 0)} – {stats.get('technical_max_score', 0)} |")
            lines.append(f"| Q1 / Q3 | {stats.get('technical_q1', 0):.0f} / {stats.get('technical_q3', 0):.0f} |")
        
        avg_post = stats.get('avg_post_process_potential')
        if avg_post is not None:
            lines.append(f"| Post-Process Potential | {avg_post:.1f}/100 |")
        
        # Score Distribution
        lines.append("")
        lines.append("## Score Distribution")
        lines.append("```")
        score_bins_sorted = sorted(
            stats['score_distribution'].items(),
            key=lambda item: int(item[0].split('-')[0])
        )
        max_overall = max((count for _, count in score_bins_sorted), default=1)
        scale_overall = max_overall / 40 if max_overall > 40 else 1
        for bin_range, count in score_bins_sorted:
            bar = '█' * max(1, int(count / scale_overall)) if count > 0 else ''
            lines.append(f"{bin_range:>8}: {bar} ({count})")
        lines.append("```")
        
        # Technical Score Distribution
        tech_dist = stats.get('technical_score_distribution')
        if tech_dist and stats.get('avg_technical_score') is not None:
            lines.append("")
            lines.append("## Technical Score Distribution")
            lines.append("```")
            tech_bins_sorted = sorted(
                tech_dist.items(),
                key=lambda item: int(item[0].split('-')[0])
            )
            max_tech = max((count for _, count in tech_bins_sorted), default=1)
            scale_tech = max_tech / 40 if max_tech > 40 else 1
            for bin_range, count in tech_bins_sorted:
                bar = '█' * max(1, int(count / scale_tech)) if count > 0 else ''
                lines.append(f"{bin_range:>8}: {bar} ({count})")
            lines.append("```")
        
        # Technical Metrics Details
        tech_metrics_stats = stats.get('technical_metrics_stats', {})
        if tech_metrics_stats:
            lines.append("")
            lines.append("## Technical Metrics Details")
            metric_order = ['sharpness', 'noise_score', 'histogram_clipping_highlights', 
                          'histogram_clipping_shadows', 'color_cast_delta', 'brightness', 
                          'contrast', 'megapixels']
            for metric_name in metric_order:
                if metric_name in tech_metrics_stats:
                    lines.extend(format_metric_stats_markdown(metric_name, tech_metrics_stats[metric_name]))
            for metric_name, stat in tech_metrics_stats.items():
                if metric_name not in metric_order:
                    lines.extend(format_metric_stats_markdown(metric_name, stat))
        
        # IQA Model Details
        iqa_metrics_stats = stats.get('iqa_metrics_stats', {})
        if iqa_metrics_stats:
            lines.append("")
            lines.append("## IQA Model Score Details")
            for model_name in sorted(iqa_metrics_stats.keys()):
                lines.extend(format_metric_stats_markdown(model_name, iqa_metrics_stats[model_name]))
    
    lines.append("")
    lines.append("---")
    lines.append("*Note: JPEG/PNG use PIL, RAW/TIFF use exiftool for metadata embedding*")
    
    return '\n'.join(lines)


def rollback_images(folder_path: str, backup_dir: Optional[str] = None):
    """Restore images from backups."""
    restored_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if '.original' in filename:
                backup_path = os.path.join(root, filename)
                
                if backup_dir:
                    # Extract original path from backup dir structure
                    rel_backup = os.path.relpath(backup_path, backup_dir)
                    original_path = rel_backup.replace('.original', '')
                else:
                    # Original path is in same directory
                    original_path = backup_path.replace('.original', '')
                
                try:
                    if os.path.exists(original_path):
                        import shutil
                        shutil.copy2(backup_path, original_path)
                        print(f"✓ Restored: {original_path}")
                        restored_count += 1
                except Exception as e:
                    logger.error(f"Failed to restore {original_path}: {e}")
                    failed_count += 1
    
    print(f"\nRollback complete: {restored_count} restored, {failed_count} failed")


def prepare_cli_args(cli_args: List[str]) -> Tuple[List[str], Optional[str]]:
    """
    Normalize CLI arguments so the process command can run with sensible defaults.
    
    Returns the argument list to parse and a string describing whether the command
    was inferred ('implicit' for no args, 'inferred' when the user omitted the
    command but provided other arguments).
    """
    if not cli_args:
        return ['process'], 'implicit'
    if cli_args[0] in ('-h', '--help'):
        return cli_args, None
    if cli_args[0] in {'process', 'rollback', 'stats', 'prep-context'}:
        return cli_args, None
    return ['process'] + cli_args, 'inferred'


def prompt_for_image_folder(default_path: Optional[str]) -> str:
    """
    Prompt the user for an image folder when no CLI argument was provided.
    Requires an interactive terminal; otherwise instructs the user to pass a path.
    """
    if not sys.stdin.isatty():
        raise SystemExit(
            "Image folder path is required when running non-interactively. "
            "Provide it as the first positional argument or set IMAGE_EVAL_DEFAULT_FOLDER."
        )

    while True:
        prompt = "Enter path to the image folder"
        if default_path:
            prompt += f" [{default_path}]"
        prompt += ": "
        try:
            response = input(prompt)
        except EOFError:
            raise SystemExit(
                "Image folder path input was interrupted. "
                "Provide it as the first argument or set IMAGE_EVAL_DEFAULT_FOLDER."
            )

        candidate = response.strip() or (default_path or "")
        if candidate:
            return candidate
        print("Please enter a valid folder path (Ctrl+C to cancel).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and evaluate images with AI.')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process and evaluate images')
    process_parser.add_argument(
        'folder_path',
        type=str,
        nargs='?',
        default=None,
        help='Path to the folder containing images (prompted if omitted; defaults to IMAGE_EVAL_DEFAULT_FOLDER when confirmed)'
    )
    process_parser.add_argument(
        'ollama_host_url',
        type=str,
        nargs='?',
        default=None,
        help=f'Full url of your Ollama API endpoint (default: {DEFAULT_OLLAMA_URL})'
    )
    process_parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKER_COUNT,
        help=f'Number of parallel workers (default: IMAGE_EVAL_WORKERS or {DEFAULT_WORKER_COUNT})'
    )
    process_parser.add_argument('--csv', type=str, default=None, help='Path to save CSV report (default: auto-generated)')
    process_parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                               help=f'Ollama model used for context classification and optional metadata. Default: {DEFAULT_MODEL}')
    process_parser.add_argument('--pyiqa-model', type=str, default=DEFAULT_CLIPIQ_MODEL,
                               help=f'Base PyIQA metric to use for clipiqa_z (default: {DEFAULT_CLIPIQ_MODEL})')
    process_parser.add_argument('--pyiqa-device', type=str, default=None,
                               help='Device for PyIQA (e.g., cuda:0 or cpu). Defaults to CUDA if available.')
    process_parser.add_argument('--pyiqa-score-shift', type=float, default=None,
                               help='Additive adjustment (0-100 scale) applied to PyIQA scores (default: model-specific)')
    process_parser.add_argument('--pyiqa-scale-factor', type=float, default=None,
                               help='Optional multiplier applied to PyIQA raw scores before calibration (auto-detect if omitted)')
    process_parser.add_argument('--pyiqa-max-models', type=int, default=5,
                               help='Maximum PyIQA models kept in GPU memory at once (default: 5, lower = less VRAM)')
    process_parser.add_argument('--pyiqa-batch-size', type=int, default=8,
                               help='Number of images to score per PyIQA batch (default: 8, lower = less VRAM)')
    process_parser.add_argument('--iqa-calibration', type=str, default=str(DEFAULT_IQA_CALIBRATION_PATH),
                               help='Path to iqa_calibration.json containing mu/sigma (default: ./iqa_calibration.json)')
    process_parser.add_argument('--context-host-url', type=str, default=None,
                               help='Override endpoint for context classification (default: same as Ollama host)')
    process_parser.add_argument('--ollama-metadata', action='store_true',
                               help='Use Ollama to generate titles/descriptions/keywords after scoring')
    process_parser.add_argument('--stock-eval', action='store_true',
                               help='Request stock photography suitability assessment (commercial/release scoring)')
    process_parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip images with existing metadata (default: True)')
    process_parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing', help='Process all images, even with existing metadata')
    process_parser.add_argument('--min-score', type=int, default=None, help='Only save results with score >= this value')
    process_parser.add_argument('--file-types', type=str, default=None, help='Comma-separated list of file extensions (e.g., jpg,png,dng)')
    process_parser.add_argument('--dry-run', action='store_true', help='Preview what would be processed without making changes')
    process_parser.add_argument('--backup-dir', type=str, default=None, help='Directory to store backups (default: same directory as originals)')
    process_parser.add_argument('--verify', action='store_true', help='Verify metadata was correctly embedded after writing')
    process_parser.add_argument('--log-file', type=str, default=None, help='Path to log file (default: auto-generated)')
    process_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose debug output')
    process_parser.add_argument('--debug', action='store_true', help='Enable debug logging (includes PyIQA output)')
    process_parser.add_argument('--cache', action='store_true', help='Enable API response caching')
    process_parser.add_argument('--cache-dir', type=str, default=CACHE_DIR, help=f'Cache directory (default: {CACHE_DIR})')
    process_parser.add_argument('--clear-cache', action='store_true', help='Clear cache before processing')
    process_parser.add_argument('--context', type=str, default=None,
                              help='Manual context override (e.g., landscape, portrait_neutral, studio_photography)')
    process_parser.add_argument('--no-context-classification', action='store_true',
                              help='Skip automatic context classification (still uses EXIF-cached context if available, otherwise studio_photography)')
    
    # Prep-context command - classify and embed context only (no scoring)
    prep_parser = subparsers.add_parser('prep-context', help='Classify images and embed context only (no scoring)')
    prep_parser.add_argument('folder_path', type=str, nargs='?', default=None,
                            help='Path to the folder containing images')
    prep_parser.add_argument('ollama_host_url', type=str, nargs='?', default=None,
                            help='URL of the Ollama API endpoint')
    prep_parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                            help=f'Ollama model to use for context classification (default: {DEFAULT_MODEL})')
    prep_parser.add_argument('--csv', type=str, default=None, dest='csv_output',
                            help='Output CSV file for context assignments')
    prep_parser.add_argument('--backup-dir', type=str, default=None,
                            help='Directory for image backups before embedding')
    prep_parser.add_argument('--dry-run', action='store_true',
                            help='Classify images but do not embed metadata')
    prep_parser.add_argument('--workers', type=int, default=DEFAULT_WORKER_COUNT,
                            help=f'Number of parallel workers (default: {DEFAULT_WORKER_COUNT})')
    prep_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    prep_parser.add_argument('--context', type=str, default=None,
                            help='Manual context override for all images (skip classification)')
    prep_parser.add_argument('--retry-contexts', type=str, default=None,
                            help='Comma-separated list of contexts to re-classify (e.g., "macro_food,macro_nature")')
    prep_parser.add_argument('--force', action='store_true',
                            help='Re-classify ALL images, even those with existing non-fallback context')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Restore images from backups')
    rollback_parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    rollback_parser.add_argument('--backup-dir', type=str, default=None, help='Directory where backups are stored')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Print statistics for an existing CSV report')
    stats_parser.add_argument('csv_path', type=str, help='Path to a CSV created by this tool')
    stats_parser.add_argument('--output', '-o', type=str, default=None, 
                             help='Output markdown file (default: prints to console)')
    
    raw_cli_args = sys.argv[1:]
    normalized_args, inferred_command = prepare_cli_args(raw_cli_args)
    args = parser.parse_args(normalized_args)
    
    if hasattr(args, 'debug') and args.debug:
        args.verbose = True

    if inferred_command == 'implicit':
        print("No command supplied. Defaulting to 'process'.")
    elif inferred_command == 'inferred':
        print("Command not specified. Assuming 'process' for provided arguments.")
    
    # Handle rollback command
    if args.command == 'rollback':
        print(f"Rolling back images in: {args.folder_path}")
        if args.backup_dir:
            print(f"Using backup directory: {args.backup_dir}")
        rollback_images(args.folder_path, getattr(args, 'backup_dir', None))
        sys.exit(0)

    if args.command == 'stats':
        if not os.path.exists(args.csv_path):
            logger.error(f"CSV file '{args.csv_path}' not found.")
            sys.exit(1)
        results = load_results_from_csv(args.csv_path)
        stats = calculate_statistics(results)
        
        # Generate markdown output
        markdown_content = generate_statistics_markdown(stats, args.csv_path)
        
        if args.output:
            # Write to specified file
            output_path = args.output
        else:
            # Default: derive output filename from CSV path
            csv_base = os.path.splitext(args.csv_path)[0]
            output_path = f"{csv_base}_stats.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Statistics saved to: {output_path}")
        
        # Also print to console
        print_statistics(stats)
        sys.exit(0)
    
    # Handle prep-context command
    if args.command == 'prep-context':
        # Handle missing folder path
        if not args.folder_path:
            args.folder_path = prompt_for_image_folder(DEFAULT_IMAGE_FOLDER)
        
        # Handle missing Ollama URL (only needed if not using context override)
        if not args.ollama_host_url and not args.context:
            args.ollama_host_url = DEFAULT_OLLAMA_URL
            print(f"Using default Ollama endpoint: {args.ollama_host_url}")
        
        # Validate folder exists
        if not os.path.exists(args.folder_path):
            print(f"{Fore.RED}Error: Folder '{args.folder_path}' does not exist.{Style.RESET_ALL}")
            sys.exit(1)
        
        # Setup logging for prep-context
        setup_logging(verbose=args.verbose)
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Image Context Preparation{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"  Folder:     {args.folder_path}")
        if args.context:
            print(f"  Context:    {args.context} (manual override)")
        else:
            print(f"  Model:      {args.model}")
            print(f"  Ollama:     {args.ollama_host_url}")
        print(f"  Dry-run:    {args.dry_run}")
        print(f"  Workers:    {args.workers}")
        if args.csv_output:
            print(f"  CSV output: {args.csv_output}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        # Parse retry_contexts if provided
        retry_contexts = None
        if args.retry_contexts:
            retry_contexts = [c.strip() for c in args.retry_contexts.split(',')]
            print(f"  Retry:      {', '.join(retry_contexts)}")
        
        results = process_context_only(
            folder_path=args.folder_path,
            ollama_host_url=args.ollama_host_url,
            model=args.model,
            context_override=args.context,
            csv_output=args.csv_output,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
            workers=args.workers,
            force=args.force,
            retry_contexts=retry_contexts
        )
        
        print(f"\n{Fore.GREEN}Completed! Processed {len(results)} images.{Style.RESET_ALL}")
        sys.exit(0)
    
    # Fill in defaults for positional arguments when omitted
    if args.command == 'process':
        folder_defaulted = False
        folder_prompted = False
        host_defaulted = False
        if not args.folder_path:
            folder_prompted = True
            args.folder_path = prompt_for_image_folder(DEFAULT_IMAGE_FOLDER)
            folder_defaulted = DEFAULT_IMAGE_FOLDER and args.folder_path == DEFAULT_IMAGE_FOLDER
        if not args.ollama_host_url:
            args.ollama_host_url = DEFAULT_OLLAMA_URL
            host_defaulted = True
        if not args.context_host_url:
            args.context_host_url = args.ollama_host_url
        
        if folder_prompted:
            if folder_defaulted:
                print(f"Using default image folder: {args.folder_path}")
            else:
                print(f"Image folder selected: {args.folder_path}")
        if host_defaulted:
            print(f"Using default Ollama endpoint: {args.ollama_host_url}")
    
    # Setup logging
    log_file = None
    if args.command == 'process':
        if args.log_file:
            log_file = args.log_file
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"image_evaluator_{timestamp}.log"
        
        setup_logging(log_file, verbose=args.verbose, debug=getattr(args, 'debug', False))
        logger.info(f"Starting image evaluation - log file: {log_file}")
        logger.debug(f"Arguments: {vars(args)}")
    
    # Handle cache
    cache_dir = None
    if args.command == 'process' and args.cache:
        cache_dir = args.cache_dir
        if args.clear_cache and os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache directory: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"API response caching enabled: {cache_dir}")

    pyiqa_manager: Optional[PyiqaManager] = None
    pyiqa_cache_label: Optional[str] = None

    # Validate folder path
    if not os.path.exists(args.folder_path):
        logger.error(f"The folder path '{args.folder_path}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(args.folder_path):
        logger.error(f"The path '{args.folder_path}' is not a directory.")
        sys.exit(1)
    
    # Check recursively for image files
    has_images = False
    for root, dirs, files in os.walk(args.folder_path):
        if any(f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG','NEF','nef','TIF','tif','TIFF','tiff','DNG','dng')) for f in files):
            has_images = True
            break
    
    if not has_images:
        logger.error(f"No image files found in the directory '{args.folder_path}' or its subdirectories.")
        sys.exit(1)

    if not PYIQA_AVAILABLE:
        logger.error("PyIQA backend is required but torch/pyiqa are missing.")
        sys.exit(1)
    IQA_CALIBRATION = load_iqa_calibration(getattr(args, "iqa_calibration", None))
    shift_overrides: Dict[str, float] = {}
    if args.pyiqa_score_shift is not None:
        shift_overrides[args.pyiqa_model.lower()] = args.pyiqa_score_shift
    try:
        pyiqa_manager = PyiqaManager(
            device=args.pyiqa_device,
            scale_factor=args.pyiqa_scale_factor,
            shift_overrides=shift_overrides,
            max_cached_models=args.pyiqa_max_models,
        )
        pyiqa_cache_label = f"pyiqa_profiles_{args.pyiqa_model.lower()}"
    except Exception as e:
        logger.error(f"Failed to initialize PyIQA manager: {e}")
        sys.exit(1)
    
    # Parse file types if specified
    file_types = None
    if args.file_types:
        file_types = [ext.strip() for ext in args.file_types.split(',')]
        print(f"Filtering for file types: {', '.join(file_types)}")
    print(f"\nProcessing images from: {Style.BRIGHT}{args.folder_path}{Style.RESET_ALL}")
    print(f"Ollama model: {args.model}")
    if args.context_host_url and args.context_host_url != args.ollama_host_url:
        print(f"Context classifier endpoint: {args.context_host_url}")
    if args.ollama_metadata:
        print("Ollama metadata generation: enabled")
    if args.stock_eval:
        print("Stock suitability scoring: enabled")
    print(f"PyIQA model (clipiqa_z): {args.pyiqa_model}")
    print(f"PyIQA device: {pyiqa_manager.device if pyiqa_manager else args.pyiqa_device}")
    print(f"PyIQA max cached models: {args.pyiqa_max_models}")
    if args.pyiqa_scale_factor:
        print(f"PyIQA scale factor: {args.pyiqa_scale_factor}")
    if args.pyiqa_score_shift is not None:
        print(f"PyIQA custom score shift: {args.pyiqa_score_shift:+.2f}")
    print(f"IQA calibration file: {getattr(args, 'iqa_calibration', '') or DEFAULT_IQA_CALIBRATION_PATH}")
    print("PyIQA composite models: clipiqa+_vitL14_512, laion_aes, musiq-ava, musiq-paq2piq, maniqa, disagreement penalty")
    print(f"Workers: {args.workers}")
    print(f"Skip existing: {args.skip_existing}")
    if args.min_score:
        print(f"Minimum score filter: {args.min_score}")
    if args.dry_run:
        print(f"\n⚠️  DRY RUN MODE - No changes will be made\n")
    print()
    
    # Create backup directory if specified
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)
        print(f"Backups will be saved to: {args.backup_dir}")
    
    if args.verify:
        print(f"Metadata verification: enabled")
    
    if getattr(args, 'debug', False):
        print("Debug logging: enabled")
    elif args.verbose:
        print("Verbose logging: enabled")
    
    if cache_dir:
        cache_stats = get_cache_stats(cache_dir)
        print(f"Cache: enabled ({cache_stats['entries']} entries, {cache_stats['size_mb']:.2f} MB)")
    
    logger.info(f"Processing configuration: workers={args.workers}, model={args.model}, cache={'enabled' if cache_dir else 'disabled'}")
    
    # Process images
    logger.info(f"Starting processing of images in {args.folder_path}")
    start_time = time.time()
    results = process_images_in_folder(
        args.folder_path,
        args.ollama_host_url,
        args.context_host_url,
        model=args.model,
        file_types=file_types,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        min_score=args.min_score,
        backup_dir=args.backup_dir,
        verify=args.verify,
        cache_dir=cache_dir,
        context_override=args.context,
        skip_context_classification=args.no_context_classification,
        pyiqa_manager=pyiqa_manager,
        pyiqa_cache_label=pyiqa_cache_label,
        cli_args=args,
        use_ollama_metadata=args.ollama_metadata,
        stock_eval=args.stock_eval
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Completed processing {len(results)} images in {elapsed_time:.2f} seconds")
    
    # Calculate statistics
    stats = calculate_statistics(results)
    stats['elapsed_time'] = elapsed_time
    
    # Print statistics
    print_statistics(stats)
    
    # Save to CSV
    if args.csv:
        csv_path = args.csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"image_evaluation_results_{timestamp}.csv"
    
    save_results_to_csv(results, csv_path)
    logger.info(f"Results saved to CSV: {csv_path}")
    print(f"Results saved to: {csv_path}")
    print(f"Log file: {log_file}")
    
    # Final cache stats
    if cache_dir:
        final_cache_stats = get_cache_stats(cache_dir)
        print(f"\nCache Statistics:")
        print(f"  Entries: {final_cache_stats['entries']}")
        print(f"  Size: {final_cache_stats['size_mb']:.2f} MB")
        logger.info(f"Cache stats: {final_cache_stats['entries']} entries, {final_cache_stats['size_mb']:.2f} MB")
