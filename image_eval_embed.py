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
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import numpy as np
import requests
from PIL import Image, ImageStat
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
PYIQA_MAX_LONG_EDGE = 2048
DEFAULT_CLIPIQ_MODEL = "clipiqa+_vitL14_512"

# Stock evaluation thresholds
STOCK_MIN_RESOLUTION_MP = 4.0
STOCK_RECOMMENDED_MP = 12.0
STOCK_MIN_DPI = 300
STOCK_DPI_FIXABLE = 240
STOCK_SHARPNESS_CRITICAL = 30.0
STOCK_SHARPNESS_OPTIMAL = 60.0
STOCK_NOISE_WARN = 45.0
STOCK_NOISE_HIGH = 65.0
STOCK_CLIPPING_THRESHOLD = 12.0  # percent

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
    "stock_product": "stock_product",
    "macro_food": "macro_food",
    "portrait_neutral": "portrait_neutral",
    "portrait_highkey": "portrait_highkey",
    "landscape": "landscape",
    "street_documentary": "street_documentary",
    "sports_action": "sports_action",
    "concert_night": "concert_night",
    "architecture_realestate": "architecture_realestate",
    "fineart_creative": "fineart_creative",
    # Legacy/alias contexts
    "event_lowlight": "concert_night",
    "travel_story": "street_documentary",
    "travel": "landscape",
    "travel_reportage": "street_documentary",
    "product_catalog": "stock_product",
}


def map_context_to_profile(context_label: str) -> str:
    """Map classifier output to one of the 10 defined profile keys."""
    if not context_label:
        return "stock_product"
    normalized = context_label.strip().lower()
    mapped = CONTEXT_PROFILE_MAP.get(normalized, normalized)
    if mapped not in PROFILE_CONFIG:
        return "stock_product"
    return mapped


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weight dictionary so it sums to 1.0."""
    positive_items = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(positive_items.values())
    if total <= 0:
        return {k: 1.0 / len(positive_items) for k in positive_items} if positive_items else {}
    return {k: v / total for k, v in positive_items.items()}


def compute_metric_z_scores(metric_scores: Dict[str, float]) -> Dict[str, float]:
    """Convert calibrated metric scores to z-scores using baseline statistics."""
    z_scores: Dict[str, float] = {}
    for key, value in metric_scores.items():
        stats = PYIQA_BASELINE_STATS.get(key)
        if not stats:
            z_scores[key] = 0.0
            continue
        std = stats.get("std") or 0.0
        if std <= 1e-6:
            z_scores[key] = 0.0
            continue
        mean = stats.get("mean", 0.0)
        z_scores[key] = (value - mean) / std
    return z_scores


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
    profile_cfg = PROFILE_CONFIG.get(profile_key, PROFILE_CONFIG["stock_product"])
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
    profile_cfg = PROFILE_CONFIG.get(profile_key, PROFILE_CONFIG["stock_product"])
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
    color_rules = rules.get("color_cast")
    if color_rules and color_cast_label and color_cast_label != "neutral":
        if color_cast_delta >= color_rules.get("threshold", 0.0):
            penalty += color_rules.get("penalty", 0.0)
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
    diff_z: float,
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
        metric_summary_parts.append(
            f"{metric}:{detail['calibrated']:.1f} (z={z_val:+.2f})"
        )
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
            notes.append(f"soft focus (sharpness {sharpness:.1f})")
        elif sharpness < STOCK_SHARPNESS_OPTIMAL:
            notes.append(f"moderate softness (sharpness {sharpness:.1f})")

    noise_score = technical_metrics.get('noise_score')
    if isinstance(noise_score, (int, float)):
        if noise_score > STOCK_NOISE_HIGH:
            notes.append(f"high noise ({noise_score:.1f})")
        elif noise_score > STOCK_NOISE_WARN:
            notes.append(f"moderate noise ({noise_score:.1f})")

    highlights = technical_metrics.get('histogram_clipping_highlights')
    if isinstance(highlights, (int, float)) and highlights > STOCK_CLIPPING_THRESHOLD:
        notes.append(f"highlight clipping {highlights:.1f}%")

    shadows = technical_metrics.get('histogram_clipping_shadows')
    if isinstance(shadows, (int, float)) and shadows > STOCK_CLIPPING_THRESHOLD:
        notes.append(f"shadow clipping {shadows:.1f}%")

    dpi_x = technical_metrics.get('dpi_x')
    if isinstance(dpi_x, (int, float)) and dpi_x < STOCK_MIN_DPI:
        notes.append(f"low DPI ({dpi_x:.0f})")
        if dpi_x >= STOCK_DPI_FIXABLE:
            fixable.append(f"DPI metadata: {dpi_x:.0f} → {STOCK_MIN_DPI}")

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            np_img = np.asarray(img).astype('float32') / 255.0
    except Exception as exc:
        logger.error(f'Failed to load image for PyIQA preprocessing {image_path}: {exc}')
        return None
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
    return tensor


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

    def score_batch(self, image_paths: List[str]) -> Dict[str, float]:
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
                    image_tensor = load_image_tensor_with_max_edge(image_path, self.max_long_edge)
                    if image_tensor is not None:
                        image_tensor = image_tensor.to(self.device)
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
                    # Explicit tensor cleanup to prevent memory leaks
                    if image_tensor is not None:
                        del image_tensor
                    if score is not None:
                        del score
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
    ):
        self.device = device
        self.scale_factor = scale_factor
        self.shift_overrides = {k.lower(): v for k, v in (shift_overrides or {}).items()}
        self.max_cached_models = max(1, int(max_cached_models))
        self.scorers: Dict[str, PyIqaScorer] = {}
        self.scorer_usage: List[str] = []

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
        scores: Dict[str, Dict[str, float]] = {}
        for metric_key in metric_keys:
            if metric_key == "pyiqa_diff_z":
                continue
            model_name = resolve_metric_model_name(metric_key, args)
            if not model_name:
                continue
            scorer = self.get_scorer(model_name)
            try:
                raw, scaled, calibrated = scorer.score_image(image_path)
            except RuntimeError as exc:
                logger.error(f"PyIQA metric {model_name} failed for {image_path}: {exc}")
                if "out of memory" in str(exc).lower() and torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            scores[metric_key] = {
                "raw": float(raw),
                "scaled": float(scaled),
                "calibrated": float(calibrated),
                "model_name": model_name,
            }
        return scores


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
# Technical analysis constants
COLOR_CAST_THRESHOLD = 15.0  # Mean difference threshold for color cast detection
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


@contextmanager
def open_image_for_analysis(image_path: str):
    """Context manager to open image files (including RAW) for analysis.
    
    Yields a PIL Image object. For RAW files, uses rawpy to decode.
    """
    ext = Path(image_path).suffix.lower()
    if ext in RAW_EXTENSIONS:
        if not RAWPY_AVAILABLE:
            _warn_rawpy_missing()
            raise RuntimeError("rawpy is required to process RAW files")
        raw = rawpy.imread(image_path)  # type: ignore
        try:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                gamma=(1.0, 1.0)
            )
        finally:
            raw.close()

        img = Image.fromarray(rgb.astype('uint8'))
        try:
            yield img
        finally:
            img.close()
    else:
        with Image.open(image_path) as img:
            yield img


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


# Lightweight classification prompt for 10 contexts
IMAGE_CONTEXT_CLASSIFIER_PROMPT = """You are analyzing this photograph to classify it into ONE category.

Look at the image and determine which category it belongs to:

1. landscape - outdoor nature scenes, mountains, forests, seascapes, sunsets, natural vistas
2. portrait_neutral - people photos with standard lighting, headshots, portraits
3. portrait_highkey - bright, overexposed people photos, airy portraits
4. macro_food - extreme close-ups of food or small objects, detailed macro shots
5. street_documentary - candid street photography, photojournalism, documentary style
6. sports_action - fast motion, sports, wildlife in motion, action photography
7. concert_night - dark scenes, concerts, night cityscapes, low-light photography
8. architecture_realestate - buildings, interiors, rooms, architectural photography
9. stock_product - clean product shots on white/neutral backgrounds, catalog photography
10. fineart_creative - abstract, experimental, artistic, intentionally unconventional

Respond with ONLY the category name from the list above. Examples: "landscape" or "portrait_neutral" or "street_documentary"

Your response:"""


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
    """Classify image context with retry logic and confidence tracking."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = _classify_image_context_once(image_path, ollama_host_url, model)
            result.retries = attempt
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
            return ClassificationResult('stock_product', 'low', 'error', str(e), attempt)
    
    logger.warning(f"Classification exhausted retries for {image_path}: {last_error}")
    return ClassificationResult('stock_product', 'low', 'retry_exhausted', str(last_error), max_retries)


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
        "images": [encoded_image],
        "prompt": IMAGE_CONTEXT_CLASSIFIER_PROMPT,
        "options": {
            "temperature": 0.3,
            "num_predict": 100,
            "top_p": 0.9,
            "repeat_penalty": 1.1
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
            return ClassificationResult('stock_product', 'low', 'empty_response', '', 0)
    
    # Handle numbered responses (e.g., "1", "1.", "5. landscape")
    number_to_context = {
        '1': 'landscape',
        '2': 'portrait_neutral',
        '3': 'portrait_highkey',
        '4': 'macro_food',
        '5': 'street_documentary',
        '6': 'sports_action',
        '7': 'concert_night',
        '8': 'architecture_realestate',
        '9': 'stock_product',
        '10': 'fineart_creative'
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
    positive_markers = ["include", "includes", "fits", "fit", "belongs to", "classified as", "this is", "is a", "matches", "best fits"]
    negative_markers = ["don't fit", "doesn't fit", "does not fit", "not ", "no ", "without"]
    candidate_scores = {}
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
            if known_context in lowered:
                score = 2 if sentiment == 1 else (-2 if sentiment == -1 else 1)
                candidate_scores[known_context] = max(score, candidate_scores.get(known_context, -10))
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
                  f"Defaulting to 'stock_product' (most restrictive). Consider manual context override.")
    return ClassificationResult('stock_product', 'low', 'fallback', raw_response, 0)


def analyze_image_technical(image_path: str, iso_value: Optional[int] = None, context: str = 'stock_product') -> Dict:
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

            if CV2_AVAILABLE and cv2 is not None:
                laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()

                # Scale-normalize: adjust for image resolution
                # Larger images naturally have higher variance, normalize to 1MP baseline
                h, w = gray_array.shape
                mp = (h * w) / 1_000_000
                scale_factor = np.sqrt(max(mp, 0.1))  # Avoid division by zero
                metrics['sharpness'] = float(laplacian_var / scale_factor)

                # --- Camera/ISO-agnostic noise estimation ---
                # 1) Standardize size
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

                # 2) Normalize to [0,1]
                gray_f = gray_small.astype(np.float32)
                max_val = gray_f.max()
                if max_val > 1.5:  # assume 8-bit-equivalent
                    gray_f /= 255.0
                elif max_val > 0:  # already 0–1
                    gray_f /= max_val  # safety normalization on odd inputs

                # 3) High-frequency residual
                blurred = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1.0, sigmaY=1.0)
                residual = gray_f - blurred

                # 4) Mask to flat regions (avoid edges and extremes)
                gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
                grad_mag = np.sqrt(gx**2 + gy**2)

                edge_thresh = np.percentile(grad_mag, 75.0)
                flat_mask = grad_mag < edge_thresh

                p_low, p_high = np.percentile(gray_f, [5.0, 95.0])
                luminance_mask = (gray_f > p_low) & (gray_f < p_high)

                final_mask = flat_mask & luminance_mask
                flat_residuals = residual[final_mask]

                # Fallback: need at least 1% of pixels for reliable noise estimate
                min_pixels = int(0.01 * residual.size)
                if flat_residuals.size < min_pixels:
                    # If too few flat pixels, use full image but log it
                    flat_residuals = residual.flatten()
                    logger.debug(f"Noise estimation using full image for {image_path} (insufficient flat regions)")

                # Robust sigma via MAD
                med = float(np.median(flat_residuals))  # type: ignore
                mad = float(np.median(np.abs(flat_residuals - med)))  # type: ignore
                if mad < 1e-6:
                    sigma_noise = 0.0
                else:
                    sigma_noise = 1.4826 * mad  # approx std for Gaussian

                # 5) Normalize by effective dynamic range
                p1, p99 = np.percentile(gray_f, [1.0, 99.0])
                dynamic_range = max(float(p99 - p1), 1e-6)
                relative_noise = sigma_noise / dynamic_range

                # 6) Map to 0–100 severity score
                REL_NOISE_MIN = 0.002
                REL_NOISE_MAX = 0.04
                rn_clipped = min(max(relative_noise, REL_NOISE_MIN), REL_NOISE_MAX)
                noise_score = (rn_clipped - REL_NOISE_MIN) / (REL_NOISE_MAX - REL_NOISE_MIN) * 100.0

                metrics['noise_sigma'] = float(sigma_noise)
                metrics['noise_score'] = float(noise_score)
                metrics['noise'] = float(noise_score)  # backward-compatible alias
            else:
                logger.debug("OpenCV not available, skipping Laplacian/denoise metrics for %s", image_path)
                # Fallback sharpness proxy using grayscale standard deviation
                gray_stat = ImageStat.Stat(img_gray)
                metrics['sharpness'] = float(sum(gray_stat.stddev) / len(gray_stat.stddev))
    
    except ImageResolutionTooSmallError:
        raise
    except Exception as e:
        logger.debug(f"Could not analyze technical metrics for {image_path}: {e}")
        metrics['status'] = 'error'
    
    return metrics


def assess_technical_metrics(technical_metrics: Dict, context: str = "stock_product") -> List[str]:
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


def compute_post_process_potential(technical_metrics: Dict, context: str = "stock_product") -> int:
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
        image_context = context_override if context_override in PROFILE_CONFIG else 'stock_product'
        logger.info(f"Using manual context override: {image_context}")
    else:
        cached_context = read_cached_context(image_path)
        if cached_context:
            image_context = cached_context
            logger.info(f"Using cached context from EXIF: {image_context}")
        elif skip_context_classification:
            image_context = 'stock_product'
            logger.debug(f"Context classification disabled, using default: {image_context}")
        else:
            try:
                image_context = classify_image_context(image_path, ollama_host_url, model)
            except Exception as e:
                logger.warning(f"Context classification failed for {image_path}: {e}, using default")
                image_context = 'stock_product'

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
    log_level = logging.DEBUG if (verbose or debug) else logging.INFO
    
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
    force: bool = False
) -> List[Tuple[str, str, str]]:
    """Classify images and embed context only (no scoring).
    
    By default, processes images that:
    - Have no context embedded, OR
    - Have 'stock_product' context (the fallback)
    
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
                elif cached and cached != 'stock_product':
                    # Has real classification - skip and record
                    results.append((image_path, cached, 'cached'))
                    print(f"{Fore.YELLOW}[Cached]{Style.RESET_ALL} {os.path.basename(image_path)}: {cached}")
                else:
                    # No context OR stock_product fallback - needs (re)classification
                    if cached == 'stock_product':
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
            context = context_override if context_override in PROFILE_CONFIG else 'stock_product'
            confidence = 'override'
        else:
            try:
                result = classify_image_context_detailed(image_path, ollama_host_url, model)
                context = result.context
                confidence = result.confidence
            except Exception as e:
                logger.warning(f"Classification failed for {image_path}: {e}")
                context = 'stock_product'
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
                    results.append((path, 'stock_product', 'error'))
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

            profile_key = map_context_to_profile(image_context)
            profile_cfg = PROFILE_CONFIG.get(profile_key, PROFILE_CONFIG["stock_product"])
            metric_keys = list(profile_cfg.get("model_weights", {}).keys())
            try:
                metric_details = pyiqa_manager.score_metrics(image_path, metric_keys, args)
            except Exception as exc:
                logger.error(f"PyIQA scoring failed for {image_path}: {exc}")
                results.append((image_path, None))
                pbar.update(1)
                continue

            if not metric_details:
                logger.error(f"No PyIQA metrics computed for {image_path}")
                results.append((image_path, None))
                pbar.update(1)
                continue

            calibrated_scores = {k: v["calibrated"] for k, v in metric_details.items()}
            z_scores = compute_metric_z_scores(calibrated_scores)
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
                diff_z=diff_z,
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

            try:
                embed_metadata(image_path, metadata, backup_dir, verify)
            except Exception as exc:
                logger.error(f"Failed to embed metadata for {image_path}: {exc}")
                results.append((image_path, None))
                pbar.update(1)
                continue

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
        fieldnames = [
            'file_path', 'overall_score', 'technical_score', 'composition_score',
            'lighting_score', 'creativity_score', 'title', 'description',
            'keywords', 'status', 'context', 'context_profile', 'sharpness', 'brightness', 'contrast',
            'histogram_clipping_highlights', 'histogram_clipping_shadows',
            'color_cast', 'noise_sigma', 'noise_score', 'technical_warnings', 'post_process_potential',
            'resolution_mp', 'dpi', 'stock_notes', 'stock_fixable',
            'stock_overall_score', 'stock_recommendation', 'stock_primary_category',
            'stock_commercial_viability', 'stock_technical_quality', 'stock_composition_clarity',
            'stock_keyword_potential', 'stock_release_concerns', 'stock_rejection_risks',
            'stock_llm_notes', 'stock_llm_issues'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for file_path, metadata in results:
            if metadata:
                technical_metrics = metadata.get('technical_metrics', {})
                warnings = metadata.get('technical_warnings', [])
                warnings_str = '; '.join(warnings) if warnings else ''
                writer.writerow({
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
                    'status': 'success'
                })
            else:
                writer.writerow({
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
                })


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


def calculate_statistics(results: List[Tuple[str, Optional[Dict]]]) -> Dict:
    """Calculate statistics from processing results."""
    scores = []
    tech_scores: List[int] = []
    raw_count = 0
    pil_count = 0
    potentials = []
    
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
    
    warning_images = sum(1 for _, md in results if md and md.get('technical_warnings'))

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
    }


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
    
    print(f"\n{'='*60}")
    print(f"Note: JPEG/PNG use PIL, RAW/TIFF use exiftool for metadata embedding")
    print(f"{'='*60}\n")


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
    process_parser.add_argument('--pyiqa-max-models', type=int, default=1,
                               help='Maximum PyIQA models kept in GPU memory at once (default: 1, lower = less VRAM)')
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
                              help='Manual context override (e.g., landscape, portrait_neutral, stock_product)')
    process_parser.add_argument('--no-context-classification', action='store_true',
                              help='Skip automatic context classification, use stock_product for all')
    
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
    prep_parser.add_argument('--force', action='store_true',
                            help='Re-classify ALL images, even those with existing non-fallback context')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Restore images from backups')
    rollback_parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    rollback_parser.add_argument('--backup-dir', type=str, default=None, help='Directory where backups are stored')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Print statistics for an existing CSV report')
    stats_parser.add_argument('csv_path', type=str, help='Path to a CSV created by this tool')
    
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
        
        results = process_context_only(
            folder_path=args.folder_path,
            ollama_host_url=args.ollama_host_url,
            model=args.model,
            context_override=args.context,
            csv_output=args.csv_output,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run,
            workers=args.workers,
            force=args.force
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
