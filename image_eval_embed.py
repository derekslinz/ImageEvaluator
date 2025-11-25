from __future__ import annotations

import argparse
import base64
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import cv2
import piexif
import piexif.helper
import requests
from PIL import Image, ImageStat
from colorama import Fore, Style
from pydantic import BaseModel, ConfigDict, field_validator
from tqdm import tqdm

try:
    import rawpy
except ImportError:
    rawpy = None  # type: ignore

RAWPY_AVAILABLE = rawpy is not None
RAWPY_IMPORT_WARNINGED = False
RAW_EXTENSIONS = {'.nef', '.cr2', '.cr3', '.arw', '.rw2', '.raf', '.orf', '.dng'}

try:
    import torch
    import pyiqa  # type: ignore
    PYIQA_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    pyiqa = None  # type: ignore
    PYIQA_AVAILABLE = False


def list_pyiqa_metrics() -> List[str]:
    if not PYIQA_AVAILABLE or pyiqa is None:
        return []
    try:
        return sorted(getattr(pyiqa, "AVAILABLE_METRICS", []))
    except Exception:
        return []


def get_default_pyiqa_shift(model_name: str) -> float:
    defaults = {
        'clipiqa+_vitl14_512': 14.0,
        'maniqa': 14.0,
        'maniqa-kadid': 14.0,
        'maniqa-pipal': 14.0,
    }
    return defaults.get(model_name.lower(), 0.0)

# Increase PIL decompression bomb limit for large legitimate images
Image.MAX_IMAGE_PIXELS = None  # Remove limit entirely (or set to a higher value like 500000000)


PYIQA_MAX_LONG_EDGE = 2048


# Configure basic logging (will be updated based on verbose flag)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)





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
            autocast_ctx = lambda: torch.amp.autocast('cuda')
        else:
            autocast_ctx = nullcontext
        assert torch is not None, "PyTorch is required for PyIQA scoring but was not imported."
        with torch.no_grad():
            for image_path in image_paths:
                try:
                    image_tensor = load_image_tensor_with_max_edge(image_path, self.max_long_edge) if torch is not None else None
                    if image_tensor is not None:
                        image_tensor = image_tensor.to(self.device)
                        with autocast_ctx():
                            score = self.metric(image_tensor)
                    else:
                        with autocast_ctx():
                            score = self.metric(image_path)
                    if hasattr(score, 'item'):
                        value = float(score.item())
                    else:
                        value = float(score)
                    results[image_path] = value
                except Exception as exc:
                    logger.error(f"PyIQA scoring failed for {image_path}: {exc}")
        return results

    def convert_score(self, raw_score: float) -> float:
        if self.scale_factor is not None:
            return raw_score * self.scale_factor
        if 0.0 <= raw_score <= 1.0:
            return raw_score * 100.0
        if 0.0 <= raw_score <= 10.0:
            return raw_score * 10.0
        return raw_score


def _is_raw_image(image_path: str) -> bool:
    return Path(image_path).suffix.lower() in RAW_EXTENSIONS

def _warn_rawpy_missing():
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
DEFAULT_ENSEMBLE_PASSES = 1  # Number of evaluation passes for ensemble scoring

# Technical analysis constants
COLOR_CAST_THRESHOLD = 15.0  # Mean difference threshold for color cast detection
HISTOGRAM_HIGHLIGHT_RANGE = (250, 256)  # Histogram bins considered as highlights
HISTOGRAM_SHADOW_RANGE = (0, 6)  # Histogram bins considered as shadows

DEFAULT_PROMPT = """You are a highly critical professional photography judge with expertise in technical and artistic evaluation. Your role is to evaluate photographs purely on their photographic merits with strict standards, regardless of subject matter.

If you are required to refuse evaluation for any reason, briefly state that you cannot evaluate the image instead of providing scores. For all other images, do not let the subject matter influence your scoring; focus only on photographic execution.

DETAILED EVALUATION CRITERIA

All scores below are integers from 1-100. Use the full range.

1. TECHNICAL QUALITY (30% weight)
Evaluate:
- Exposure: Tonal distribution appears appropriate; no obviously blown highlights or crushed shadows unless clearly stylistic.
- Focus & Sharpness: Critical focus on the subject, appropriate depth of field, no unwanted motion blur. If you cannot clearly see a creative intention behind blur, treat it as a flaw.
- Noise/Grain: Acceptable noise levels, especially in shadows. Grain is acceptable only if it clearly supports the style.
- Color: Pleasing color rendition, plausible white balance, no strong color cast unless clearly intentional and effective.
- Lens Quality: Minimal chromatic aberration, distortion, and vignetting, or creatively justified.
- Processing: Natural tone curve and contrast; no obvious over-sharpening, halos, or artifacts.

2. COMPOSITION (30% weight)
Evaluate:
- Visual Balance: Effective subject placement (rule of thirds, golden ratio, or purposeful centering).
- Leading Lines: Lines and shapes guide the eye through the frame.
- Framing: Clean edges, no distracting cut-offs or edge-clutter, appropriate cropping.
- Depth & Layers: Foreground/midground/background create depth where appropriate.
- Negative Space: Use of empty space that emphasizes the subject.
- Perspective: Compelling angle and viewpoint; horizon level when needed or deliberately tilted.

3. LIGHTING (20% weight)
Evaluate:
- Quality: Soft or hard light used appropriately for the subject; pleasant shadow quality.
- Direction: Front/side/back lighting chosen deliberately to shape the subject.
- Dynamic Range: Sufficient detail retained in important highlights and shadows.
- Color Temperature: Warmth/coolness supports the mood.
- Contrast: Good tonal separation between key elements.

4. CREATIVITY & IMPACT (20% weight)
Evaluate:
- Unique Perspective: Fresh or interesting viewpoint, not a cliche angle.
- Emotional Impact: Conveys feeling or story, even subtly.
- Artistic Vision: Clear intent and cohesive style.
- Originality: Stands out within its genre or subject type.
- Moment: For action or candid scenes, quality of the captured moment.

IMPORTANT - SCORING FRAMEWORK (USE 1-100)

Calibrate scores based on absolute quality, not relative to other images in the same batch:
- 95-100: Near-perfect, museum/gallery-level, iconic imagery (extremely rare: ~1 in 10,000).
- 90-94: Award-winning excellence, top competition material (top ~1%).
- 85-89: Outstanding professional work, competition finalist quality (top ~5%).
- 80-84: Excellent professional quality, strong technique and vision (top ~10%).
- 75-79: Very good work, advanced skill level (top ~20%).
- 70-74: Solid, competent professional/serious amateur work (top ~30%).
- 65-69: Above average, decent technical execution.
- 60-64: Average competent photography.
- 55-59: Below average, noticeable issues.
- 50-54: Mediocre, multiple problems.
- 40-49: Poor quality, significant technical failures.
- 30-39: Very poor, severe problems.
- 20-29: Barely usable.
- 1-19: Fundamentally failed.

CALIBRATION GUIDE

Use these as anchors:
- Would win a major international photography competition: typically 88-95.
- Could be published in National Geographic / top-tier magazine: typically 85-92.
- Shows professional skill but not groundbreaking: typically 75-84.
- Technically sound but uninspired: typically 65-74.
- Typical amateur snapshot without care: typically 50-64.
- Major technical problems: usually below 50.

Assume most random user-submitted photos fall in the 45-65 overall range. Scores above 80 are uncommon; above 90 are extremely rare. Do not cluster everything in 75-85.

High-end anchor: Award 90+ when the image is technically flawless, beautifully lit, and über compelling; explicitly state which attributes (sharpness, lighting control, composition) justify such a grade rather than letting minor clipping concerns dominate the score, and treat small warnings (<10% clipping or sharpness >40) as advisory unless they severely impair quality.

OVERALL SCORE

Compute the overall score as a weighted combination:
- technical_score: 30%
- composition_score: 30%
- lighting_score: 20%
- creativity_score: 20%

Use the formula:
overall_score = round(0.3*technical_score + 0.3*composition_score + 0.2*lighting_score + 0.2*creativity_score)

OUTPUT FORMAT (STRICT)

Return only a single valid JSON object, with exactly these fields and no others. Do not include any explanation, commentary, or markdown code fences.

Field requirements:
- technical_score: integer 1-100
- composition_score: integer 1-100
- lighting_score: integer 1-100
- creativity_score: integer 1-100
- overall_score: integer 1-100 (using the weighted formula above)
- title: concise descriptive title, maximum 60 characters
- description: image description, maximum 200 characters, objective and free of speculation about unseen context
- keywords: comma-separated list of up to 12 relevant English keywords or short phrases, all lowercase, no hashtags, no quotes, no duplicates

Example format (structure only; values are illustrative):
{"technical_score": 55, "composition_score": 62, "lighting_score": 58, "creativity_score": 60, "overall_score": 58, "title": "sunset over mountains", "description": "Decent sunset composition with strong colors but slightly overexposed highlights and soft detail in the foreground.", "keywords": "sunset, mountains, landscape, golden hour, clouds, peaks, nature, scenic, dramatic sky, outdoor, wilderness"}"""


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


def load_prompt_from_file(prompt_file: str) -> str:
    """Load custom prompt from file."""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load prompt from {prompt_file}: {e}")
        sys.exit(1)


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
    """Extract and validate score is between 1-100. Accepts int, str, or other types."""
    # Handle integer input directly
    if isinstance(score_input, int):
        if 1 <= score_input <= 100:
            return score_input
        else:
            logger.warning(f"Integer score out of range: {score_input}")
            return None
    
    # Handle string or other input
    try:
        # Try direct conversion first
        score = int(score_input)
        if 1 <= score <= 100:
            return score
    except (ValueError, TypeError):
        pass
    
    # Try to extract first two-digit or single-digit number from string
    # Look for standalone numbers (with word boundaries)
    matches = re.findall(r'\b(\d{1,3})\b', str(score_input))
    for match in matches:
        score = int(match)
        if 1 <= score <= 100:
            return score
    
    # Fallback: try to find any number
    match = re.search(r'\d+', str(score_input))
    if match:
        score = int(match.group())
        if 1 <= score <= 100:
            return score
    
    # Truncate long error messages
    score_preview = str(score_input)[:100] + '...' if len(str(score_input)) > 100 else str(score_input)
    logger.warning(f"Invalid score (no valid number 1-100 found): {score_preview}")
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


# Context-aware technical evaluation profiles
# Each profile defines numeric thresholds for warnings and post-processing potential scoring
TECH_PROFILES = {
    # 1. Stock / product / catalog: strict, neutral, technically clean
    "stock_product": {
        "name": "Stock/Product Photography",
        "post_base": 70,
        "sharpness_crit": 30.0,
        "sharpness_soft": 60.0,
        "sharpness_post_heavy": 35.0,
        "sharpness_post_soft": 60.0,
        "clip_warn": 5.0,
        "clip_penalty_mid": 6.0,
        "clip_penalty_high": 12.0,
        "clip_bonus_max": 0.5,
        "color_cast_threshold": 15.0,
        "color_cast_penalty": 8,
        "noise_warn": 30.0,
        "noise_high": 60.0,
        "noise_penalty_mid": 8,
        "noise_penalty_high": 18,
        "brightness_range": (200, 245),
    },
    
    # 2. Macro / food
    "macro_food": {
        "name": "Macro/Food Photography",
        "post_base": 70,
        "sharpness_crit": 35.0,
        "sharpness_soft": 65.0,
        "sharpness_post_heavy": 40.0,
        "sharpness_post_soft": 65.0,
        "clip_warn": 3.0,
        "clip_penalty_mid": 5.0,
        "clip_penalty_high": 10.0,
        "clip_bonus_max": 0.5,
        "color_cast_threshold": 18.0,
        "color_cast_penalty": 6,
        "noise_warn": 30.0,
        "noise_high": 55.0,
        "noise_penalty_mid": 8,
        "noise_penalty_high": 16,
        "brightness_range": (180, 240),
    },
    
    # 3. Portrait, neutral/standard
    "portrait_neutral": {
        "name": "Portrait (Neutral/Studio)",
        "post_base": 70,
        "sharpness_crit": 25.0,
        "sharpness_soft": 55.0,
        "sharpness_post_heavy": 30.0,
        "sharpness_post_soft": 55.0,
        "clip_warn": 6.0,
        "clip_penalty_mid": 10.0,
        "clip_penalty_high": 18.0,
        "clip_bonus_max": 1.0,
        "color_cast_threshold": 12.0,
        "color_cast_penalty": 10,
        "noise_warn": 35.0,
        "noise_high": 65.0,
        "noise_penalty_mid": 8,
        "noise_penalty_high": 16,
        "brightness_range": (100, 200),
    },
    
    # 4. Portrait, high-key
    "portrait_highkey": {
        "name": "High-Key Portrait",
        "post_base": 70,
        "sharpness_crit": 20.0,
        "sharpness_soft": 50.0,
        "sharpness_post_heavy": 25.0,
        "sharpness_post_soft": 50.0,
        "clip_warn": 12.0,
        "clip_penalty_mid": 25.0,
        "clip_penalty_high": 40.0,
        "clip_bonus_max": 2.0,
        "color_cast_threshold": 18.0,
        "color_cast_penalty": 4,
        "noise_warn": 40.0,
        "noise_high": 70.0,
        "noise_penalty_mid": 6,
        "noise_penalty_high": 12,
        "brightness_range": (180, 255),
    },
    
    # 5. Landscape / nature
    "landscape": {
        "name": "Landscape/Nature",
        "post_base": 70,
        "sharpness_crit": 30.0,
        "sharpness_soft": 60.0,
        "sharpness_post_heavy": 35.0,
        "sharpness_post_soft": 60.0,
        "clip_warn": 4.0,
        "clip_penalty_mid": 8.0,
        "clip_penalty_high": 15.0,
        "clip_bonus_max": 0.5,
        "color_cast_threshold": 18.0,
        "color_cast_penalty": 5,
        "noise_warn": 30.0,
        "noise_high": 60.0,
        "noise_penalty_mid": 8,
        "noise_penalty_high": 16,
        "brightness_range": (100, 200),
    },
    
    # 6. Street / documentary
    "street_documentary": {
        "name": "Street/Documentary",
        "post_base": 70,
        "sharpness_crit": 20.0,
        "sharpness_soft": 45.0,
        "sharpness_post_heavy": 25.0,
        "sharpness_post_soft": 45.0,
        "clip_warn": 10.0,
        "clip_penalty_mid": 25.0,
        "clip_penalty_high": 45.0,
        "clip_bonus_max": 2.0,
        "color_cast_threshold": 22.0,
        "color_cast_penalty": 4,
        "noise_warn": 40.0,
        "noise_high": 75.0,
        "noise_penalty_mid": 4,
        "noise_penalty_high": 10,
        "brightness_range": (50, 200),
    },
    
    # 7. Sports / action / wildlife
    "sports_action": {
        "name": "Sports/Action/Wildlife",
        "post_base": 70,
        "sharpness_crit": 25.0,
        "sharpness_soft": 55.0,
        "sharpness_post_heavy": 30.0,
        "sharpness_post_soft": 55.0,
        "clip_warn": 8.0,
        "clip_penalty_mid": 20.0,
        "clip_penalty_high": 35.0,
        "clip_bonus_max": 1.0,
        "color_cast_threshold": 20.0,
        "color_cast_penalty": 6,
        "noise_warn": 40.0,
        "noise_high": 70.0,
        "noise_penalty_mid": 6,
        "noise_penalty_high": 12,
        "brightness_range": (80, 220),
    },
    
    # 8. Concert / night / city at night
    "concert_night": {
        "name": "Concert/Night/Low-Light",
        "post_base": 70,
        "sharpness_crit": 15.0,
        "sharpness_soft": 40.0,
        "sharpness_post_heavy": 20.0,
        "sharpness_post_soft": 40.0,
        "clip_warn": 20.0,
        "clip_penalty_mid": 40.0,
        "clip_penalty_high": 70.0,
        "clip_bonus_max": 3.0,
        "color_cast_threshold": 30.0,
        "color_cast_penalty": 0,
        "noise_warn": 50.0,
        "noise_high": 80.0,
        "noise_penalty_mid": 4,
        "noise_penalty_high": 10,
        "brightness_range": (20, 120),
    },
    
    # 9. Architecture / real estate
    "architecture_realestate": {
        "name": "Architecture/Real Estate",
        "post_base": 70,
        "sharpness_crit": 30.0,
        "sharpness_soft": 65.0,
        "sharpness_post_heavy": 35.0,
        "sharpness_post_soft": 65.0,
        "clip_warn": 4.0,
        "clip_penalty_mid": 8.0,
        "clip_penalty_high": 15.0,
        "clip_bonus_max": 0.5,
        "color_cast_threshold": 15.0,
        "color_cast_penalty": 8,
        "noise_warn": 30.0,
        "noise_high": 55.0,
        "noise_penalty_mid": 8,
        "noise_penalty_high": 18,
        "brightness_range": (120, 220),
    },
    
    # 10. Fine-art / experimental / ICM
    "fineart_creative": {
        "name": "Fine Art/Creative/Experimental",
        "post_base": 70,
        "sharpness_crit": 10.0,
        "sharpness_soft": 25.0,
        "sharpness_post_heavy": 15.0,
        "sharpness_post_soft": 25.0,
        "clip_warn": 25.0,
        "clip_penalty_mid": 60.0,
        "clip_penalty_high": 90.0,
        "clip_bonus_max": 5.0,
        "color_cast_threshold": 25.0,
        "color_cast_penalty": 0,
        "noise_warn": 60.0,
        "noise_high": 90.0,
        "noise_penalty_mid": 2,
        "noise_penalty_high": 6,
        "brightness_range": (0, 255),
    },
}

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


def classify_image_context(image_path: str, ollama_host_url: str, model: str) -> str:
    """Quickly classify image context using vision model."""
    try:
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
                return 'stock_product'
        
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
            return matched_context
        
        # Explicit "category X" reference takes priority
        import re
        category_matches = re.findall(r'category\s*(\d+)', context_label)
        if category_matches:
            last_match = category_matches[-1]
            if last_match in number_to_context:
                matched_context = number_to_context[last_match]
                logger.info(f"Context classification: {matched_context} (from 'category {last_match}') for {image_path}")
                return matched_context

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
            for known_context in TECH_PROFILES.keys():
                if known_context in lowered:
                    score = 2 if sentiment == 1 else (-2 if sentiment == -1 else 1)
                    candidate_scores[known_context] = max(score, candidate_scores.get(known_context, -10))
        if candidate_scores:
            best_context, best_score = max(candidate_scores.items(), key=lambda kv: kv[1])
            if best_score > 0:
                logger.info(f"Context classification: {best_context} (sentence-weighted) for {image_path}")
                return best_context

        # Validate against known contexts (exact match)
        if context_label in TECH_PROFILES:
            logger.info(f"Context classification: {context_label} (exact match) for {image_path}")
            return context_label
        
        # Try to extract valid context from response (partial match)
        for known_context in TECH_PROFILES.keys():
            if known_context in context_label:
                logger.info(f"Context classification: {known_context} (extracted from '{context_label}') for {image_path}")
                return known_context
        
        # Log the failure with the actual response
        logger.warning(f"Context classification failed: unknown response '{raw_response}' (normalized: '{context_label}') for {image_path}. "
                      f"Defaulting to 'stock_product' (most restrictive). Consider manual context override.")
        return 'stock_product'
        
    except Exception as e:
        logger.warning(f"Context classification failed for {image_path}: {e}, defaulting to 'stock_product'")
        return 'stock_product'


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
    profile = TECH_PROFILES.get(context, TECH_PROFILES['stock_product'])
    metrics['context_profile'] = profile['name']
    
    try:
        # Read image with PIL (or rawpy) for stats
        with open_image_for_analysis(image_path) as img:
            img_rgb = img if img.mode == 'RGB' else img.convert('RGB')
            
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
            laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()
            
            # Scale-normalize: adjust for image resolution
            # Larger images naturally have higher variance, normalize to 1MP baseline
            h, w = gray_array.shape
            mp = (h * w) / 1_000_000
            scale_factor = np.sqrt(max(mp, 0.1))  # Avoid division by zero
            metrics['sharpness'] = float(laplacian_var / scale_factor)

            # --- Camera/ISO-agnostic noise estimation ---
            # 1) Standardize size
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
    
    except Exception as e:
        logger.debug(f"Could not analyze technical metrics for {image_path}: {e}")
        metrics['status'] = 'error'
    
    return metrics


def get_profile(context: str) -> Dict:
    """Retrieve technical profile for a given context, with fallback."""
    return TECH_PROFILES.get(context, TECH_PROFILES["stock_product"])


def assess_technical_metrics(technical_metrics: Dict, context: str = "stock_product") -> List[str]:
    """Generate human-readable warnings based on measured metrics and context."""
    profile = get_profile(context)
    warnings: List[str] = []

    # Sharpness
    sharpness = technical_metrics.get('sharpness')
    if sharpness is not None:
        if sharpness < profile["sharpness_crit"]:
            warnings.append(f"Sharpness critically low ({sharpness:.1f})")
        elif sharpness < profile["sharpness_soft"]:
            warnings.append(f"Lower sharpness ({sharpness:.1f}) may impact detail")

    # Highlight and shadow clipping
    highlights = float(technical_metrics.get('histogram_clipping_highlights', 0.0))
    shadows = float(technical_metrics.get('histogram_clipping_shadows', 0.0))

    if highlights > profile["clip_warn"]:
        warnings.append(f"Highlight clipping {highlights:.1f}% reduces tonal range")

    if shadows > profile["clip_warn"]:
        warnings.append(f"Shadow clipping {shadows:.1f}% removes shadow detail")

    # Color cast (respect profile penalties)
    color_cast = technical_metrics.get('color_cast', 'neutral')
    if color_cast != 'neutral' and profile["color_cast_penalty"] > 0:
        warnings.append(f"Color cast detected: {color_cast}")

    # Noise (0–100 severity)
    noise_score = float(technical_metrics.get('noise_score', 0.0))
    if noise_score > profile["noise_high"]:
        warnings.append(f"High noise level (score {noise_score:.1f}/100)")
    elif noise_score > profile["noise_warn"]:
        warnings.append(f"Elevated noise (score {noise_score:.1f}/100)")

    return warnings


def compute_post_process_potential(technical_metrics: Dict, context: str = "stock_product") -> int:
    """Estimate how much post-processing can improve this image (0–100)."""
    profile = get_profile(context)
    base_score = float(profile["post_base"])

    # Sharpness contribution
    sharpness = technical_metrics.get('sharpness')
    if sharpness is not None:
        if sharpness < profile["sharpness_post_heavy"]:
            base_score -= 25
        elif sharpness < profile["sharpness_post_soft"]:
            base_score -= 10
        else:
            base_score += 5

    # Clipping contribution
    highlights = float(technical_metrics.get('histogram_clipping_highlights', 0.0))
    shadows = float(technical_metrics.get('histogram_clipping_shadows', 0.0))
    clipping = max(highlights, shadows)

    if clipping > profile["clip_penalty_high"]:
        base_score -= 20
    elif clipping > profile["clip_penalty_mid"]:
        base_score -= 10
    elif clipping < profile["clip_bonus_max"]:
        base_score += 5
    # else: neutral zone (between clip_bonus_max and clip_penalty_mid) - no adjustment

    # Noise contribution
    noise_score = float(technical_metrics.get('noise_score', 0.0))
    if noise_score > profile["noise_high"]:
        base_score -= profile["noise_penalty_high"]
    elif noise_score > profile["noise_warn"]:
        base_score -= profile["noise_penalty_mid"]

    # Color cast contribution
    color_cast = technical_metrics.get('color_cast', 'neutral')
    if color_cast != 'neutral':
        base_score -= profile["color_cast_penalty"]

    post_score = max(0, min(100, int(round(base_score))))
    return post_score


def create_enhanced_prompt(base_prompt: str, exif_data: Dict, technical_metrics: Dict) -> str:
    """Enhance prompt with technical context from image analysis, including context awareness."""
    context_parts = []
    
    # Add image context classification
    image_context = technical_metrics.get('context', 'stock_product')
    context_profile_name = technical_metrics.get('context_profile', 'Stock/Product Photography')
    context_parts.append(f"IMAGE CONTEXT: {context_profile_name}")
    
    profile = get_profile(image_context)
    context_parts.append(f"Expected brightness: {profile['brightness_range'][0]}-{profile['brightness_range'][1]}")
    context_parts.append(f"Sharpness thresholds: critical<{profile['sharpness_crit']}, soft<{profile['sharpness_soft']}")
    context_parts.append(f"Clipping tolerance: warn>{profile['clip_warn']}%, penalty>{profile['clip_penalty_mid']}%")
    context_parts.append(f"Color cast sensitivity: {profile['color_cast_penalty']} point penalty if detected")
    
    # Add EXIF context
    if exif_data.get('iso'):
        iso_val = exif_data['iso']
        if isinstance(iso_val, str):
            context_parts.append(f"ISO: {iso_val}")
        elif iso_val > 3200:
            context_parts.append(f"High ISO ({iso_val}) - check for noise")
        elif iso_val > 800:
            context_parts.append(f"Moderate ISO ({iso_val})")
        else:
            context_parts.append(f"ISO: {iso_val}")
    
    if exif_data.get('aperture'):
        context_parts.append(f"Aperture: {exif_data['aperture']}")
    
    if exif_data.get('shutter_speed'):
        context_parts.append(f"Shutter: {exif_data['shutter_speed']}")
    
    # Add technical analysis context
    brightness = technical_metrics.get('brightness')
    if brightness is not None:
        context_parts.append(f"Measured brightness: {brightness:.0f}")
    
    highlights = technical_metrics.get('histogram_clipping_highlights')
    if highlights is not None:
        context_parts.append(f"Highlight clipping: {highlights:.1f}%")

    shadows = technical_metrics.get('histogram_clipping_shadows')
    if shadows is not None:
        context_parts.append(f"Shadow clipping: {shadows:.1f}%")

    sharpness = technical_metrics.get('sharpness')
    if sharpness is not None:
        context_parts.append(f"Sharpness metric: {sharpness:.1f}")

    noise_score = technical_metrics.get('noise_score', 0)
    if noise_score > 0:
        context_parts.append(f"Noise score: {noise_score:.1f}/100")

    color_cast = technical_metrics.get('color_cast')
    if color_cast and color_cast != 'neutral':
        context_parts.append(f"Color cast: {color_cast}")

    for warning in technical_metrics.get('warnings', []):
        context_parts.append(f"Warning: {warning}")
    
    # Build enhanced prompt
    if context_parts:
        technical_context = "\n\nTECHNICAL CONTEXT:\n" + "\n".join(f"- {part}" for part in context_parts)
        technical_context += "\n\nConsider these technical factors and the image context in your evaluation. "
        technical_context += f"Note that this image is classified as '{context_profile_name}', which may have different "
        technical_context += "expectations for technical attributes like clipping, sharpness, and noise tolerance."
        return base_prompt + technical_context
    
    return base_prompt


def build_pyiqa_metadata(raw_score: float, calibrated_score: float, model_name: str, score_shift: float = 0.0) -> Dict[str, Any]:
    """Build metadata entries for PyIQA-based scoring."""
    overall_score = max(1, min(100, int(round(calibrated_score + score_shift))))
    metadata: Dict[str, Any] = {
        'overall_score': str(overall_score),
        'technical_score': '',
        'composition_score': str(overall_score),
        'lighting_score': str(overall_score),
        'creativity_score': str(overall_score),
        'score': str(overall_score),
        'title': f"PyIQA {model_name} score",
        'description': f"PyIQA {model_name} raw {raw_score:.4f} (scaled {calibrated_score:.2f}, calibrated {overall_score}/100)",
        'keywords': f"pyiqa,{model_name},automated score",
        'pyiqa_raw_score': f"{raw_score:.4f}",
        'pyiqa_scaled_score': f"{calibrated_score:.4f}",
        'pyiqa_model': model_name,
        'technical_metrics': {},
        'technical_warnings': [],
        'post_process_potential': overall_score,
    }
    return metadata


def analyze_image_with_context(image_path: str, ollama_host_url: str, model: str,
                               context_override: Optional[str], skip_context_classification: bool
                               ) -> Tuple[str, Dict, Dict, List[str]]:
    """Determine context, extract EXIF, and compute technical metrics for an image."""
    if context_override:
        image_context = context_override if context_override in TECH_PROFILES else 'stock_product'
        logger.info(f"Using manual context override: {image_context}")
    elif skip_context_classification:
        image_context = 'stock_product'
        logger.debug(f"Context classification disabled, using default: {image_context}")
    else:
        try:
            image_context = classify_image_context(image_path, ollama_host_url, model)
        except Exception as e:
            logger.warning(f"Context classification failed for {image_path}: {e}, using default")
            image_context = 'stock_product'

    logger.info(f"Image context for {image_path}: {image_context} ({TECH_PROFILES[image_context]['name']})")

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

    return image_context, exif_data, technical_metrics, technical_warnings


def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    """Configure logging with file handler and appropriate level."""
    # Set level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    
    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()
    
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
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


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
            image_path
        ]
        
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


def make_single_evaluation(image_path: str, encoded_image: str, ollama_host_url: str, model: str, 
                          prompt: str, headers: Dict) -> Optional[Dict]:
    """Make a single evaluation API call."""
    payload = {
        "model": model,
        "stream": False,
        "images": [encoded_image],
        "prompt": prompt,
        "format": {
            "type": "object",
            "properties": {
                "technical_score": {"type": ["integer", "string"]},
                "composition_score": {"type": ["integer", "string"]},
                "lighting_score": {"type": ["integer", "string"]},
                "creativity_score": {"type": ["integer", "string"]},
                "overall_score": {"type": ["integer", "string"]},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "keywords": {"type": "string"}
            },
            "required": [
                "technical_score",
                "composition_score",
                "lighting_score",
                "creativity_score",
                "overall_score",
                "title",
                "description",
                "keywords"
            ]
        },
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }

    def make_request():
        response = requests.post(ollama_host_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response
    
    try:
        response = retry_with_backoff(make_request)
    except Exception as e:
        logger.error(f"Request failed after retries for {image_path}: {e}")
        return None
    
    if response and response.status_code == 200:
        response_data = response.json()
        metadata_text = response_data.get('response') or response_data.get('thinking', '')
        
        if not metadata_text:
            logger.error(f"No response or thinking field found for {image_path}")
            return None
        
        # Check for content policy violations
        metadata_lower = metadata_text.lower()
        if 'violates content policies' in metadata_lower or 'cannot proceed' in metadata_lower:
            logger.warning(f"Model refused due to content policy: {image_path}")
            return None
        
        try:
            response_metadata = json.loads(metadata_text)
            
            # Validate all score fields
            score_fields = ['technical_score', 'composition_score', 'lighting_score', 'creativity_score', 'overall_score']
            for field in score_fields:
                if field in response_metadata:
                    validated = validate_score(response_metadata[field])
                    if validated is None:
                        logger.warning(f"Invalid {field} for {image_path}")
                        return None
                    response_metadata[field] = str(validated)
                else:
                    logger.warning(f"Missing {field} in response for {image_path}")
                    return None
            
            return response_metadata
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {image_path}: {e}")
            return None
    
    return None


def ensemble_evaluate(image_path: str, encoded_image: str, ollama_host_url: str, model: str,
                     prompt: str, headers: Dict, num_passes: int = 3) -> Optional[Dict]:
    """Perform ensemble evaluation with multiple passes and aggregate results."""
    if num_passes == 1:
        return make_single_evaluation(image_path, encoded_image, ollama_host_url, model, prompt, headers)
    
    logger.info(f"Performing {num_passes}-pass ensemble evaluation for {image_path}")
    evaluations = []
    
    for pass_num in range(num_passes):
        logger.debug(f"Ensemble pass {pass_num + 1}/{num_passes} for {image_path}")
        result = make_single_evaluation(image_path, encoded_image, ollama_host_url, model, prompt, headers)
        if result:
            evaluations.append(result)
        time.sleep(0.5)  # Small delay between passes
    
    if not evaluations:
        logger.error(f"All ensemble passes failed for {image_path}")
        return None
    
    # Aggregate scores (median for robustness)
    score_fields = ['technical_score', 'composition_score', 'lighting_score', 'creativity_score', 'overall_score']
    aggregated = {}
    
    for field in score_fields:
        scores = [int(e[field]) for e in evaluations if field in e]
        if scores:
            # Use median for robust aggregation
            scores.sort()
            median_idx = len(scores) // 2
            aggregated[field] = str(scores[median_idx] if len(scores) % 2 == 1 else (scores[median_idx - 1] + scores[median_idx]) // 2)
    
    # Use first evaluation's text fields
    aggregated['title'] = evaluations[0].get('title', '')
    aggregated['description'] = evaluations[0].get('description', '')
    aggregated['keywords'] = evaluations[0].get('keywords', '')
    
    # Calculate score variance for logging
    overall_scores = [int(e['overall_score']) for e in evaluations if 'overall_score' in e]
    if len(overall_scores) > 1:
        variance = sum((s - sum(overall_scores)/len(overall_scores))**2 for s in overall_scores) / len(overall_scores)
        std_dev = variance ** 0.5
        logger.info(f"Ensemble std dev for {image_path}: {std_dev:.2f} (scores: {overall_scores})")
    
    return aggregated


def process_single_image(image_path: str, ollama_host_url: str, model: str, prompt: str, 
                        dry_run: bool = False, backup_dir: Optional[str] = None, verify: bool = False,
                        cache_dir: Optional[str] = None, ensemble_passes: int = 1,
                        context_override: Optional[str] = None, skip_context_classification: bool = False,
                        scoring_engine: str = 'ollama',
                        pyiqa_backend: Optional[PyIqaScorer] = None) -> Tuple[str, Optional[Dict]]:
    """Process a single image using the selected scoring engine and return result."""
    logger.debug(f"Processing image: {image_path}")
    headers = {'Content-Type': 'application/json'}
    
    # Dry run mode - just return dummy data
    if dry_run:
        logger.debug(f"Dry run - skipping processing for {image_path}")
        return (image_path, {'score': '0', 'title': '[DRY RUN]', 'description': 'Would process this image', 'keywords': 'dry-run'})
    
    # Check cache first
    cached_metadata = load_from_cache(image_path, model, cache_dir) if cache_dir else None
    if cached_metadata:
        logger.info(f"Using cached result for {image_path}")
        # Still embed metadata even if cached
        try:
            embed_metadata(image_path, cached_metadata, backup_dir, verify)
            return (image_path, cached_metadata)
        except Exception as e:
            logger.error(f"Error embedding cached metadata for {image_path}: {e}")
            return (image_path, None)
    
    # Encode image once for all operations (only needed for Ollama backend)
    encoded_image = None
    if scoring_engine == 'ollama':
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    technical_metrics: Dict[str, Any] = {}
    technical_warnings: List[str] = []
    exif_data: Dict = {}
    
    if scoring_engine == 'ollama':
        _, exif_data, technical_metrics, technical_warnings = analyze_image_with_context(
            image_path, ollama_host_url, model, context_override, skip_context_classification
        )
    
    # Step 3: Enhance prompt with technical context (only needed for Ollama backend)
    enhanced_prompt = prompt
    if scoring_engine == 'ollama':
        enhanced_prompt = create_enhanced_prompt(prompt, exif_data, technical_metrics)

    if scoring_engine == 'pyiqa':
        if not pyiqa_backend:
            logger.error("PyIQA backend requested but not initialized.")
            return (image_path, None)
        try:
            score_value = pyiqa_backend.score_batch([image_path]).get(image_path)
            if score_value is None:
                logger.error(f"PyIQA did not return a score for {image_path}")
                return (image_path, None)
            scaled_score = pyiqa_backend.convert_score(score_value)
            response_metadata = build_pyiqa_metadata(score_value, scaled_score, pyiqa_backend.model_name, pyiqa_backend.score_shift)
        except Exception as e:
            logger.error(f"PyIQA evaluation failed for {image_path}: {e}")
            return (image_path, None)

        if cache_dir:
            save_to_cache(image_path, model, response_metadata, cache_dir)

        try:
            embed_metadata(image_path, response_metadata, backup_dir, verify)
            logger.info(f"PyIQA score for {image_path}: {response_metadata.get('score')}")
            return (image_path, response_metadata)
        except Exception as e:
            logger.error(f"Error embedding PyIQA metadata for {image_path}: {e}")
            return (image_path, None)

    # Step 4: Retry logic for malformed responses (Ollama backend)
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Perform ensemble evaluation
            response_metadata = ensemble_evaluate(image_path, encoded_image, ollama_host_url, 
                                                 model, enhanced_prompt, headers, ensemble_passes)
            
            if response_metadata is None:
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying {image_path} (attempt {attempt + 2}/{max_attempts})")
                    time.sleep(2)
                    continue
                return (image_path, None)

            # Add 'score' field for backwards compatibility (use overall_score)
            if 'overall_score' in response_metadata:
                response_metadata['score'] = response_metadata['overall_score']

            response_metadata['technical_metrics'] = technical_metrics
            response_metadata['technical_warnings'] = technical_warnings
            response_metadata['post_process_potential'] = technical_metrics.get('post_process_potential')

            # Save to cache
            if cache_dir:
                save_to_cache(image_path, model, response_metadata, cache_dir)
                logger.debug(f"Saved response to cache for {image_path}")

            # Embed metadata
            try:
                embed_metadata(image_path, response_metadata, backup_dir, verify)
                logger.info(f"Successfully processed {image_path} with score {response_metadata.get('score')}")
                return (image_path, response_metadata)
            except Exception as e:
                logger.error(f"Error embedding metadata for {image_path}: {e}")
                return (image_path, None)
                
        except Exception as e:
            logger.error(f"Unexpected error processing {image_path}: {e}")
            if attempt < max_attempts - 1:
                logger.info(f"Retrying {image_path} (attempt {attempt + 2}/{max_attempts})")
                time.sleep(2)
                continue
            return (image_path, None)
    
    # Should never reach here
    return (image_path, None)


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
    model: str,
    cache_dir: Optional[str],
    backup_dir: Optional[str],
    verify: bool,
    pyiqa_backend: "PyIqaScorer",
    min_score: Optional[int],
    batch_size: int,
    dry_run: bool
) -> List[Tuple[str, Optional[Dict]]]:
    results: List[Tuple[str, Optional[Dict]]] = []
    pending: List[str] = []

    def flush_pending() -> List[Tuple[str, Optional[Dict]]]:
        if not pending:
            return []
        batch = pending.copy()
        pending.clear()
        scores = pyiqa_backend.score_batch(batch)
        batch_results: List[Tuple[str, Optional[Dict]]] = []
        for image_path in batch:
            score_value = scores.get(image_path)
            if score_value is None:
                batch_results.append((image_path, None))
                continue
            scaled_score = pyiqa_backend.convert_score(score_value)
            response_metadata = build_pyiqa_metadata(score_value, scaled_score, pyiqa_backend.model_name, pyiqa_backend.score_shift)
            if cache_dir:
                save_to_cache(image_path, model, response_metadata, cache_dir)
            try:
                embed_metadata(image_path, response_metadata, backup_dir, verify)
                logger.info(f"PyIQA score for {image_path}: {response_metadata.get('score')}")
                batch_results.append((image_path, response_metadata))
            except Exception as exc:
                logger.error(f"Failed to embed PyIQA metadata for {image_path}: {exc}")
                batch_results.append((image_path, None))
        return batch_results

    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
        for image_path in image_paths:
            immediate = _handle_cached_or_dry_run(image_path, cache_dir, model, backup_dir, verify, dry_run=dry_run)
            if immediate:
                path, metadata = immediate
                if min_score is None or metadata is None:
                    results.append((path, metadata))
                else:
                    try:
                        score = int(validate_score(metadata.get('score', '0')) or 0)
                        if score >= min_score:
                            results.append((path, metadata))
                    except (ValueError, TypeError):
                        results.append((path, metadata))
                pbar.update(1)
                continue

            pending.append(image_path)
            if len(pending) >= batch_size:
                batch_results = flush_pending()
                pbar.update(len(batch_results))
                for path, metadata in batch_results:
                    if min_score is not None and metadata is not None:
                        try:
                            score = int(validate_score(metadata.get('score', '0')) or 0)
                            if score < min_score:
                                continue
                        except (ValueError, TypeError):
                            pass
                    results.append((path, metadata))

        batch_results = flush_pending()
        pbar.update(len(batch_results))
        for path, metadata in batch_results:
            if min_score is not None and metadata is not None:
                try:
                    score = int(validate_score(metadata.get('score', '0')) or 0)
                    if score < min_score:
                        continue
                except (ValueError, TypeError):
                    pass
            results.append((path, metadata))

    return results


def process_images_in_folder(folder_path: str, ollama_host_url: str, max_workers: int = 4, 
                            model: str = DEFAULT_MODEL, prompt: str = DEFAULT_PROMPT,
                            file_types: Optional[List[str]] = None, skip_existing: bool = True,
                            dry_run: bool = False, min_score: Optional[int] = None,
                            backup_dir: Optional[str] = None, verify: bool = False,
                            cache_dir: Optional[str] = None, ensemble_passes: int = 1,
                            context_override: Optional[str] = None,
                            skip_context_classification: bool = False,
                            scoring_engine: str = 'pyiqa',
                            pyiqa_backend: Optional[PyIqaScorer] = None,
                            pyiqa_batch_size: int = 4) -> List[Tuple[str, Optional[Dict]]]:
    """Process images with parallel execution and progress bar."""
    # Collect all images to process
    image_paths = collect_images(folder_path, file_types=file_types, skip_existing=skip_existing)
    
    if not image_paths:
        logger.warning("No images found to process")
        return []
    
    results: List[Tuple[str, Optional[Dict]]] = []

    # Use PyIQA batch processing if that engine is selected
    if scoring_engine == 'pyiqa' and pyiqa_backend is not None:
        return process_images_with_pyiqa(
            image_paths=image_paths,
            model=model,
            cache_dir=cache_dir,
            backup_dir=backup_dir,
            verify=verify,
            pyiqa_backend=pyiqa_backend,
            min_score=min_score,
            batch_size=pyiqa_batch_size,
            dry_run=dry_run
        )

    # Ollama processing with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image, image_path, ollama_host_url, model, prompt,
                dry_run, backup_dir, verify, cache_dir, ensemble_passes,
                context_override, skip_context_classification, scoring_engine, pyiqa_backend
            ): image_path
            for image_path in image_paths
        }
        
        with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
            for future in as_completed(futures):
                image_path = futures[future]
                try:
                    result = future.result()
                    if min_score is not None and result[1] is not None:
                        try:
                            score = int(validate_score(result[1].get('score', '0')) or 0)
                            if score < min_score:
                                pbar.update(1)
                                continue
                        except (ValueError, TypeError):
                            pass
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    results.append((image_path, None))
                pbar.update(1)
    
    return results


def save_results_to_csv(results: List[Tuple[str, Optional[Dict]]], output_path: str):
    """Save processing results to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'file_path', 'overall_score', 'technical_score', 'composition_score',
            'lighting_score', 'creativity_score', 'title', 'description',
            'keywords', 'status', 'context', 'context_profile', 'sharpness', 'brightness', 'contrast',
            'histogram_clipping_highlights', 'histogram_clipping_shadows',
            'color_cast', 'noise_sigma', 'noise_score', 'technical_warnings', 'post_process_potential',
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
                import re
                match = re.search(r'\d+', score_str)
                if match:
                    score = int(match.group())
                    if 1 <= score <= 100:
                        scores.append(score)
            except (ValueError, AttributeError):
                continue
        if metadata and metadata.get('technical_score'):
            try:
                tech_val = int(str(metadata['technical_score']))
                if 1 <= tech_val <= 100:
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
    if cli_args[0] in {'process', 'rollback', 'stats'}:
        return cli_args, None
    return ['process'] + cli_args, 'inferred'


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
        help='Path to the folder containing images (default: current working directory or IMAGE_EVAL_DEFAULT_FOLDER)'
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
                               help=f'Model identifier (LLM name or custom label). Default: {DEFAULT_MODEL}')
    process_parser.add_argument('--score-engine', choices=['ollama', 'pyiqa'], default='pyiqa',
                               help='Choose between Ollama (LLM) or PyIQA scoring (default: PyIQA)')
    process_parser.add_argument('--pyiqa-model', type=str, default='clipiqa+_vitl14_512',
                               help='PyIQA metric name to use when --score-engine pyiqa (default: clipiqa+_vitl14_512)')
    process_parser.add_argument('--pyiqa-device', type=str, default=None,
                               help='Device for PyIQA (e.g., cuda:0 or cpu). Defaults to CUDA if available.')
    process_parser.add_argument('--pyiqa-score-shift', type=float, default=None,
                               help='Additive adjustment (0-100 scale) applied to PyIQA scores (default: model-specific)')
    process_parser.add_argument('--pyiqa-scale-factor', type=float, default=None,
                               help='Optional multiplier applied to PyIQA raw scores before calibration (auto-detect if omitted)')
    process_parser.add_argument('--pyiqa-batch-size', type=int, default=4,
                               help='Images per batch for PyIQA evaluation (default: 4)')
    process_parser.add_argument('--prompt-file', type=str, default=None, help='Path to custom prompt file (overrides default prompt)')
    process_parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip images with existing metadata (default: True)')
    process_parser.add_argument('--no-skip-existing', action='store_false', dest='skip_existing', help='Process all images, even with existing metadata')
    process_parser.add_argument('--min-score', type=int, default=None, help='Only save results with score >= this value')
    process_parser.add_argument('--file-types', type=str, default=None, help='Comma-separated list of file extensions (e.g., jpg,png,dng)')
    process_parser.add_argument('--dry-run', action='store_true', help='Preview what would be processed without making changes')
    process_parser.add_argument('--backup-dir', type=str, default=None, help='Directory to store backups (default: same directory as originals)')
    process_parser.add_argument('--verify', action='store_true', help='Verify metadata was correctly embedded after writing')
    process_parser.add_argument('--log-file', type=str, default=None, help='Path to log file (default: auto-generated)')
    process_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose debug output')
    process_parser.add_argument('--cache', action='store_true', help='Enable API response caching')
    process_parser.add_argument('--cache-dir', type=str, default=CACHE_DIR, help=f'Cache directory (default: {CACHE_DIR})')
    process_parser.add_argument('--clear-cache', action='store_true', help='Clear cache before processing')
    process_parser.add_argument('--ensemble', type=int, default=DEFAULT_ENSEMBLE_PASSES, 
                              help=f'Number of evaluation passes for ensemble scoring (default: {DEFAULT_ENSEMBLE_PASSES})')
    process_parser.add_argument('--context', type=str, default=None,
                              help='Manual context override (e.g., landscape, portrait_neutral, stock_product)')
    process_parser.add_argument('--no-context-classification', action='store_true',
                              help='Skip automatic context classification, use stock_product for all')
    
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
    
    # Fill in defaults for positional arguments when omitted
    if args.command == 'process':
        folder_defaulted = False
        host_defaulted = False
        if not args.folder_path:
            args.folder_path = DEFAULT_IMAGE_FOLDER
            folder_defaulted = True
        if not args.ollama_host_url:
            args.ollama_host_url = DEFAULT_OLLAMA_URL
            host_defaulted = True
        
        if folder_defaulted:
            print(f"Using default image folder: {args.folder_path}")
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
        
        setup_logging(log_file, args.verbose)
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

    pyiqa_backend = None

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

    if args.score_engine == 'pyiqa':
        if not PYIQA_AVAILABLE:
            logger.error("PyIQA backend requested but dependencies are missing. Install torch and pyiqa.")
            sys.exit(1)
        pyiqa_shift = args.pyiqa_score_shift
        if pyiqa_shift is None:
            pyiqa_shift = get_default_pyiqa_shift(args.pyiqa_model)
        try:
            pyiqa_backend = PyIqaScorer(
                model_name=args.pyiqa_model,
                device=args.pyiqa_device,
                score_shift=pyiqa_shift,
                scale_factor=args.pyiqa_scale_factor,
                max_long_edge=PYIQA_MAX_LONG_EDGE,
            )
        except Exception as e:
            logger.error(f"Failed to initialize PyIQA backend: {e}")
            sys.exit(1)
        args.model = f"pyiqa_{args.pyiqa_model}"

    # Load custom prompt if specified
    prompt = DEFAULT_PROMPT
    if args.prompt_file:
        prompt = load_prompt_from_file(args.prompt_file)
        print(f"Loaded custom prompt from: {args.prompt_file}")
    
    # Parse file types if specified
    file_types = None
    if args.file_types:
        file_types = [ext.strip() for ext in args.file_types.split(',')]
        print(f"Filtering for file types: {', '.join(file_types)}")
    print(f"\nProcessing images from: {Style.BRIGHT}{args.folder_path}{Style.RESET_ALL}")
    print(f"Model: {args.model}")
    print(f"Scoring engine: {args.score_engine}")
    if args.score_engine == 'pyiqa':
        print(f"PyIQA model: {args.pyiqa_model}")
        print(f"PyIQA device: {pyiqa_backend.device if pyiqa_backend else args.pyiqa_device}")
        print(f"PyIQA batch size: {args.pyiqa_batch_size}")
        if args.pyiqa_scale_factor:
            print(f"PyIQA scale factor: {args.pyiqa_scale_factor}")
        if pyiqa_backend:
            print(f"PyIQA score shift: {pyiqa_backend.score_shift:+.2f}")
    print(f"Workers: {args.workers}")
    print(f"Skip existing: {args.skip_existing}")
    if args.ensemble > 1:
        print(f"{Fore.CYAN}Ensemble mode: {args.ensemble} evaluation passes per image{Fore.RESET}")
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
    
    if args.verbose:
        print(f"Verbose logging: enabled")
    
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
        max_workers=args.workers,
        model=args.model,
        prompt=prompt,
        file_types=file_types,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        min_score=args.min_score,
        backup_dir=args.backup_dir,
        verify=args.verify,
        cache_dir=cache_dir,
        ensemble_passes=args.ensemble,
        context_override=args.context,
        skip_context_classification=args.no_context_classification,
        scoring_engine=args.score_engine,
        pyiqa_backend=pyiqa_backend,
        pyiqa_batch_size=args.pyiqa_batch_size
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
