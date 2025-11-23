import argparse
import base64
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json  # Import json for parsing the response
import logging
import logging.handlers
import os
import pickle
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    rawpy = None

RAWPY_AVAILABLE = rawpy is not None
RAWPY_IMPORT_WARNINGED = False
RAW_EXTENSIONS = {'.nef', '.cr2', '.cr3', '.arw', '.rw2', '.raf', '.orf', '.dng'}

# Increase PIL decompression bomb limit for large legitimate images
Image.MAX_IMAGE_PIXELS = None  # Remove limit entirely (or set to a higher value like 500000000)

# Configure basic logging (will be updated based on verbose flag)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def _safe_raw_metadata_value(meta, *names):
    for name in names:
        value = getattr(meta, name, None)
        if value not in (None, 0, 0.0, ''):
            return value
    return None


def _extract_rawpy_metadata(image_path: str) -> Dict[str, Optional[str]]:
    if not RAWPY_AVAILABLE:
        return {}

    try:
        raw = rawpy.imread(image_path)
    except Exception as exc:
        logger.debug(f"rawpy failed to read {image_path}: {exc}")
        return {}

    try:
        meta = getattr(raw, 'metadata', None)
        if meta is None:
            logger.debug(f"rawpy metadata not available for {image_path}")
            return {}
        raw_metadata: Dict[str, Optional[str]] = {}
        iso = _safe_raw_metadata_value(meta, 'iso_speed', 'iso')
        if iso:
            raw_metadata['iso'] = int(iso)

        aperture = _safe_raw_metadata_value(meta, 'aperture', 'f_number')
        if aperture:
            raw_metadata['aperture'] = f"f/{float(aperture):.1f}"

        shutter = _safe_raw_metadata_value(meta, 'shutter', 'exposure_time')
        if shutter:
            raw_metadata['shutter_speed'] = f"{float(shutter):.3f}s"

        focal_length = _safe_raw_metadata_value(meta, 'focal_length')
        if focal_length:
            raw_metadata['focal_length'] = f"{float(focal_length):.0f}mm"

        camera_make = _safe_raw_metadata_value(meta, 'camera_maker', 'camera')
        if camera_make:
            raw_metadata['camera_make'] = str(camera_make).strip()

        camera_model = _safe_raw_metadata_value(meta, 'camera_model')
        if camera_model:
            raw_metadata['camera_model'] = str(camera_model).strip()

        lens_model = _safe_raw_metadata_value(meta, 'lens')
        if lens_model:
            raw_metadata['lens_model'] = str(lens_model).strip()

        return raw_metadata
    finally:
        raw.close()


@contextmanager
def open_image_for_analysis(image_path: str):
    ext = Path(image_path).suffix.lower()
    if ext in RAW_EXTENSIONS:
        if not RAWPY_AVAILABLE:
            _warn_rawpy_missing()
            raise RuntimeError("rawpy is required to process RAW files")
        raw = rawpy.imread(image_path)
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
            raw = rawpy.imread(image_path)
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
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
            time.sleep(delay)
    return None


def extract_exif_metadata(image_path: str) -> Dict[str, Optional[str]]:
    """Extract technical metadata from EXIF data."""
    metadata: Dict[str, Optional[str]] = {
        'iso': None,
        'aperture': None,
        'shutter_speed': None,
        'focal_length': None,
        'camera_make': None,
        'camera_model': None,
        'lens_model': None
    }
    if _is_raw_image(image_path):
        metadata.update(_extract_rawpy_metadata(image_path))
        return metadata
    
    try:
        with Image.open(image_path) as img:
            exif_data = img.info.get("exif")
            if exif_data:
                exif_dict = piexif.load(exif_data)
                
                # Extract ISO
                if piexif.ExifIFD.ISOSpeedRatings in exif_dict.get("Exif", {}):
                    metadata['iso'] = exif_dict["Exif"][piexif.ExifIFD.ISOSpeedRatings]
                
                # Extract Aperture (FNumber)
                if piexif.ExifIFD.FNumber in exif_dict.get("Exif", {}):
                    f_num = exif_dict["Exif"][piexif.ExifIFD.FNumber]
                    if isinstance(f_num, tuple):
                        metadata['aperture'] = f"f/{f_num[0]/f_num[1]:.1f}"
                
                # Extract Shutter Speed (ExposureTime)
                if piexif.ExifIFD.ExposureTime in exif_dict.get("Exif", {}):
                    exp_time = exif_dict["Exif"][piexif.ExifIFD.ExposureTime]
                    if isinstance(exp_time, tuple):
                        metadata['shutter_speed'] = f"{exp_time[0]}/{exp_time[1]}s"
                
                # Extract Focal Length
                if piexif.ExifIFD.FocalLength in exif_dict.get("Exif", {}):
                    focal = exif_dict["Exif"][piexif.ExifIFD.FocalLength]
                    if isinstance(focal, tuple):
                        metadata['focal_length'] = f"{focal[0]/focal[1]:.0f}mm"
                
                # Extract Camera Make/Model
                if piexif.ImageIFD.Make in exif_dict.get("0th", {}):
                    metadata['camera_make'] = exif_dict["0th"][piexif.ImageIFD.Make].decode('utf-8', errors='ignore').strip()
                
                if piexif.ImageIFD.Model in exif_dict.get("0th", {}):
                    metadata['camera_model'] = exif_dict["0th"][piexif.ImageIFD.Model].decode('utf-8', errors='ignore').strip()
                
                # Extract Lens Model
                if piexif.ExifIFD.LensModel in exif_dict.get("Exif", {}):
                    metadata['lens_model'] = exif_dict["Exif"][piexif.ExifIFD.LensModel].decode('utf-8', errors='ignore').strip()
    
    except Exception as e:
        logger.debug(f"Could not extract EXIF metadata from {image_path}: {e}")
    
    return metadata


def analyze_image_technical(image_path: str) -> Dict:
    """Analyze image for technical quality metrics."""
    metrics = {
        'sharpness': 0,
        'brightness': 0,
        'contrast': 0,
        'histogram_clipping_highlights': 0,
        'histogram_clipping_shadows': 0,
        'color_cast': 'neutral'
    }
    
    try:
        # Read image with PIL (or rawpy) for stats
        with open_image_for_analysis(image_path) as img:
            img_rgb = img if img.mode == 'RGB' else img.convert('RGB')
            stat = ImageStat.Stat(img_rgb)
            metrics['brightness'] = sum(stat.mean) / len(stat.mean)
            metrics['contrast'] = sum(stat.stddev) / len(stat.stddev)

            histogram = img_rgb.histogram()
            total_pixels = img_rgb.size[0] * img_rgb.size[1]

            for channel_idx in range(3):  # R, G, B
                channel_hist = histogram[channel_idx * 256:(channel_idx + 1) * 256]
                highlight_pixels = sum(channel_hist[250:256])
                metrics['histogram_clipping_highlights'] += (highlight_pixels / total_pixels) * 100

                shadow_pixels = sum(channel_hist[0:6])
                metrics['histogram_clipping_shadows'] += (shadow_pixels / total_pixels) * 100

            metrics['histogram_clipping_highlights'] /= 3
            metrics['histogram_clipping_shadows'] /= 3

            r_mean, g_mean, b_mean = stat.mean[:3]
            max_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
            if max_diff > 15:
                if r_mean > g_mean and r_mean > b_mean:
                    metrics['color_cast'] = 'warm/red'
                elif b_mean > r_mean and b_mean > g_mean:
                    metrics['color_cast'] = 'cool/blue'
                elif g_mean > r_mean and g_mean > b_mean:
                    metrics['color_cast'] = 'green'

            img_gray = img_rgb.convert('L')
            gray_array = np.array(img_gray)
            laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()
            metrics['sharpness'] = laplacian_var
    
    except Exception as e:
        logger.debug(f"Could not analyze technical metrics for {image_path}: {e}")
    
    return metrics


def assess_technical_metrics(technical_metrics: Dict) -> List[str]:
    """Generate human-readable warnings based on measured metrics."""
    warnings = []
    sharpness = technical_metrics.get('sharpness')
    if sharpness is not None:
        if sharpness < 30:
            warnings.append(f"Sharpness critically low ({sharpness:.1f})")
        elif sharpness < 60:
            warnings.append(f"Lower sharpness ({sharpness:.1f}) may impact detail")

    highlights = technical_metrics.get('histogram_clipping_highlights', 0)
    if highlights > 5:
        warnings.append(f"Highlight clipping {highlights:.1f}% reduces tonal range")

    shadows = technical_metrics.get('histogram_clipping_shadows', 0)
    if shadows > 5:
        warnings.append(f"Shadow clipping {shadows:.1f}% removes shadow detail")

    color_cast = technical_metrics.get('color_cast', 'neutral')
    if color_cast != 'neutral':
        warnings.append(f"Color cast detected: {color_cast}")

    return warnings


def compute_post_process_potential(technical_metrics: Dict) -> int:
    """Estimate how much post-processing can improve this image."""
    base_score = 70
    sharpness = technical_metrics.get('sharpness')
    if sharpness is not None:
        if sharpness < 35:
            base_score -= 25
        elif sharpness < 60:
            base_score -= 10
        else:
            base_score += 5

    highlights = technical_metrics.get('histogram_clipping_highlights', 0)
    shadows = technical_metrics.get('histogram_clipping_shadows', 0)
    clipping = max(highlights, shadows)
    if clipping > 12:
        base_score -= 20
    elif clipping > 6:
        base_score -= 10
    elif clipping == 0:
        base_score += 5

    color_cast = technical_metrics.get('color_cast', 'neutral')
    if color_cast != 'neutral':
        base_score -= 8

    post_score = max(0, min(100, int(base_score)))
    return post_score


def create_enhanced_prompt(base_prompt: str, exif_data: Dict, technical_metrics: Dict) -> str:
    """Enhance prompt with technical context from image analysis."""
    context_parts = []
    
    # Add EXIF context
    if exif_data.get('iso'):
        iso_val = exif_data['iso']
        if iso_val > 3200:
            context_parts.append(f"High ISO ({iso_val}) - check for noise")
        elif iso_val > 800:
            context_parts.append(f"Moderate ISO ({iso_val})")
    
    if exif_data.get('aperture'):
        context_parts.append(f"Aperture: {exif_data['aperture']}")
    
    if exif_data.get('shutter_speed'):
        context_parts.append(f"Shutter: {exif_data['shutter_speed']}")
    
    # Add technical analysis context
    highlights = technical_metrics.get('histogram_clipping_highlights')
    if highlights is not None and highlights > 12:
        context_parts.append(f"Highlight clipping: {highlights:.1f}%")

    shadows = technical_metrics.get('histogram_clipping_shadows')
    if shadows is not None and shadows > 12:
        context_parts.append(f"Shadow clipping: {shadows:.1f}%")

    sharpness = technical_metrics.get('sharpness')
    if sharpness is not None and sharpness < 35:
        context_parts.append(f"Sharpness metric: {sharpness:.1f}")

    color_cast = technical_metrics.get('color_cast')
    if color_cast and color_cast != 'neutral':
        context_parts.append(f"Color cast: {color_cast}")

    for warning in technical_metrics.get('warnings', []):
        context_parts.append(f"Warning: {warning}")
    
    # Build enhanced prompt
    if context_parts:
        technical_context = "\n\nTECHNICAL CONTEXT:\n" + "\n".join(f"- {part}" for part in context_parts)
        technical_context += "\n\nConsider these technical factors in your evaluation."
        return base_prompt + technical_context
    
    return base_prompt


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
        cmd = [
            'exiftool',
            '-overwrite_original',  # Don't create _original files
            f'-UserComment={metadata.get("score", "")}',
            f'-XPTitle={metadata.get("title", "")}',
            f'-XPComment={metadata.get("description", "")}',
            f'-XPKeywords={metadata.get("keywords", "")}',
            image_path
        ]
        
        # Create manual backup before modifying
        if not os.path.exists(backup_image_path):
            import shutil
            shutil.copy2(image_path, backup_image_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{Fore.BLUE}Embedding score:{Fore.RESET} {Fore.GREEN}{metadata.get('score', '')}{Fore.RESET}")
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
                        cache_dir: Optional[str] = None, ensemble_passes: int = 1) -> Tuple[str, Optional[Dict]]:
    """Process a single image and return result."""
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
    
    # Extract EXIF and technical metrics
    exif_data = extract_exif_metadata(image_path)
    technical_metrics = analyze_image_technical(image_path)
    technical_warnings = assess_technical_metrics(technical_metrics)
    technical_metrics['warnings'] = technical_warnings
    technical_metrics['post_process_potential'] = compute_post_process_potential(technical_metrics)

    # Enhance prompt with technical context
    enhanced_prompt = create_enhanced_prompt(prompt, exif_data, technical_metrics)
    
    # Retry logic for malformed responses
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
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


def process_images_in_folder(folder_path: str, ollama_host_url: str, max_workers: int = 4, 
                            model: str = DEFAULT_MODEL, prompt: str = DEFAULT_PROMPT,
                            file_types: Optional[List[str]] = None, skip_existing: bool = True,
                            dry_run: bool = False, min_score: Optional[int] = None,
                            backup_dir: Optional[str] = None, verify: bool = False,
                            cache_dir: Optional[str] = None, ensemble_passes: int = 1) -> List[Tuple[str, Optional[Dict]]]:
    """Process images with parallel execution and progress bar."""
    # Collect all images to process
    image_paths = collect_images(folder_path, file_types=file_types, skip_existing=skip_existing)
    
    if not image_paths:
        logger.warning("No images found to process")
        return []
    
    results = []
    
    # Process images in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_single_image, path, ollama_host_url, model, prompt, dry_run, backup_dir, verify, cache_dir, ensemble_passes): path 
            for path in image_paths
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
            for future in as_completed(future_to_path):
                result = future.result()
                
                # Filter by min_score if specified
                if min_score is not None and result[1] is not None:
                    try:
                        score = int(validate_score(result[1].get('score', '0')) or 0)
                        if score < min_score:
                            pbar.update(1)
                            continue  # Skip images below minimum score
                    except (ValueError, TypeError):
                        pass
                
                results.append(result)
                pbar.update(1)
    
    return results


def save_results_to_csv(results: List[Tuple[str, Optional[Dict]]], output_path: str):
    """Save processing results to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'file_path', 'overall_score', 'technical_score', 'composition_score',
            'lighting_score', 'creativity_score', 'title', 'description',
            'keywords', 'status', 'sharpness', 'brightness', 'contrast',
            'histogram_clipping_highlights', 'histogram_clipping_shadows',
            'color_cast', 'technical_warnings', 'post_process_potential'
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
                    'sharpness': technical_metrics.get('sharpness', ''),
                    'brightness': technical_metrics.get('brightness', ''),
                    'contrast': technical_metrics.get('contrast', ''),
                    'histogram_clipping_highlights': technical_metrics.get('histogram_clipping_highlights', ''),
                    'histogram_clipping_shadows': technical_metrics.get('histogram_clipping_shadows', ''),
                    'color_cast': technical_metrics.get('color_cast', ''),
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
                    'sharpness': '',
                    'brightness': '',
                    'contrast': '',
                    'histogram_clipping_highlights': '',
                    'histogram_clipping_shadows': '',
                    'color_cast': '',
                    'technical_warnings': '',
                    'post_process_potential': ''
                })


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
        'avg_post_process_potential': sum(potentials)/len(potentials) if potentials else 0
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
        avg_post = stats.get('avg_post_process_potential')
        if avg_post is not None:
            print(f"Average post-process potential: {avg_post:.1f}/100")
        
        print(f"\n{'='*60}")
        print(f"SCORE DISTRIBUTION")
        print(f"{'='*60}")
        for bin_range, count in sorted(stats['score_distribution'].items()):
            bar = '█' * count
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
    if cli_args[0] in {'process', 'rollback'}:
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
    process_parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help=f'Ollama model to use (default: {DEFAULT_MODEL})')
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
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Restore images from backups')
    rollback_parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    rollback_parser.add_argument('--backup-dir', type=str, default=None, help='Directory where backups are stored')
    
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
        ensemble_passes=args.ensemble
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
