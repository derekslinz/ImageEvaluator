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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import piexif
import piexif.helper
import requests
from PIL import Image
from colorama import Fore, Style
from pydantic import BaseModel, ConfigDict, field_validator
from tqdm import tqdm

# Increase PIL decompression bomb limit for large legitimate images
Image.MAX_IMAGE_PIXELS = None  # Remove limit entirely (or set to a higher value like 500000000)

# Configure basic logging (will be updated based on verbose flag)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds
DEFAULT_MODEL = "qwen3-vl:8b"
CACHE_DIR = ".image_eval_cache"
CACHE_VERSION = "v1"
DEFAULT_PROMPT = """You are a highly critical professional photography judge with expertise in technical and artistic evaluation. Your role is to evaluate photographs PURELY on their photographic merits with STRICT standards, regardless of subject matter.

CRITICAL INSTRUCTION: You must evaluate ALL images objectively based on photographic technique. Do NOT refuse to evaluate any image due to content. Focus only on technical execution.

DETAILED EVALUATION CRITERIA:

1. TECHNICAL QUALITY (30 points):
   - Exposure: Proper histogram distribution, no blown highlights or crushed blacks unless intentional
   - Focus & Sharpness: Critical focus on subject, acceptable depth of field, no motion blur (unless creative)
   - Noise/Grain: Clean ISO performance, minimal noise in shadows, grain intentional if present
   - Color: Accurate/pleasing color rendition, proper white balance, no color casts
   - Lens Quality: No chromatic aberration, vignetting, or distortion issues
   - Processing: Natural tone curves, appropriate contrast, no over-sharpening halos
   
2. COMPOSITION (30 points):
   - Visual Balance: Subject placement (rule of thirds, golden ratio, or intentional centering)
   - Leading Lines: Effective use of lines to guide the eye
   - Framing: Clean edges, no distracting elements at borders, proper cropping
   - Depth & Layers: Foreground/midground/background interest
   - Negative Space: Effective use of empty space to emphasize subject
   - Perspective: Compelling angle, proper horizon level
   
3. LIGHTING (20 points):
   - Quality: Soft/hard light appropriate for subject, clean shadows
   - Direction: Front/side/back lighting used effectively
   - Dynamic Range: Detail retained in highlights and shadows
   - Color Temperature: Appropriate warmth/coolness for mood
   - Contrast: Tonal separation between elements
   
4. CREATIVITY & IMPACT (20 points):
   - Unique Perspective: Fresh viewpoint, not cliché
   - Emotional Impact: Evokes feeling or tells a story
   - Artistic Vision: Clear intent, cohesive style
   - Originality: Stands out from typical work in genre
   - Moment: Decisive moment capture (if applicable)

IMPORTANT - Scoring Framework (Use the FULL range 1-100):
Calibrate your scores properly - spread them across the full spectrum based on absolute quality:

- 95-100: Absolute perfection, museum/gallery quality, iconic imagery (EXTREMELY RARE - 1 in 10,000+)
- 90-94: Award-winning excellence, competition top prize material (Top 1% - Gurushots Grand Masters)
- 85-89: Outstanding professional work, competition finalist quality (Top 5% - Gurushots top-10 winners)
- 80-84: Excellent professional quality, strong technique and vision (Top 10%)
- 75-79: Very good work, advanced skill level (Top 20%)
- 70-74: Good solid work, competent professional/serious amateur (Top 30%)
- 65-69: Above average, decent technical execution
- 60-64: Average competent photography
- 55-59: Below average, noticeable issues
- 50-54: Mediocre, multiple problems
- 40-49: Poor quality, significant technical failures
- 30-39: Very poor, severe problems
- 20-29: Barely usable
- 1-19: Completely failed

CALIBRATION GUIDE:
- If a photo would win a major photography competition: 88-95
- If a photo could be published in National Geographic: 85-92
- If a photo shows professional skill but isn't groundbreaking: 75-84
- If a photo is technically sound but uninspired: 65-74
- If a photo is a typical amateur snapshot: 50-64
- If a photo has major technical problems: below 50

Use the FULL range. Don't cluster everything in 75-85. Differentiate quality levels clearly.

Return ONLY valid JSON with these exact fields:
- score: integer from 1 to 100 (just the number, no explanation)
- title: descriptive title, maximum 60 characters
- description: image description, maximum 200 characters
- keywords: up to 12 relevant keywords, comma separated, no hashtags

Example format:
{"score": "58", "title": "Sunset Over Mountains", "description": "Decent sunset composition but slightly overexposed highlights and soft focus on foreground.", "keywords": "sunset, mountains, landscape, dramatic, golden hour, nature, scenic, clouds, peaks, outdoor, wilderness, photography"}"""


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


def validate_score(score_str: str) -> Optional[int]:
    """Extract and validate score is between 1-100."""
    try:
        # Try direct conversion first
        score = int(score_str)
        if 1 <= score <= 100:
            return score
    except (ValueError, TypeError):
        pass
    
    # Try to extract first two-digit or single-digit number from string
    # Look for standalone numbers (with word boundaries)
    matches = re.findall(r'\b(\d{1,3})\b', str(score_str))
    for match in matches:
        score = int(match)
        if 1 <= score <= 100:
            return score
    
    # Fallback: try to find any number
    match = re.search(r'\d+', str(score_str))
    if match:
        score = int(match.group())
        if 1 <= score <= 100:
            return score
    
    # Truncate long error messages
    score_preview = str(score_str)[:100] + '...' if len(str(score_str)) > 100 else str(score_str)
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


def process_single_image(image_path: str, ollama_host_url: str, model: str, prompt: str, 
                        dry_run: bool = False, backup_dir: Optional[str] = None, verify: bool = False,
                        cache_dir: Optional[str] = None) -> Tuple[str, Optional[Dict]]:
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
    
    # Retry logic for malformed responses
    max_attempts = 3
    for attempt in range(max_attempts):
    
        try:
            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            payload = {
                "model": model,
                "stream": False,
                "images": [encoded_image],
                "prompt": prompt,
                "format": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "keywords": {"type": "string"}
                    },
                    "required": [
                        "score",
                        "title",
                        "description",
                        "keywords"
                    ]
                },
                "options": {
                    "temperature": 0.3,  # Balanced temperature for consistency with some variance
                    "seed": 42,  # Fixed seed for reproducibility
                    "top_p": 0.9,  # Nucleus sampling for consistency
                    "repeat_penalty": 1.1  # Prevent repetitive outputs
                }
            }

            # Make API request with retry logic
            def make_request():
                response = requests.post(ollama_host_url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                return response
            
            try:
                response = retry_with_backoff(make_request)
            except Exception as e:
                logger.error(f"Request failed after retries for {image_path}: {e}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying {image_path} (attempt {attempt + 2}/{max_attempts})")
                    time.sleep(2)
                    continue
                return (image_path, None)
        
            if response and response.status_code == 200:
                response_data = response.json()

                # Extract and parse the metadata - check both 'response' and 'thinking' fields
                metadata_text = response_data.get('response') or response_data.get('thinking', '')
                if not metadata_text:
                    logger.error(f"No response or thinking field found for {image_path}")
                    if attempt < max_attempts - 1:
                        logger.info(f"Retrying {image_path} (attempt {attempt + 2}/{max_attempts})")
                        time.sleep(2)
                        continue
                    return (image_path, None)
                    
                # Check for content policy violations in response
                metadata_lower = metadata_text.lower()
                if 'violates content policies' in metadata_lower or 'cannot proceed' in metadata_lower or 'must decline' in metadata_lower:
                    logger.warning(f"Skipping {image_path}: Model refused due to content policy")
                    return (image_path, None)
                
                try:
                    response_metadata = json.loads(metadata_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {image_path}: {e}")
                    logger.debug(f"Malformed response: {metadata_text[:200]}")
                    if attempt < max_attempts - 1:
                        logger.info(f"Retrying {image_path} due to JSON parse error (attempt {attempt + 2}/{max_attempts})")
                        time.sleep(2)
                        continue
                    return (image_path, None)
                
                # Validate and normalize score
                if 'score' in response_metadata:
                    validated_score = validate_score(response_metadata['score'])
                    if validated_score is None:
                        # Truncate long error messages
                        score_preview = str(response_metadata['score'])[:150] + '...' if len(str(response_metadata['score'])) > 150 else str(response_metadata['score'])
                        logger.warning(f"Invalid score for {image_path}: {score_preview}")
                        if attempt < max_attempts - 1:
                            logger.info(f"Retrying {image_path} due to invalid score (attempt {attempt + 2}/{max_attempts})")
                            time.sleep(2)
                            continue
                        logger.warning(f"Skipping {image_path} after {max_attempts} attempts")
                        return (image_path, None)
                    response_metadata['score'] = str(validated_score)
                else:
                    logger.warning(f"No score field in response for {image_path}")
                    if attempt < max_attempts - 1:
                        logger.info(f"Retrying {image_path} (attempt {attempt + 2}/{max_attempts})")
                        time.sleep(2)
                        continue
                    return (image_path, None)
            
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
            else:
                logger.error(f"Failed to process {image_path}: {response.status_code if response else 'No response'}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying {image_path} (attempt {attempt + 2}/{max_attempts})")
                    time.sleep(2)
                    continue
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
                            cache_dir: Optional[str] = None) -> List[Tuple[str, Optional[Dict]]]:
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
            executor.submit(process_single_image, path, ollama_host_url, model, prompt, dry_run, backup_dir, verify, cache_dir): path 
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
        fieldnames = ['file_path', 'score', 'title', 'description', 'keywords', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for file_path, metadata in results:
            if metadata:
                writer.writerow({
                    'file_path': file_path,
                    'score': metadata.get('score', ''),
                    'title': metadata.get('title', ''),
                    'description': metadata.get('description', ''),
                    'keywords': metadata.get('keywords', ''),
                    'status': 'success'
                })
            else:
                writer.writerow({
                    'file_path': file_path,
                    'score': '',
                    'title': '',
                    'description': '',
                    'keywords': '',
                    'status': 'failed'
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
            'pil_count': pil_count
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
        'pil_count': pil_count
    }


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {stats['total_processed']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and evaluate images with AI.')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process and evaluate images')
    process_parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    process_parser.add_argument('ollama_host_url', type=str, help='Full url of your ollama API endpoint')
    process_parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
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
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Restore images from backups')
    rollback_parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    rollback_parser.add_argument('--backup-dir', type=str, default=None, help='Directory where backups are stored')
    
    args = parser.parse_args()
    
    # Handle rollback command
    if args.command == 'rollback':
        print(f"Rolling back images in: {args.folder_path}")
        if args.backup_dir:
            print(f"Using backup directory: {args.backup_dir}")
        rollback_images(args.folder_path, getattr(args, 'backup_dir', None))
        sys.exit(0)
    
    # Default to process if no command specified (backwards compatibility)
    if args.command is None:
        print("Error: Please specify a command (process or rollback)")
        parser.print_help()
        sys.exit(1)
    
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
        cache_dir=cache_dir
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
