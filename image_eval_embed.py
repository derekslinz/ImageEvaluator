import argparse
import base64
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json  # Import json for parsing the response
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import piexif
import piexif.helper
import requests
from PIL import Image
from colorama import Fore
from pydantic import BaseModel, ConfigDict, field_validator
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def embed_metadata_exiftool(image_path: str, metadata: Dict) -> bool:
    """
    Embed metadata into RAW/TIFF files using exiftool.
    Returns True if successful, False otherwise.
    """
    try:
        # Create backup first
        backup_image_path = f"{os.path.splitext(image_path)[0]}.original{os.path.splitext(image_path)[1]}"
        
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
            print(f"Embedding score: {metadata.get('score', '')}")
            print(f"Embedding Title: {metadata.get('title', '')}")
            print(f"Embedding Description: {metadata.get('description', '')}")
            print(f"Embedding Keywords: {metadata.get('keywords', '')}")
            print(Fore.GREEN + f"Metadata successfully embedded in {image_path} using exiftool" + Fore.RESET)
            return True
        else:
            logger.error(f"exiftool failed for {image_path}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error embedding metadata with exiftool in {image_path}: {e}")
        return False


def embed_metadata(image_path: str, metadata: Dict) -> bool:
    """
    Embed metadata into image file. Returns True if successful, False if skipped.
    Uses exiftool for RAW/TIFF files, PIL for JPEG/PNG.
    """
    try:
        # Check if file format is RAW/TIFF
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.dng', '.nef', '.tif', '.tiff']:
            return embed_metadata_exiftool(image_path, metadata)
        
        # Open the image to access its EXIF data
        with Image.open(image_path) as img:
            exif_data = img.info.get("exif", b"")
            if exif_data:
                exif_dict = piexif.load(exif_data)
            else:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}  # Create a new EXIF structure

            user_comment = piexif.helper.UserComment.dump(str(metadata.get('score', '')))
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
            print(f"Embedding score in User Comment: {user_comment}")

            # Embedding Title (UTF-16LE with BOM for Windows compatibility)
            title = metadata.get('title', '').encode('utf-16le') + b'\x00\x00'
            exif_dict["0th"][piexif.ImageIFD.XPTitle] = title
            print(f"Embedding Title: {metadata.get('title', '')}")

            # Embedding Description
            description = metadata.get('description', '').encode('utf-16le') + b'\x00\x00'
            exif_dict["0th"][piexif.ImageIFD.XPComment] = description
            print(f"Embedding Description: {metadata.get('description', '')}")

            # Embedding Keywords
            keywords = metadata.get('keywords', '').encode('utf-16le') + b'\x00\x00'
            exif_dict["0th"][piexif.ImageIFD.XPKeywords] = keywords
            print(f"Embedding Keywords: {metadata.get('keywords', '')}")

            # Prepare Exif data with sanitized strings
            exif_bytes = piexif.dump(exif_dict)

            # Backup original image by appending .original suffix
            backup_image_path = f"{os.path.splitext(image_path)[0]}.original{os.path.splitext(image_path)[1]}"
            if os.path.exists(image_path):  # Ensure the file exists before renaming
                os.rename(image_path, backup_image_path)

            # Save with new metadata
            img.save(image_path, exif=exif_bytes)
            print(Fore.GREEN + f"Metadata successfully embedded in {image_path}" + Fore.RESET)
            return True

    except Exception as e:
        logger.error(f"Error embedding metadata in {image_path}: {e}")
        return False


def collect_images(folder_path: str) -> List[str]:
    """Collect all image paths to process."""
    image_paths = []
    
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Skip files with '.original' in the name (before the extension)
            if '.original.' in filename:
                continue

            if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG','NEF','nef','TIF','tif','TIFF','tiff','DNG','dng')):
                image_path = os.path.join(root, filename)
                
                # Check if UserComment exists
                if has_user_comment(image_path):
                    continue
                    
                image_paths.append(image_path)
    
    return image_paths


def process_single_image(image_path: str, ollama_host_url: str) -> Tuple[str, Optional[Dict]]:
    """Process a single image and return result."""
    headers = {'Content-Type': 'application/json'}
    
    try:
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        payload = {
            "model": "qwen3-vl:8b",
            "stream": False,
            "images": [encoded_image],
            "prompt": "You are a discerning photography critic. Evaluate this photograph and provide a numerical score from 1-100 based on:\n\n1. Technical quality (exposure, focus, color balance, composition, aspect ratio)\n2. Creativity (unique perspective, mood, subject matter, emotional impact)\n3. Aesthetic appeal (visual appeal to general audience) - weighted double\n\nReturn ONLY valid JSON with these exact fields:\n- score: integer from 1 to 100 (just the number, no explanation)\n- title: descriptive title, maximum 60 characters\n- description: image description, maximum 200 characters\n- keywords: up to 12 relevant keywords, comma separated, no hashtags\n\nExample format:\n{\"score\": \"75\", \"title\": \"Sunset Over Mountains\", \"description\": \"Vibrant sunset casting golden light over mountain peaks with dramatic cloud formations.\", \"keywords\": \"sunset, mountains, landscape, dramatic, golden hour, nature, scenic, clouds, peaks, outdoor, wilderness, photography\"}",
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
            }
        }

        response = requests.post(ollama_host_url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()

            # Extract and parse the metadata - check both 'response' and 'thinking' fields
            metadata_text = response_data.get('response') or response_data.get('thinking', '')
            if not metadata_text:
                logger.error(f"No response or thinking field found for {image_path}")
                return (image_path, None)
                
            response_metadata = json.loads(metadata_text)

            # Embed metadata
            try:
                embed_metadata(image_path, response_metadata)
                return (image_path, response_metadata)
            except Exception as e:
                logger.error(f"Error embedding metadata for {image_path}: {e}")
                return (image_path, None)
        else:
            logger.error(f"Failed to process {image_path}: {response.status_code}")
            return (image_path, None)
            
    except Exception as e:
        logger.error(f"Request failed for {image_path}: {e}")
        return (image_path, None)


def process_images_in_folder(folder_path: str, ollama_host_url: str, max_workers: int = 4) -> List[Tuple[str, Optional[Dict]]]:
    """Process images with parallel execution and progress bar."""
    # Collect all images to process
    image_paths = collect_images(folder_path)
    
    if not image_paths:
        logger.warning("No images found to process")
        return []
    
    results = []
    
    # Process images in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(process_single_image, path, ollama_host_url): path 
            for path in image_paths
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
            for future in as_completed(future_to_path):
                result = future.result()
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


def calculate_statistics(results: List[Tuple[str, Optional[Dict]]]) -> Dict:
    """Calculate statistics from processing results."""
    scores = []
    for _, metadata in results:
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
            'score_distribution': {}
        }
    
    # Calculate score distribution (bins of 10)
    distribution = {}
    for i in range(0, 100, 10):
        bin_label = f"{i}-{i+9}"
        distribution[bin_label] = sum(1 for s in scores if i <= s < i+10)
    distribution["90-100"] = sum(1 for s in scores if 90 <= s <= 100)
    
    return {
        'total_processed': len(results),
        'successful': len(scores),
        'failed': len(results) - len(scores),
        'avg_score': sum(scores) / len(scores) if scores else 0,
        'min_score': min(scores) if scores else 0,
        'max_score': max(scores) if scores else 0,
        'score_distribution': distribution
    }


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {stats['total_processed']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    
    if stats['successful'] > 0:
        print(f"\n{'='*60}")
        print(f"SCORE STATISTICS")
        print(f"{'='*60}")
        print(f"Average score: {stats['avg_score']:.1f}")
        print(f"Minimum score: {stats['min_score']}")
        print(f"Maximum score: {stats['max_score']}")
        
        print(f"\n{'='*60}")
        print(f"SCORE DISTRIBUTION")
        print(f"{'='*60}")
        for bin_range, count in sorted(stats['score_distribution'].items()):
            bar = '█' * count
            print(f"{bin_range:>8}: {bar} ({count})")
    
    print(f"\n{'='*60}")
    print(f"Note: JPEG/PNG use PIL, RAW/TIFF use exiftool for metadata embedding")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images in a specified folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    parser.add_argument('ollama_host_url', type=str, help='Full url of your ollama API endpoint')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--csv', type=str, default=None, help='Path to save CSV report (default: auto-generated)')
    args = parser.parse_args()

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

    print(f"\nProcessing images from: {args.folder_path}")
    print(f"Using {args.workers} parallel workers\n")
    
    # Process images
    results = process_images_in_folder(args.folder_path, args.ollama_host_url, max_workers=args.workers)
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Print statistics
    print_statistics(stats)
    
    # Save to CSV
    if args.csv:
        csv_path = args.csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"image_evaluation_results_{timestamp}.csv"
    
    save_results_to_csv(results, csv_path)
    print(f"Results saved to: {csv_path}")
