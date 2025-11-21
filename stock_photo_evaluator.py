#!/usr/bin/env python3
"""
Stock Photography Suitability Evaluator

Evaluates images for stock photography requirements including:
- Commercial viability and marketability
- Technical quality standards (resolution, sharpness, noise)
- Composition and subject clarity
- Keyword potential and searchability
- Model/property release requirements
- Common rejection reasons
- Category recommendations
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, ImageStat
import numpy as np
import cv2
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'stock_evaluator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MIN_RESOLUTION_MP = 4.0  # Minimum megapixels for stock
RECOMMENDED_RESOLUTION_MP = 12.0  # Recommended megapixels
MIN_DPI = 300  # Minimum DPI at standard print sizes
DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_WORKERS = 4

# Stock photography evaluation prompt
STOCK_EVALUATION_PROMPT = """You are an expert stock photography reviewer for major agencies such as Shutterstock, Adobe Stock, and Getty Images.

Your task: Evaluate a single image for stock photography suitability using strict professional standards. First, inspect the image itself. Then, incorporate the technical analysis provided.

TECHNICAL ANALYSIS PROVIDED:
{technical_context}

Evaluate the following criteria and provide scores from 0-100 (integers). Use the full range where appropriate, and be slightly conservative: most amateur images should fall below 70 on several dimensions.

1. COMMERCIAL_VIABILITY (0-100)
Consider:
- Market demand and searchability
- Versatility for multiple uses (e.g., ads, blogs, corporate, editorial)
- Relevance to current and enduring themes
- Universal appeal vs very narrow niche
Interpretation:
- Higher = broader, clearer commercial applications and strong demand potential.
- Lower = niche, unclear concept, or weak market demand.

2. TECHNICAL_QUALITY (0-100)
Consider only technical aspects:
- Sharpness and focus accuracy on the main subject
- Exposure and usable dynamic range
- Color accuracy and white balance
- Noise levels, banding, halos, chromatic aberration
- Edge quality, artifacts, and compression issues
Interpretation:
- Higher = meets or exceeds professional stock standards at large sizes.
- Lower = clearly below professional standards or likely technical rejection.

3. COMPOSITION_CLARITY (0-100)
Consider:
- Clear, readable subject
- Simple, uncluttered and non-distracting background
- Professional framing (rule of thirds, leading lines, balance)
- Negative space / copy space suitable for text overlays
- Visual hierarchy and conceptual clarity
Interpretation:
- Higher = clean, purposeful composition ideal for designers.
- Lower = cluttered, confusing, or poorly framed.

4. KEYWORD_POTENTIAL (0-100)
Consider:
- How many distinct, truthful, and commercially relevant concepts/keywords the image naturally supports
- Clarity of subject and context (easy to describe with keywords)
- Mix of literal and conceptual tags (e.g., "mountain, hiking, adventure, freedom")
- Seasonal vs evergreen appeal
Interpretation:
- Higher = many accurate, high-demand keywords and concepts.
- Lower = ambiguous, hard to describe, or few useful concepts.

5. RELEASE_CONCERNS (0-100) - "Release safety"
Consider:
- Identifiable people requiring model releases
- Recognizable private property, interiors, trademarks, logos
- Copyrighted artwork/architecture and design elements
- Any other legal/IP exposure based on typical stock agency policies
Interpretation (IMPORTANT):
- 100 = no releases needed and no recognizable people/property/logos.
- 50 = some potential release issues or ambiguity.
- 0 = multiple serious release problems (likely legal rejection).
If the presence of releases is unknown, assume they are NOT available unless explicitly stated in {technical_context}, and reflect that in the score and ISSUES.

6. REJECTION_RISKS (0-100) - "Rejection safety"
Consider:
- Likely technical flags (noise, blur, artifacts, over-processing)
- Clichéd or over-supplied concepts
- Poor or flat lighting, awkward poses, distracting elements
- Dated styling or short-lived trends
- Any mismatch between image content and stock agency guidelines
Interpretation (IMPORTANT):
- 100 = very low rejection probability across major agencies.
- 50 = borderline; likely to be rejected by at least one major agency.
- 0 = very likely rejection.

7. OVERALL_STOCK_SCORE (0-100)
Overall suitability for stock photography submission. Base this primarily on:
- COMMERCIAL_VIABILITY and TECHNICAL_QUALITY as the core factors,
- Then adjust up or down (up to +/- 10 points) based on COMPOSITION_CLARITY, KEYWORD_POTENTIAL, RELEASE_CONCERNS, and REJECTION_RISKS.
Use the full 0-100 range and do not cluster most images between 70-90.

Map OVERALL_STOCK_SCORE to a categorical recommendation:
- EXCELLENT: >= 85
- GOOD: 70-84
- MARGINAL: 50-69
- REJECT: < 50

OUTPUT FORMAT (STRICT - NO EXTRA TEXT BEFORE OR AFTER):

COMMERCIAL_VIABILITY: [0-100 integer]
TECHNICAL_QUALITY: [0-100 integer]
COMPOSITION_CLARITY: [0-100 integer]
KEYWORD_POTENTIAL: [0-100 integer]
RELEASE_CONCERNS: [0-100 integer]
REJECTION_RISKS: [0-100 integer]
OVERALL_STOCK_SCORE: [0-100 integer]
RECOMMENDATION: [EXCELLENT/GOOD/MARGINAL/REJECT]
PRIMARY_CATEGORY: [one broad category such as Business, Nature, Lifestyle, Technology, Food, Travel, etc.]
SUGGESTED_KEYWORDS: [comma-separated list of 10-15 lowercase English keywords or short phrases, no hashtags, no quotes]
ISSUES: [brief bullet-style text listing the main problems or risks; write "None" only if the image is genuinely strong on all criteria]
STRENGTHS: [brief bullet-style text summarizing the main selling points and best use cases]

Be honest and critical. Remember that major stock agencies routinely reject 70-90% of submissions; err slightly on the side of lower scores and stricter standards rather than generosity."""


@dataclass
class StockEvaluation:
    """Stock photography evaluation result"""
    file_path: str
    commercial_viability: int
    technical_quality: int
    composition_clarity: int
    keyword_potential: int
    release_concerns: int
    rejection_risks: int
    overall_stock_score: int
    recommendation: str  # EXCELLENT/GOOD/MARGINAL/MARGINAL-FIXABLE/REJECT
    primary_category: str
    suggested_keywords: str
    issues: str
    strengths: str
    resolution_mp: float
    dimensions: str
    file_size_mb: float
    technical_notes: str
    fixable_issues: str = ""  # Issues that can be easily corrected
    status: str = "success"
    error_message: str = ""


def analyze_technical_quality(image_path: str) -> Dict:
    """Analyze technical aspects for stock photography standards"""
    try:
        # Open image
        img = Image.open(image_path)
        width, height = img.size
        megapixels = (width * height) / 1_000_000
        
        # Convert to numpy for OpenCV analysis
        img_array = np.array(img)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array
        
        gray = np.array(gray)
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(100, laplacian_var / 10)
        
        # Noise estimation (standard deviation in flat areas)
        noise_estimate = float(np.std(gray.astype(np.float64)))
        
        # Brightness and contrast
        stats = ImageStat.Stat(img)
        brightness = sum(stats.mean) / len(stats.mean)
        contrast = sum(stats.stddev) / len(stats.stddev)
        
        # Histogram analysis for clipping
        if len(img_array.shape) == 3:
            hist_data = []
            for channel in range(img_array.shape[2]):
                hist = np.histogram(img_array[:,:,channel], bins=256, range=(0,256))[0]
                hist_data.append(hist)
            
            # Check for clipping
            highlight_clip = sum(h[-5:].sum() for h in hist_data) / (width * height * len(hist_data))
            shadow_clip = sum(h[:5].sum() for h in hist_data) / (width * height * len(hist_data))
        else:
            hist = np.histogram(gray, bins=256, range=(0,256))[0]
            highlight_clip = hist[-5:].sum() / (width * height)
            shadow_clip = hist[:5].sum() / (width * height)
        
        # DPI check (if available)
        dpi = img.info.get('dpi', (72, 72))
        dpi_value = dpi[0] if isinstance(dpi, tuple) else dpi
        
        # Build technical notes
        notes = []
        
        # Resolution check
        if megapixels < MIN_RESOLUTION_MP:
            notes.append(f"⚠️ LOW RESOLUTION: {megapixels:.1f}MP (min {MIN_RESOLUTION_MP}MP required)")
        elif megapixels < RECOMMENDED_RESOLUTION_MP:
            notes.append(f"⚠️ Below recommended: {megapixels:.1f}MP (recommend {RECOMMENDED_RESOLUTION_MP}MP)")
        else:
            notes.append(f"✓ Good resolution: {megapixels:.1f}MP")
        
        # Sharpness check
        if sharpness_score < 30:
            notes.append(f"⚠️ SOFT/BLURRY: Sharpness {sharpness_score:.1f}/100")
        elif sharpness_score < 60:
            notes.append(f"⚠️ Below optimal sharpness: {sharpness_score:.1f}/100")
        else:
            notes.append(f"✓ Sharp: {sharpness_score:.1f}/100")
        
        # Noise check
        if noise_estimate > 50:
            notes.append(f"⚠️ HIGH NOISE: {noise_estimate:.1f}")
        elif noise_estimate > 30:
            notes.append(f"⚠️ Moderate noise: {noise_estimate:.1f}")
        
        # Clipping check
        if highlight_clip > 0.02:
            notes.append(f"⚠️ Highlight clipping: {highlight_clip*100:.1f}%")
        if shadow_clip > 0.02:
            notes.append(f"⚠️ Shadow clipping: {shadow_clip*100:.1f}%")
        
        # DPI check
        fixable_issues = []
        if dpi_value < MIN_DPI:
            notes.append(f"⚠️ LOW DPI: {dpi_value} (recommend {MIN_DPI})")
            if dpi_value >= 240:  # 240 DPI is easily fixable
                fixable_issues.append(f"DPI: {dpi_value} → {MIN_DPI} (metadata correction)")
        
        # Aspect ratio check
        aspect_ratio = width / height
        common_aspects = [3/2, 4/3, 16/9, 1/1, 2/3, 3/4]
        is_common = any(abs(aspect_ratio - ar) < 0.05 for ar in common_aspects)
        if not is_common:
            notes.append(f"⚠️ Unusual aspect ratio: {aspect_ratio:.2f}")
        
        return {
            'megapixels': megapixels,
            'dimensions': f"{width}x{height}",
            'sharpness': sharpness_score,
            'noise': noise_estimate,
            'brightness': brightness,
            'contrast': contrast,
            'highlight_clip': highlight_clip,
            'shadow_clip': shadow_clip,
            'dpi': dpi_value,
            'aspect_ratio': aspect_ratio,
            'notes': ' | '.join(notes),
            'fixable_issues': fixable_issues
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {e}")
        return {
            'megapixels': 0,
            'dimensions': 'unknown',
            'notes': [f"Error: {str(e)}"]
        }


def create_stock_prompt(base_prompt: str, technical_data: Dict) -> str:
    """Create enhanced prompt with technical context"""
    context_parts = []
    
    # Resolution info
    mp = technical_data.get('megapixels', 0)
    dims = technical_data.get('dimensions', 'unknown')
    context_parts.append(f"Resolution: {mp:.1f} megapixels ({dims})")
    
    # Quality metrics
    sharpness = technical_data.get('sharpness', 0)
    context_parts.append(f"Sharpness: {sharpness:.1f}/100")
    
    noise = technical_data.get('noise', 0)
    if noise > 30:
        context_parts.append(f"Noise level: {noise:.1f} (elevated)")
    
    # Clipping warnings
    highlight_clip = technical_data.get('highlight_clip', 0)
    shadow_clip = technical_data.get('shadow_clip', 0)
    if highlight_clip > 0.02:
        context_parts.append(f"⚠️ Highlight clipping detected ({highlight_clip*100:.1f}%)")
    if shadow_clip > 0.02:
        context_parts.append(f"⚠️ Shadow clipping detected ({shadow_clip*100:.1f}%)")
    
    # DPI
    dpi = technical_data.get('dpi', 72)
    context_parts.append(f"DPI: {dpi}")
    
    # Technical notes
    notes = technical_data.get('notes', [])
    if notes:
        context_parts.append("\nTechnical Issues:")
        context_parts.extend(f"  - {note}" for note in notes)
    
    technical_context = "\n".join(context_parts)
    return base_prompt.format(technical_context=technical_context)


def evaluate_image_for_stock(
    image_path: str,
    api_url: str,
    model: str,
    prompt: str,
    timeout: int = 300
) -> StockEvaluation:
    """Evaluate a single image for stock photography suitability"""
    
    try:
        # Get file info
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        
        # Analyze technical quality
        logger.debug(f"Analyzing technical quality: {image_path}")
        technical_data = analyze_technical_quality(image_path)
        
        # Create enhanced prompt
        enhanced_prompt = create_stock_prompt(prompt, technical_data)
        
        # Encode image to base64
        import base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Make API request
        payload = {
            "model": model,
            "prompt": enhanced_prompt,
            "images": [image_data],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "seed": 42,
                "top_p": 0.9
            }
        }
        
        logger.debug(f"Sending request to {api_url}")
        try:
            response = requests.post(api_url, json=payload, timeout=timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout for {image_path}, retrying...")
            time.sleep(2)
            response = requests.post(api_url, json=payload, timeout=timeout)
            response.raise_for_status()
        
        if response and response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            
            # Parse response
            scores = {}
            for line in response_text.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper().replace(' ', '_')
                    value = value.strip()
                    
                    # Extract scores
                    if key in ['COMMERCIAL_VIABILITY', 'TECHNICAL_QUALITY', 'COMPOSITION_CLARITY',
                              'KEYWORD_POTENTIAL', 'RELEASE_CONCERNS', 'REJECTION_RISKS', 
                              'OVERALL_STOCK_SCORE']:
                        try:
                            scores[key.lower()] = int(value)
                        except ValueError:
                            scores[key.lower()] = 0
                    elif key == 'RECOMMENDATION':
                        scores['recommendation'] = value
                    elif key == 'PRIMARY_CATEGORY':
                        scores['primary_category'] = value
                    elif key == 'SUGGESTED_KEYWORDS':
                        scores['suggested_keywords'] = value
                    elif key == 'ISSUES':
                        scores['issues'] = value
                    elif key == 'STRENGTHS':
                        scores['strengths'] = value
            
            # Build technical notes string
            technical_notes = "; ".join(technical_data.get('notes', []))
            
            # Analyze fixable issues and adjust recommendation
            fixable_list = technical_data.get('fixable_issues', [])
            fixable_issues_str = '; '.join(fixable_list) if fixable_list else 'None'
            
            recommendation = scores.get('recommendation', 'UNKNOWN')
            overall_stock_score = scores.get('overall_stock_score', 0)
            issues = scores.get('issues', '')
            
            # If recommendation is MARGINAL and there are only fixable issues, upgrade to MARGINAL-FIXABLE
            if recommendation == "MARGINAL" and fixable_list:
                # Check if main issues are fixable (DPI, minor metadata)
                issues_lower = issues.lower()
                has_only_fixable = (
                    fixable_list and  # Has fixable issues
                    overall_stock_score >= 45 and  # Reasonable base score
                    'resolution' not in issues_lower and  # No resolution problems
                    'extreme' not in issues_lower and  # No extreme quality issues
                    'severe' not in issues_lower
                )
                if has_only_fixable:
                    recommendation = "MARGINAL-FIXABLE"
            
            return StockEvaluation(
                file_path=image_path,
                commercial_viability=scores.get('commercial_viability', 0),
                technical_quality=scores.get('technical_quality', 0),
                composition_clarity=scores.get('composition_clarity', 0),
                keyword_potential=scores.get('keyword_potential', 0),
                release_concerns=scores.get('release_concerns', 0),
                rejection_risks=scores.get('rejection_risks', 0),
                overall_stock_score=overall_stock_score,
                recommendation=recommendation,
                primary_category=scores.get('primary_category', 'Unknown'),
                suggested_keywords=scores.get('suggested_keywords', ''),
                issues=issues,
                strengths=scores.get('strengths', ''),
                resolution_mp=technical_data.get('megapixels', 0),
                dimensions=technical_data.get('dimensions', 'unknown'),
                file_size_mb=file_size_mb,
                technical_notes=technical_notes,
                fixable_issues=fixable_issues_str,
                status="success"
            )
        else:
            # Handle unsuccessful response
            error_msg = f"API request failed with status code {response.status_code if response else 'unknown'}"
            return StockEvaluation(
                file_path=image_path,
                commercial_viability=0,
                technical_quality=0,
                composition_clarity=0,
                keyword_potential=0,
                release_concerns=0,
                rejection_risks=0,
                overall_stock_score=0,
                recommendation="ERROR",
                primary_category="",
                suggested_keywords="",
                issues=error_msg,
                strengths="",
                resolution_mp=technical_data.get('megapixels', 0),
                dimensions=technical_data.get('dimensions', 'unknown'),
                file_size_mb=file_size_mb,
                technical_notes="; ".join(technical_data.get('notes', [])),
                fixable_issues="None",
                status="error",
                error_message=error_msg
            )
        
    except Exception as e:
        logger.error(f"Error evaluating {image_path}: {e}")
        return StockEvaluation(
            file_path=image_path,
            commercial_viability=0,
            technical_quality=0,
            composition_clarity=0,
            keyword_potential=0,
            release_concerns=0,
            rejection_risks=0,
            overall_stock_score=0,
            recommendation="ERROR",
            primary_category="",
            suggested_keywords="",
            issues=str(e),
            strengths="",
            resolution_mp=0,
            dimensions="",
            file_size_mb=0,
            technical_notes="",
            fixable_issues="None",
            status="error",
            error_message=str(e)
        )


def find_images(directory: str, extensions: Optional[List[str]] = None) -> List[str]:
    """Find all images in directory recursively"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Skip files with 'original' in the name
            if 'original' in file.lower():
                continue
            
            if any(file.lower().endswith(ext) for ext in extensions):
                images.append(os.path.join(root, file))
    
    return images


def save_results_to_csv(results: List[StockEvaluation], output_path: str):
    """Save evaluation results to CSV"""
    fieldnames = [
        'file_path', 'overall_stock_score', 'recommendation', 
        'commercial_viability', 'technical_quality', 'composition_clarity',
        'keyword_potential', 'release_concerns', 'rejection_risks',
        'primary_category', 'resolution_mp', 'dimensions', 'file_size_mb',
        'suggested_keywords', 'strengths', 'issues', 'fixable_issues',
        'technical_notes', 'status', 'error_message'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    
    logger.info(f"Results saved to: {output_path}")


def print_summary(results: List[StockEvaluation]):
    """Print summary statistics"""
    if not results:
        print("No results to summarize")
        return
    
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "error"]
    
    if successful:
        scores = [r.overall_stock_score for r in successful]
        avg_score = sum(scores) / len(scores)
        
        # Count recommendations
        excellent = sum(1 for r in successful if r.recommendation == "EXCELLENT")
        good = sum(1 for r in successful if r.recommendation == "GOOD")
        marginal = sum(1 for r in successful if r.recommendation == "MARGINAL")
        marginal_fixable = sum(1 for r in successful if r.recommendation == "MARGINAL-FIXABLE")
        reject = sum(1 for r in successful if r.recommendation == "REJECT")
        
        # Resolution check
        below_min = sum(1 for r in successful if r.resolution_mp < MIN_RESOLUTION_MP)
        below_rec = sum(1 for r in successful if MIN_RESOLUTION_MP <= r.resolution_mp < RECOMMENDED_RESOLUTION_MP)
        
        print("\n" + "="*80)
        print(f"{Fore.CYAN}{Style.BRIGHT}STOCK PHOTOGRAPHY EVALUATION SUMMARY{Style.RESET_ALL}")
        print("="*80)
        print(f"\nTotal images: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        print(f"\n{Style.BRIGHT}AVERAGE STOCK SCORE: {avg_score:.1f}/100{Style.RESET_ALL}")
        
        print(f"\n{Style.BRIGHT}RECOMMENDATIONS:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}EXCELLENT: {excellent}{Style.RESET_ALL} ({excellent/len(successful)*100:.1f}%)")
        print(f"  {Fore.BLUE}GOOD: {good}{Style.RESET_ALL} ({good/len(successful)*100:.1f}%)")
        print(f"  {Fore.CYAN}MARGINAL-FIXABLE: {marginal_fixable}{Style.RESET_ALL} ({marginal_fixable/len(successful)*100:.1f}%) - Easy fixes available")
        print(f"  {Fore.YELLOW}MARGINAL: {marginal}{Style.RESET_ALL} ({marginal/len(successful)*100:.1f}%)")
        print(f"  {Fore.RED}REJECT: {reject}{Style.RESET_ALL} ({reject/len(successful)*100:.1f}%)")
        
        print(f"\n{Style.BRIGHT}RESOLUTION STATUS:{Style.RESET_ALL}")
        print(f"  {Fore.RED}Below minimum ({MIN_RESOLUTION_MP}MP): {below_min}{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Below recommended ({RECOMMENDED_RESOLUTION_MP}MP): {below_rec}{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}Meets recommended: {len(successful)-below_min-below_rec}{Style.RESET_ALL}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate images for stock photography suitability',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('directory', help='Directory containing images to evaluate')
    parser.add_argument('api_url', help='Ollama API URL (e.g., http://localhost:11434/api/generate)')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'Model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help=f'Number of parallel workers (default: {DEFAULT_WORKERS})')
    parser.add_argument('--csv', help='Output CSV file (default: auto-generated)')
    parser.add_argument('--min-score', type=int, help='Only show results with stock score >= this value')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png'], help='Image file extensions to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"{Fore.RED}Error: Directory not found: {args.directory}{Style.RESET_ALL}")
        sys.exit(1)
    
    # Find images
    print(f"\n{Fore.CYAN}Scanning for images...{Style.RESET_ALL}")
    images = find_images(args.directory, args.extensions)
    
    if not images:
        print(f"{Fore.RED}No images found in {args.directory}{Style.RESET_ALL}")
        sys.exit(1)
    
    print(f"Found {Fore.GREEN}{len(images)}{Style.RESET_ALL} images")
    
    # Process images
    results = []
    start_time = time.time()
    
    print(f"\n{Fore.CYAN}Evaluating images for stock photography...{Style.RESET_ALL}")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_image = {
            executor.submit(
                evaluate_image_for_stock,
                image,
                args.api_url,
                args.model,
                STOCK_EVALUATION_PROMPT
            ): image
            for image in images
        }
        
        with tqdm(total=len(images), desc="Processing") as pbar:
            for future in as_completed(future_to_image):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Filter by min score if specified
    if args.min_score:
        results = [r for r in results if r.overall_stock_score >= args.min_score]
    
    # Save results
    if args.csv:
        output_csv = args.csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"stock_evaluation_{timestamp}.csv"
    
    save_results_to_csv(results, output_csv)
    
    # Print summary
    print_summary(results)
    
    print(f"\n{Fore.CYAN}Processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes){Style.RESET_ALL}")
    print(f"{Fore.CYAN}Time per image: {elapsed_time/len(images):.2f} seconds{Style.RESET_ALL}")
    print(f"\nResults saved to: {Fore.GREEN}{output_csv}{Style.RESET_ALL}")


if __name__ == '__main__':
    main()
