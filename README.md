# Image Evaluator

## AI-Powered Image Evaluation Suite

A toolkit for evaluating images using multiple scoring backends:
1. **PyIQA (default)** - State-of-the-art perceptual image quality metrics (CLIP-IQA+, MUSIQ, MANIQA, etc.)
2. **Ollama (LLM)** - AI vision models for artistic merit evaluation with EXIF metadata embedding
3. **Stock Photo Evaluator** - Commercial stock photography suitability assessment

### Features

#### Image Evaluator (image_eval_embed.py)

**Scoring Backends:**
- **PyIQA (default)**: Fast GPU-accelerated scoring using state-of-the-art metrics
  - CLIP-IQA+ (ViT-L/14): Best balance of speed and accuracy
  - MUSIQ: Multi-scale image quality transformer
  - MANIQA: Multi-dimension attention network
  - 40+ additional metrics available
- **Ollama**: Full LLM-based evaluation with detailed feedback
  - Multi-criteria scoring (technical, composition, lighting, creativity)
  - Title, description, and keyword generation
  - Ensemble scoring for consistency

**Core Functionality:**
- Multi-format support: JPEG, PNG, DNG, NEF, TIF/TIFF
- AI evaluation with intelligent scoring
- EXIF metadata embedding: score, title, description, keywords
- Recursive directory processing
- Automatic backups before modification

**Advanced Features:**
- Multi-criteria scoring: technical, composition, lighting, creativity
- Ensemble scoring: multiple passes with median aggregation for consistency
- EXIF metadata extraction: ISO, aperture, shutter speed, focal length
- Technical analysis: histogram, color cast detection, sharpness measurement
- Caching system: MD5-based caching for repeat evaluations

**Performance:**
- Parallel processing with configurable workers
- Real-time progress tracking
- Comprehensive statistics and score distribution
- CSV export with detailed breakdowns

**Quality & Safety:**
- Score validation and extraction from malformed responses
- Exponential backoff retry logic with permanent error detection
- Image validation and corruption detection
- Optional metadata verification
- Backup directory support
- Full rollback capability
- Camera/ISO-agnostic noise estimation using MAD (Median Absolute Deviation)
- Scale-normalized sharpness metric for consistent measurements across resolutions
- Post-processing potential score (0-100) derived from measured sharpness/clipping so you can prioritize images worth retouching
- Context-aware technical evaluation with 10 photography profiles
- Status tracking for technical analysis (success/error detection)

**Flexibility:**
- Configurable models (default: qwen3-vl:8b)
- Custom evaluation prompts
- Selective processing by file type, score, metadata
- Dry-run mode for previewing

#### Stock Photo Evaluator (stock_photo_evaluator.py)

**Evaluation Criteria:**
- Commercial viability and market demand
- Technical quality standards (resolution, sharpness, noise, DPI)
- Composition clarity and copy space
- Keyword potential and searchability
- Model/property release requirements
- Common rejection risk assessment

**Technical Analysis:**
- Resolution validation (4MP minimum, 12MP recommended)
- DPI checking (300 DPI standard)
- Scale-normalized sharpness measurement via Laplacian variance
- Camera/ISO-agnostic noise estimation with 0-100 severity scoring
- Histogram clipping detection (per-channel max for accurate reporting)
- Color cast detection with dominant channel identification
- Aspect ratio validation
- Brightness and contrast measurements

**Recommendations:**
- EXCELLENT: Ready for immediate submission
- GOOD: Strong candidate with minor considerations
- MARGINAL-FIXABLE: Needs easy corrections (e.g., DPI metadata) - validated against technical measurements
- MARGINAL: Needs improvement before submission
- REJECT: Does not meet stock standards

**Smart Classification:**
- MARGINAL-FIXABLE uses robust keyword detection (checks for unfixable issues like blur, noise, composition)
- Validates fixability against actual technical measurements (sharpness ≥30, noise ≤65, resolution ≥4MP)
- Minimum score threshold of 45 for upgrade consideration

**Output:**
- Detailed CSV with all scores and analysis
- Suggested keywords for stock submission
- Issues identification with fixable vs. critical problems
- Strengths highlighting
- Category recommendations

### Requirements

- Python 3.8 or higher
- `exiftool` for RAW file metadata embedding
- **For PyIQA backend (default):**
  - `torch` - PyTorch for GPU acceleration
  - `pyiqa` - Image quality assessment metrics
- **For Ollama backend:**
  - Ollama server with vision model (recommended: `qwen3-vl:8b`)
- Required Python libraries (see `requirements.txt`):
  - `Pillow` - Image processing
  - `numpy` - Numerical operations
  - `opencv-python` - Technical analysis
  - `requests` - API calls
  - `piexif` - EXIF manipulation
  - `pydantic` - Data validation
  - `colorama` - Colored terminal output
  - `tqdm` - Progress bars
  - `rawpy` - RAW (NEF/CR2/ARW) decoding for technical analysis

Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

#### PyIQA Scoring (Default - Fastest)

```bash
# Basic usage with CLIP-IQA+ (default)
python image_eval_embed.py process /path/to/images

# Use a different PyIQA metric
python image_eval_embed.py process /path/to/images --pyiqa-model musiq

# Adjust batch size for memory constraints
python image_eval_embed.py process /path/to/images --pyiqa-batch-size 8
```

#### Ollama LLM Scoring (Most Detailed)

1. **Start Ollama server:**
   ```bash
   ollama serve
   ollama pull qwen3-vl:8b
   ```

2. **Evaluate images:**
   ```bash
   python image_eval_embed.py process /path/to/images --score-engine ollama
   ```

3. **View results:**
   - Console displays statistics and score distribution (warning count now reported)
   - CSV report saved with timestamp and includes the measured technical metrics, `post_process_potential`, and any automated warnings/filter cues
   - Metadata embedded in image EXIF

#### Stock Photo Evaluator

1. **Evaluate for stock suitability:**
   ```bash
   python stock_photo_evaluator.py /path/to/images http://localhost:11434/api/generate --model qwen3-vl:8b --workers 4 --csv stock_results.csv
   ```

2. **Review results:**
   - Summary shows recommendation breakdown
   - CSV contains detailed scores and issues
   - Fixable issues identified separately

### Usage

#### Process Command

Basic syntax:
```bash
python image_eval_embed.py process <folder_path> [OPTIONS]
```

Quick start with defaults (uses PyIQA with CLIP-IQA+):
```bash
python image_eval_embed.py
```
Running without arguments assumes the `process` command, uses your current working directory (or `IMAGE_EVAL_DEFAULT_FOLDER`) as the image source.

**Environment overrides for defaults:**
- `IMAGE_EVAL_DEFAULT_FOLDER` - absolute path to use when a folder is not provided
- `IMAGE_EVAL_OLLAMA_URL` - Ollama endpoint to use when omitted (for Ollama backend)
- `IMAGE_EVAL_WORKERS` - worker count used when `--workers` is not supplied

**Arguments:**
- `folder_path`: Directory containing images (processes recursively); defaults to current working directory or `IMAGE_EVAL_DEFAULT_FOLDER` when omitted
- `ollama_url`: Full Ollama API endpoint (only needed for `--score-engine ollama`)

**Scoring Engine Options:**
```bash
--score-engine ENGINE    # Choose: pyiqa (default) or ollama
```

**PyIQA Options:**
```bash
--pyiqa-model NAME       # PyIQA metric (default: clipiqa+_vitl14_512)
--pyiqa-device DEVICE    # Device: cuda:0, cpu (default: auto-detect)
--pyiqa-score-shift N    # Score adjustment (default: model-specific)
--pyiqa-scale-factor N   # Raw score multiplier (default: auto-detect)
--pyiqa-batch-size N     # Images per batch (default: 4)
```

**Ollama Options:**
```bash
--model NAME             # Ollama model (default: qwen3-vl:8b)
--ensemble N             # Evaluation passes for consistency (default: 1)
--prompt-file FILE       # Custom prompt template file
--context NAME           # Manual context override (e.g., landscape, portrait)
--no-context-classification  # Skip auto context classification
```

**General Options:**
```bash
--workers N              # Parallel workers (default: 4)
--csv PATH               # Custom CSV output path
--skip-existing          # Skip images with metadata (default: True)
--no-skip-existing       # Process all images
--min-score N            # Only save results >= N
--file-types EXT,EXT     # Filter by extensions (e.g., jpg,png)
--dry-run                # Preview without changes
--backup-dir DIR         # Store backups separately
--verify                 # Verify metadata after embedding
--cache                  # Enable API response caching
--cache-dir DIR          # Cache directory location
--verbose, -v            # Enable debug output
```

**Examples:**
```bash
# PyIQA with CLIP-IQA+ (fastest, default)
python image_eval_embed.py process /photos

# PyIQA with MUSIQ metric
python image_eval_embed.py process /photos --pyiqa-model musiq

# Ollama LLM with ensemble scoring
python image_eval_embed.py process /photos --score-engine ollama \
  --ensemble 3 \
  --backup-dir /backups/photos

# Only top-quality JPEGs
python image_eval_embed.py process /photos \
  --file-types jpg,jpeg \
  --min-score 80

# Preview without changes
python image_eval_embed.py process /photos --dry-run

# Reprocess everything with verification
python image_eval_embed.py process /photos \
  --no-skip-existing \
  --verify
```

#### Rollback Command

Restore images from backups:
```bash
python image_eval_embed.py rollback <folder_path> [--backup-dir DIR]
```

**Examples:**
```bash
# Restore from default backups (same directory)
python image_eval_embed.py rollback /photos

# Restore from separate backup directory
python image_eval_embed.py rollback /photos --backup-dir /backups/photos
```

#### Stats Command

Print statistics for an existing CSV report:
```bash
python image_eval_embed.py stats /path/to/results.csv
```

```bash
python stock_photo_evaluator.py <directory> <api_url> [OPTIONS]
```

**Arguments:**
- `directory`: Directory containing images to evaluate
- `api_url`: Ollama API endpoint (e.g., `http://localhost:11434/api/generate`)

**Options:**
```bash
--model NAME             # Ollama model (default: qwen3-vl:8b)
--workers N              # Parallel workers (default: 4)
--csv PATH               # Output CSV file (default: auto-generated)
--min-score N            # Only show results with score >= N
--extensions EXT EXT     # File extensions to process (default: jpg jpeg png)
-v, --verbose            # Verbose output
```

By default, the stock evaluator enforces 300 DPI by calling `exiftool -overwrite_original -XResolution=300 -YResolution=300 -ResolutionUnit=inches` before analyzing each image, so `exiftool` must be installed and on your `PATH`.

**Examples:**
```bash
# Basic evaluation
python stock_photo_evaluator.py /photos http://localhost:11434/api/generate

# Custom output with filtering
python stock_photo_evaluator.py /photos http://localhost:11434/api/generate \
  --csv my_stock_eval.csv \
  --min-score 60

# High-performance with specific model
python stock_photo_evaluator.py /photos http://localhost:11434/api/generate \
  --model qwen3-vl:8b \
  --workers 8 \
  --verbose
```

#### Context-Aware Evaluation

The image evaluator now includes 10 specialized photography context profiles:

1. **stock_product** - Clean product/catalog with neutral background (strictest standards)
2. **macro_food** - Close-up food and macro photography with fine detail requirements
3. **portrait_neutral** - Standard portrait with controlled lighting
4. **portrait_highkey** - Bright, airy portrait with intentional highlight blowout tolerance
5. **landscape** - Nature, outdoor scenery, wide vistas
6. **street_documentary** - Candid, street photography, documentary style (flexible noise/clipping)
7. **sports_action** - Sports, action, wildlife, fast motion
8. **concert_night** - Night photography, concerts, low-light (most forgiving noise/clipping)
9. **architecture_realestate** - Buildings, interiors, real estate
10. **fineart_creative** - Experimental, abstract, ICM (most flexible technical standards)

Each profile has customized thresholds for:
- Sharpness requirements (critical/soft boundaries)
- Clipping tolerance (highlights/shadows warning levels)
- Color cast sensitivity and penalties
- Noise acceptability ranges
- Brightness expectations
- Post-processing potential scoring

Context is automatically classified or can be manually overridden with `--context` flag.

#### Example Stock Evaluation Summary
```text
===============================================================================
STOCK PHOTOGRAPHY EVALUATION SUMMARY
================================================================================

Total images: 245
Successful: 236
Failed: 9

AVERAGE STOCK SCORE: 51.3/100

RECOMMENDATIONS:
  EXCELLENT: 0 (0.0%)
  GOOD: 14 (5.9%)
  MARGINAL-FIXABLE: 132 (55.9%) - Easy fixes available (validated against technical metrics)
  MARGINAL: 50 (21.2%)
  REJECT: 35 (14.8%)

RESOLUTION STATUS:
  Below minimum (4.0MP): 3
  Below recommended (12.0MP): 17
  Meets recommended: 216

================================================================================

Processing time: 5554.6 seconds (92.6 minutes)
Time per image: 22.67 seconds
```

#### Technical analysis helper

For a quick single-file check, run:
```bash
python technical_analysis.py /path/to/image.NEF
```
Pass `--format json` to get the raw metric dictionary, or let the default text output surface sharpness, noise, clipping, DPI, and the human-readable notes the stock prompt uses.

#### Upload Results to PostgreSQL

Once you have a CSV from either evaluator, use `csv_to_postgres.py` to push it into Postgres:
```bash
python csv_to_postgres.py path/to/results.csv --db-url postgres://localhost:5432/photos
```
The script normalizes the header, creates `image_evaluation_results` (or your chosen table) with text columns, and inserts all rows; pass `--truncate` if you want to drop existing rows first. Ensure `psycopg2-binary` is installed along with the other requirements.
It captures the new `post_process_potential` field so you can query for shots with high improvement potential.

#### Example Output: model qwen3-vl:8b running on ollama on an Apple M4 Max with --ensemble=3 (3 passes per image). Each image is ~50 megapixels. Running on gemma3:4b was faster, but the results were all clustered around the same score. It's notably faster on a RTX 3090, but still time consuming with 3 passes. Smaller images process faster, obviously
```bash
 $ python image_eval_embed.py process /Volumes/NVMe/Lightroom Test Export/   http://localhost:11434/api/generate --no-skip-existing --csv "Lightroom Test Export.csv"   --model qwen3-vl:8b --ensemble 3 --workers 8
2025-11-21 03:27:27,264 - INFO - Logging to file: image_evaluator_20251121_032727.log
2025-11-21 03:27:27,264 - INFO - Starting image evaluation - log file: image_evaluator_20251121_032727.log

Processing images from: /Volumes/NVMe/Lightroom Test Export/
Model: qwen3-vl:8b
Workers: 8
Skip existing: False
Ensemble mode: 3 evaluation passes per image

2025-11-21 03:27:27,264 - INFO - Processing configuration: workers=8, model=qwen3-vl:8b, cache=disabled
2025-11-21 03:27:27,264 - INFO - Starting processing of images in /Volumes/NVMe/Lightroom Test Export/
Processing images:   0%|                                                                                                                                                                                                      | 0/245 [00:00<?, ?img/s]2025-11-21 03:28:01,304 - INFO - Performing 3-pass ensemble evaluation for /Volumes/NVMe/Lightroom Test Export/DSC_8869.jpg
2025-11-21 03:28:01,560 - INFO - Performing 3-pass ensemble evaluation for /Volumes/NVMe/Lightroom Test Export/Fireworks- December 31, 2019 -787.jpg
2025-11-21 03:28:01,607 - INFO - Performing 3-pass ensemble evaluation for /Volumes/NVMe/Lightroom Test Export/Copenhagen .Next 2019-255-HDR.jpg
2025-11-21 03:28:01,701 - INFO - Performing 3-pass ensemble evaluation for /Volumes/NVMe/Lightroom Test Export/Reflectie- March 13, 2020 -08-Pano.jpg
2025-11-21 03:28:01,704 - INFO - Performing 3-pass ensemble evaluation for /Volumes/NVMe/Lightroom Test Export/Montreux with Aisiri-74-HDR-2.jpg
2025-11-21 03:28:01,757 - INFO - Performing 3-pass ensemble evaluation for /Volumes/NVMe/Lightroom Test Export/_DSC5524.jpg
2025-11-21 03:28:01,818 - INFO - Performing 3-pass ensemble evaluation for /Volumes/NVMe/Lightroom Test Export/Venice Afternoon-7241-HDR.jpg
2025-11-21 03:28:01,979 - INFO - Performing 3-pass ensemble evaluation for /Volumes/NVMe/Lightroom Test Export/DSC_2163-Enhanced-NR.jpg
2025-11-21 03:31:10,544 - INFO - Ensemble std dev for /Volumes/NVMe/Lightroom Test Export/DSC_8869.jpg: 1.25 (scores: [89, 88, 91])
Embedding score: 89
Embedding Title: Spiral Staircase Perspective
Embedding Description: Masterful black-and-white spiral staircase shot with precise exposure and sharp focus.
Embedding Keywords: spiral staircase, architectural photography, black and white, perspective, symmetry, geometry, monochrome, interior, stairwell, design, composition, photography
Metadata successfully embedded in /Volumes/NVMe/Lightroom Test Export/DSC_8869.jpg
2025-11-21 03:31:10,671 - INFO - Successfully processed /Volumes/NVMe/Lightroom Test Export/DSC_8869.jpg with score 89
Processing images:   0%|▊                                                                                                                                                                                         | 1/245 [03:09<12:51:05, 189.61s/img]2025-11-21 03:31:11,759 - INFO - Performing 3-pass ensemble evaluation for /Volumes/NVMe/Lightroom Test Export/DSC_8667.jpg
...snip 244 more images...
2025-11-21 03:31:22,071 - INFO - Ensemble std dev for /Volumes/NVMe/Lightroom Test Export/Copenhagen .Next 2019-255-HDR.jpg: 0.00 (scores: [80, 80, 80]) 245/245 [2:30:37<00:00, 36.89s/img]
2025-11-21 05:58:38,593 - INFO - Completed processing 245 images in 9071.33 seconds

============================================================
PROCESSING SUMMARY
============================================================
Total images processed: 245
Successful: 245
Failed: 0
RAW formats (exiftool): 0
Standard formats (PIL): 245

============================================================
TIMING STATISTICS
============================================================
Total processing time: 9071.33 seconds (151.19 minutes)
Time per image: 37.03 seconds
Time per successful image: 37.03 seconds

============================================================
SCORE STATISTICS
============================================================
Average score: 80.20
Median score: 81.00
Standard deviation: 5.36
Range: 58 - 91
Quartiles (Q1/Q3): 77 / 83

============================================================
SCORE DISTRIBUTION
============================================================
     0-4:  (0)
   10-14:  (0)
   15-19:  (0)
   20-24:  (0)
   25-29:  (0)
   30-34:  (0)
   35-39:  (0)
   40-44:  (0)
   45-49:  (0)
     5-9:  (0)
   50-54:  (0)
   55-59: █ (1)
   60-64: █ (1)
   65-69: ████████ (8)
   70-74: ██████████████████████████ (26)
   75-79: ████████████████████████████████████████ (40)
   80-84: ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ (138)
   85-89: █████████████████████████ (25)
   90-94: ██████ (6)
  95-100:  (0)
   95-99:  (0)

============================================================
Note: JPEG/PNG use PIL, RAW/TIFF use exiftool for metadata embedding
============================================================

2025-11-21 05:58:38,600 - INFO - Results saved to CSV: Lightroom Test Export.csv
Log file: image_evaluator_20251121_032727.log
```

#### Example Output: PyIQA with CLIP-IQA+ on RTX 3090 (2400 images in ~2.5 minutes)
```bash
$ python image_eval_embed.py process ~/gurushots_facebook_challenge_winners/ --score-engine pyiqa --pyiqa-model clipiqa+_vitL14_512 --pyiqa-score-shift 14 --no-skip-existing --csv clipiqa+_vitL14_512_shift_14_gurushots.csv

2025-11-25 18:09:55,768 - INFO - Completed processing 2400 images in 153.80 seconds

============================================================
PROCESSING SUMMARY
============================================================
Total images processed: 2400
Successful: 2400
Failed: 0
RAW formats (exiftool): 0
Standard formats (PIL): 2400

============================================================
TIMING STATISTICS
============================================================
Total processing time: 153.80 seconds (2.56 minutes)
Time per image: 0.06 seconds
Time per successful image: 0.06 seconds

============================================================
SCORE STATISTICS
============================================================
Average score: 70.40
Median score: 71.00
Standard deviation: 9.29
Range: 33 - 97
Quartiles (Q1/Q3): 64 / 77
Average post-process potential: 70.4/100

============================================================
SCORE DISTRIBUTION
============================================================
     0-4: █ (0)
     5-9: █ (0)
   10-14: █ (0)
   15-19: █ (0)
   20-24: █ (0)
   25-29: █ (0)
   30-34: █ (1)
   35-39: █ (2)
   40-44: █ (6)
   45-49: ██████ (44)
   50-54: ████████████ (82)
   55-59: █████████████████████████ (162)
   60-64: █████████████████████████████████████████████████ (310)
   65-69: ██████████████████████████████████████████████████████████████████████ (448)
   70-74: ████████████████████████████████████████████████████████████████████████████████ (505)
   75-79: ██████████████████████████████████████████████████████████████████████ (447)
   80-84: ███████████████████████████████████████████ (273)
   85-89: ████████████████ (104)
   90-94: ██ (15)
   95-99: █ (1)
  95-100: █ (1)

============================================================
Note: JPEG/PNG use PIL, RAW/TIFF use exiftool for metadata embedding
============================================================

2025-11-25 18:09:55,791 - INFO - Results saved to CSV: clipiqa+_vitL14_512_shift_14_gurushots.csv
```

#### Example Results
```bash
XP Title                        : Intimate Portrait with Wooden Pole
XP Keywords                     : portrait, close-up, shallow depth of field, warm lighting, wooden pole, tongue, blonde hair, braid, skin texture, studio photography, candid, expressive
Image Description               : A woman is sticking her tongue out at the camera while holding up a pool cue stick.
User Comment                    : 80
```
#### Example Image
![tongue](https://github.com/user-attachments/assets/53457193-a2f0-4f88-aead-d44483da4c28)


### Key Features by Tool

#### Image Evaluator (image_eval_embed.py)

- **Multi-criteria evaluation**: Technical quality, composition, lighting, creativity scores
- **Ensemble scoring**: Multiple evaluation passes with median aggregation for consistency
- **EXIF extraction**: Reads camera settings (ISO, aperture, shutter, focal length)
- **Technical analysis**: Histogram analysis, color cast detection, sharpness measurement
- **Metadata embedding**: Embeds score, title, description, keywords into EXIF
- **Caching**: MD5-based caching system for repeat evaluations
- **Format support**: JPEG, PNG, RAW (DNG, NEF), TIFF via PIL and exiftool

#### Stock Photo Evaluator (stock_photo_evaluator.py)

- **Commercial assessment**: Evaluates market viability and searchability
- **Technical validation**: Resolution (4MP min, 12MP rec), DPI (300 standard), sharpness, noise
- **7-score system**: Commercial viability, technical quality, composition clarity, keyword potential, release concerns, rejection risks, overall score
- **Fixable issues detection**: Identifies easy corrections (DPI metadata) vs. critical problems
- **Category recommendations**: EXCELLENT, GOOD, MARGINAL-FIXABLE, MARGINAL, REJECT
- **Detailed reporting**: CSV with all scores, issues, strengths, suggested keywords

### Model Recommendations

**Recommended: qwen3-vl:8b**
- Excellent score distribution and consistency
- Average artistic scores: 80-86 for high-quality images
- Standard deviation: 5.4 (good differentiation)
- Ensemble mode: 1.44 point average difference between runs
- 79% of scores within +/- 2 points across runs

**Alternative Models Tested:**
- `gemma3:4b`: Faster but scores cluster too low (avg 70 for competition winners)
- `gemma3:12b`: Extreme clustering with minimal differentiation (std dev 2.0)
- `llama3.2-vision`: Compatible but not extensively tested

### Implementation Details

**Scoring Consistency:**
Both tools use `temperature=0.3`, `seed=42`, and `top_p=0.9` for deterministic, reproducible results while maintaining nuanced evaluation.

**Ensemble Mode:**
The image evaluator supports multiple evaluation passes per image with median aggregation to reduce variance. Recommended for critical evaluations where consistency is paramount.

**Technical Analysis Algorithms:**

- **Sharpness**: Scale-normalized Laplacian variance via OpenCV
  - Normalizes by √(megapixels) for consistent measurements across resolutions
  - 1MP baseline ensures comparability between 3MP and 50MP images
  
- **Noise**: Camera/ISO-agnostic estimation using 7-step algorithm
  1. Resize to standardized 2048px long edge
  2. Normalize to [0,1] range
  3. Extract high-frequency residual via Gaussian blur
  4. Mask to flat regions using Sobel edge detection (75th percentile threshold)
  5. Calculate robust sigma via MAD (Median Absolute Deviation)
  6. Normalize by dynamic range (p1-p99)
  7. Map to 0-100 severity score
  - Fallback uses full image if <1% pixels in flat regions (logged)
  
- **Clipping**: Per-channel histogram analysis
  - Uses **max** across R/G/B channels (not average)
  - Accurately detects single-channel clipping
  - Highlight range: bins 250-255
  - Shadow range: bins 0-5
  
- **Color Cast**: Dominant channel detection with threshold
  - Threshold: 15.0 mean difference between channels
  - Requires 5+ unit margin to identify dominant channel
  - Labels: warm/red, cool/blue, green, mixed, neutral
  
- **DPI**: Extracted from image metadata when available
  
- **Context Classification**: Automatic LLM-based classification into 10 photography contexts
  - Falls back to 'stock_product' (most restrictive) if uncertain
  - Manual override available via command-line flag

**Logging:**
Both tools use Python's logging module with file and console handlers. Log files are timestamped and include DEBUG/INFO/WARNING/ERROR levels.

**Error Handling:**
- Invalid paths and missing directories
- Corrupted or unreadable images
- API timeouts with exponential backoff retry (2s, 4s, 8s...)
- Permanent error detection (FileNotFoundError, PermissionError, ValueError, TypeError, KeyError)
- Malformed JSON responses with score extraction fallback
- Technical analysis status tracking ('success'/'error')
- Comprehensive ISO parsing (handles "ISO 1600", "1,600", "1600/3200" dual ISO, "100.0" float)

**Backup System:**
Images are backed up before metadata modification by appending `.original` to the filename. The rollback command restores from these backups.

### Running Tests

After installing the requirements, you can execute the unit tests using `pytest` from the repository root:

```bash
pytest
```

### License

This project is licensed under the MIT License. See the LICENSE file for details.
