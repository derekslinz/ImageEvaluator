# Image Evaluator

## AI-Powered Image Evaluation Suite

A toolkit for evaluating images using AI vision models (via Ollama) with two specialized tools:
1. **Image Evaluator** - Artistic merit evaluation with EXIF metadata embedding
2. **Stock Photo Evaluator** - Commercial stock photography suitability assessment

### Features

#### Image Evaluator (image_eval_embed.py)

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
- Exponential backoff retry logic
- Image validation and corruption detection
- Optional metadata verification
- Backup directory support
- Full rollback capability

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
- Sharpness measurement via Laplacian variance
- Noise estimation
- Histogram clipping detection
- Aspect ratio validation

**Recommendations:**
- EXCELLENT: Ready for immediate submission
- GOOD: Strong candidate with minor considerations
- MARGINAL-FIXABLE: Needs easy corrections (e.g., DPI metadata)
- MARGINAL: Needs improvement before submission
- REJECT: Does not meet stock standards

**Output:**
- Detailed CSV with all scores and analysis
- Suggested keywords for stock submission
- Issues identification with fixable vs. critical problems
- Strengths highlighting
- Category recommendations

### Requirements

- Python 3.8 or higher
- Ollama server with vision model (recommended: `qwen3-vl:8b`, also supports `llama3.2-vision`)
- `exiftool` for RAW file metadata embedding
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

#### Image Evaluator (Artistic Merit)

1. **Start Ollama server:**
   ```bash
   ollama serve
   ollama pull qwen3-vl:8b
   ```

2. **Evaluate images:**
   ```bash
   python image_eval_embed.py process /path/to/images http://localhost:11434/api/generate
   ```

3. **View results:**
   - Console displays statistics and score distribution (warning count now reported)
   - CSV report saved with timestamp and includes the measured technical metrics plus any automated warnings/filter cues
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
python image_eval_embed.py process <folder_path> <ollama_url> [OPTIONS]
```

Quick start with defaults:
```bash
python image_eval_embed.py
```
Running without arguments now assumes the `process` command, uses your current working directory (or `IMAGE_EVAL_DEFAULT_FOLDER`) as the image source, and talks to `http://localhost:11434/api/generate` (or `IMAGE_EVAL_OLLAMA_URL`).

**Environment overrides for defaults:**
- `IMAGE_EVAL_DEFAULT_FOLDER` – absolute path to use when a folder isn’t provided
- `IMAGE_EVAL_OLLAMA_URL` – Ollama endpoint to use when omitted
- `IMAGE_EVAL_WORKERS` – worker count used when `--workers` isn’t supplied

**Arguments:**
- `folder_path`: Directory containing images (processes recursively); defaults to current working directory or `IMAGE_EVAL_DEFAULT_FOLDER` when omitted
- `ollama_url`: Full Ollama API endpoint (e.g., `http://localhost:11434/api/generate`); defaults to `IMAGE_EVAL_OLLAMA_URL` or `http://localhost:11434/api/generate`

**Options:**
```bash
--workers N              # Parallel workers (default: IMAGE_EVAL_WORKERS or 4)
--model NAME             # Ollama model (default: qwen3-vl:8b)
--ensemble N             # Evaluation passes for consistency (default: 1)
--csv PATH               # Custom CSV output path
--prompt-file FILE       # Custom prompt template file
--skip-existing          # Skip images with metadata (default: True)
--no-skip-existing       # Process all images
--min-score N            # Only save results >= N
--file-types EXT,EXT     # Filter by extensions (e.g., jpg,png)
--dry-run                # Preview without changes
--backup-dir DIR         # Store backups separately
--verify                 # Verify metadata after embedding
--cache-dir DIR          # Enable caching for repeat evaluations
```

**Examples:**
```bash
# Basic usage (4 workers, default model)
python image_eval_embed.py process /photos http://localhost:11434/api/generate

# High-performance setup with ensemble scoring
python image_eval_embed.py process /photos http://localhost:11434/api/generate \
  --workers 8 \
  --ensemble 3 \
  --backup-dir /backups/photos

# Only top-quality JPEGs
python image_eval_embed.py process /photos http://localhost:11434/api/generate \
  --file-types jpg,jpeg \
  --min-score 80

# Custom model and prompt
python image_eval_embed.py process /photos http://localhost:11434/api/generate \
  --model llama3.2-vision \
  --prompt-file custom_critic.txt

# Preview without changes
python image_eval_embed.py process /photos http://localhost:11434/api/generate \
  --dry-run

# Reprocess everything with verification
python image_eval_embed.py process /photos http://localhost:11434/api/generate \
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

#### Stock Photo Evaluator Command

Basic syntax:
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
  MARGINAL-FIXABLE: 132 (55.9%) - Easy fixes available
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

#### Upload Results to PostgreSQL

Once you have a CSV from either evaluator, use `csv_to_postgres.py` to push it into Postgres:
```bash
python csv_to_postgres.py path/to/results.csv --db-url postgres://localhost:5432/photos
```
The script normalizes the header, creates `image_evaluation_results` (or your chosen table) with text columns, and inserts all rows; pass `--truncate` if you want to drop existing rows first. Ensure `psycopg2-binary` is installed along with the other requirements.

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

**Technical Analysis:**
- Sharpness: Laplacian variance method via OpenCV
- Noise: Standard deviation estimation in flat areas
- Clipping: Histogram analysis of highlight/shadow regions
- DPI: Extracted from image metadata when available

**Logging:**
Both tools use Python's logging module with file and console handlers. Log files are timestamped and include DEBUG/INFO/WARNING/ERROR levels.

**Error Handling:**
- Invalid paths and missing directories
- Corrupted or unreadable images
- API timeouts with automatic retry (stock evaluator: 300s timeout)
- Malformed JSON responses with score extraction fallback

**Backup System:**
Images are backed up before metadata modification by appending `.original` to the filename. The rollback command restores from these backups.

### Running Tests

After installing the requirements, you can execute the unit tests using `pytest` from the repository root:

```bash
pytest
```

### License

This project is licensed under the MIT License. See the LICENSE file for details.
