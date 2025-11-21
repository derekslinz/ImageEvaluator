# Image Evaluator

## AI-Powered Image Metadata Embedding Tool

This tool processes images in folders, evaluates them using AI vision models (via Ollama), and embeds the resulting metadata into the images' EXIF data.

### Features

#### Core Functionality
- **Multi-format support**: JPEG, PNG, DNG, NEF, TIF/TIFF
- **AI evaluation**: Uses Ollama vision models for intelligent scoring
- **Metadata embedding**: Score, title, description, and keywords
- **Recursive processing**: Processes all subdirectories
- **Smart backups**: Creates backups before modifying files

#### Performance
- **Parallel processing**: Process multiple images concurrently (configurable workers)
- **Progress tracking**: Real-time progress bar with tqdm
- **Statistics**: Avg/min/max scores, distribution histogram
- **CSV export**: Timestamped reports for all processed images

#### Quality & Safety
- **Score validation**: Ensures scores are 1-100, extracts numbers from malformed responses
- **Auto-retry**: Exponential backoff for failed API calls (up to 3 attempts)
- **Image validation**: Detects and skips corrupted files
- **Verification**: Optional metadata verification after embedding
- **Backup directory**: Store backups separately from originals
- **Rollback**: Restore from backups if needed

#### Flexibility
- **Configurable model**: Use any Ollama vision model
- **Custom prompts**: Load evaluation criteria from files
- **Selective processing**: Filter by file type, min score, skip existing
- **Dry-run mode**: Preview what would be processed without changes

### Requirements

- Python 3.8 or higher
- Ollama server with vision model (e.g., `qwen3-vl:8b`, `llama3.2-vision`)
- `exiftool` for RAW file metadata embedding
- Required Python libraries (see `requirements.txt`):
  - `Pillow` - Image processing
  - `requests` - API calls
  - `piexif` - EXIF manipulation
  - `pydantic` - Data validation
  - `colorama` - Colored output
  - `tqdm` - Progress bars

Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Start Ollama server:**
   ```bash
   ollama serve
   ollama pull qwen3-vl:8b
   ```

2. **Process images:**
   ```bash
   python image_eval_embed.py process /path/to/images http://localhost:11434/api/generate
   ```

3. **View results:**
   - Check console for statistics and distribution
   - CSV report saved automatically with timestamp

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
--csv PATH               # Custom CSV output path
--prompt-file FILE       # Custom prompt template file
--skip-existing          # Skip images with metadata (default: True)
--no-skip-existing       # Process all images
--min-score N            # Only save results >= N
--file-types EXT,EXT     # Filter by extensions (e.g., jpg,png)
--dry-run                # Preview without changes
--backup-dir DIR         # Store backups separately
--verify                 # Verify metadata after embedding
```

**Examples:**
```bash
# Basic usage (4 workers, default model)
python image_eval_embed.py process /photos http://localhost:11434/api/generate

# High-performance setup
python image_eval_embed.py process /photos http://localhost:11434/api/generate \
  --workers 8 \
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
#### Example Output: model qwen3-vl:8b running on ollama on an Applke M4 Max with --ensemble=3 (3 passes per image). Each image is ~50 megapixels. Running on gemma3:4b was faster, but the results were all clustered around the same score. It's notably faster on a RTX 3090, but still time consuming with 3 passes. Smaller images process faster, obviously
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


### Functionality

- **sanitize_string(s: str)**: Cleans up strings by removing null bytes and newlines to ensure compatibility with EXIF data.

- **Metadata Class**: Defines the structure of the metadata received from the API, including score, title, description, and keywords.

- **embed_metadata(image_path: str, metadata: Dict)**: Embeds the provided metadata into the specified image's EXIF data. It handles user comments, title, description, and keywords, ensuring proper encoding.

- **process_images_in_folder(folder_path, ollama_host_url)**: Processes each image in the specified folder, sending it to the API and embedding the returned metadata.

### Logging

The script uses Python's built-in logging module to log information and errors. Logs are printed to the console with timestamps and severity levels.

### Error Handling

The script includes error handling for:
- Invalid folder paths.
- Non-directory paths.
- Absence of image files in the specified directory.
- API request failures.

### Backup

Before embedding metadata, the script creates a backup of the original image by appending `.original` to the filename.

### Running Tests

After installing the requirements, you can execute the unit tests using `pytest` from the repository root:

```bash
pytest
```

### License

This project is licensed under the MIT License. See the LICENSE file for details.
