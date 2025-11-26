# Image Evaluator

AI-powered image quality assessment with EXIF metadata embedding.

## How It Works

1. **Context classification (Ollama-only step)**  
   Every image is resized to 1024 px and sent to `qwen3-vl:8b` (via Ollama or a remote endpoint) purely to decide which of the 10 photographic profiles best fits the scene (landscape, portrait_highkey, sports_action, etc.). No scoring happens here—this step just selects the profile so the right weights and penalties can be applied later.

2. **Profile-specific weighting**  
   Each profile stores its own weight table covering five PyIQA models:
   - `clipiqa+_vitL14_512`
   - `laion_aes`
   - `musiq-ava`
   - `musiq-paq2piq`
   - `maniqa` (any of the maniqa variants)

   In addition to the weighted models, we always compute a sixth signal (`pyiqa_diff_z`) that measures disagreement between models and acts as a stability penalty/bonus.

3. **Multi-model PyIQA scoring**  
   After the profile is known, the image is resized to 2048 px and evaluated by **all six** metrics. The calibrated scores are converted to z-scores, blended with the profile’s weights, and adjusted with technical rules (sharpness, clipping, color cast, brightness) pulled from the same profile.

4. **Metadata embedding**  
   The weighted composite score, profile name, technical metrics, and keywords are written into EXIF (JPEG/PNG via piexif, RAW/TIFF via exiftool). CSV output mirrors the embedded metadata.

## Features

- **Profile-aware PyIQA pipeline**: Automatic context detection feeds 10 tailored weighting/rule sets.
- **Multi-model fusion**: clipiqa+_vitL14_512, laion_aes, musiq-ava, maniqa, musiq-paq2piq, plus a model disagreement metric.
- **Metadata embedding**: Composite score, profile, warnings, and keywords written directly to EXIF.
- **Technical analysis**: Sharpness, noise, clipping, and color-cast metrics drive rule penalties and CSV reporting.
- **Batch processing**: Parallel execution with resumable CSV output and optional caching.
- **Format support**: JPEG, PNG, TIFF, RAW (NEF, DNG, CR2, ARW) with automatic backups.

## Installation

```bash
pip install -r requirements.txt
```

For RAW metadata embedding, install [exiftool](https://exiftool.org/).

## Quick Start

```bash
# Default: PyIQA scoring on current directory
python image_eval_embed.py

# Process a specific folder
python image_eval_embed.py process /path/to/images

# Use Ollama LLM for detailed evaluation
python image_eval_embed.py process /path/to/images --score-engine ollama
```

## Scoring Backends

PyIQA is the default and recommended backend. Ollama-based scoring (full LLM critiques) still exists for legacy workflows, but the current pipeline only uses the vision LLM for context classification before running the PyIQA ensemble. If you explicitly pass `--score-engine ollama`, the older “LLM does everything” mode is used instead of the profile pipeline.

## CLI Reference

```
python image_eval_embed.py <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `process` | Evaluate images and embed metadata (default) |
| `rollback` | Restore images from backups |
| `stats` | Print statistics from existing CSV |

### Options

Most flows only need the defaults. Options related to legacy full-LLM scoring (`--score-engine ollama`, `--prompt-file`, `--ensemble`) are provided for backward compatibility.

**Scoring Engine:**
```
--score-engine {pyiqa,ollama}  Backend (default: pyiqa)
```

**PyIQA Options:**
```
--pyiqa-model NAME        Metric name (default: clipiqa+_vitl14_512)
--pyiqa-device DEVICE     cuda:0, cpu (default: auto)
--pyiqa-batch-size N      Images per batch (default: 4)
--pyiqa-score-shift N     Score adjustment
--pyiqa-max-models N      Max models in VRAM (default: 1)
--context-host-url URL    Alternate Ollama endpoint for context classification
```

**Ollama Options:**
```
--model NAME              LLM model (default: qwen3-vl:8b)
--ensemble N              Evaluation passes (default: 1)
--prompt-file FILE        Custom prompt template
--context NAME            Manual context (e.g., landscape, portrait)
--no-context-classification  Skip auto context detection
```

**General:**
```
--workers N               Parallel workers (default: 4)
--csv PATH                Output CSV path
--skip-existing           Skip images with metadata (default)
--no-skip-existing        Reprocess all images
--min-score N             Filter results by minimum score
--file-types EXT,EXT      Filter by extension (e.g., jpg,png)
--dry-run                 Preview without changes
--backup-dir DIR          Backup location
--verify                  Verify metadata after writing
--verbose, -v             Debug output
```

## Context Profiles

The evaluator uses 10 photography profiles with tailored thresholds:

| Profile | Use Case |
|---------|----------|
| `stock_product` | Product/catalog (strictest) |
| `macro_food` | Food and macro photography |
| `portrait_neutral` | Standard portraits |
| `portrait_highkey` | Bright, airy portraits |
| `landscape` | Nature and scenery |
| `street_documentary` | Street and documentary |
| `sports_action` | Sports and wildlife |
| `concert_night` | Low-light, concerts |
| `architecture_realestate` | Buildings and interiors |
| `fineart_creative` | Experimental (most flexible) |

Override with `--context landscape` or let auto-classification choose.

## Environment Variables

```bash
IMAGE_EVAL_DEFAULT_FOLDER   # Default image folder
IMAGE_EVAL_OLLAMA_URL       # Ollama endpoint
IMAGE_EVAL_WORKERS          # Worker count
```

## Example Output

```
============================================================
PROCESSING SUMMARY
============================================================
Total images processed: 2400
Successful: 2400
Failed: 0

============================================================
TIMING STATISTICS
============================================================
Total processing time: 153.80 seconds (2.56 minutes)
Time per image: 0.06 seconds

============================================================
SCORE STATISTICS
============================================================
Average score: 70.40
Median score: 71.00
Standard deviation: 9.29
Range: 33 - 97

============================================================
SCORE DISTRIBUTION
============================================================
   65-69: ██████████████████████████████████████████████████████████████████████ (448)
   70-74: ████████████████████████████████████████████████████████████████████████████████ (505)
   75-79: ██████████████████████████████████████████████████████████████████████ (447)
   80-84: ███████████████████████████████████████████ (273)
```

## Additional Tools

### Stock Photo Evaluator

Specialized assessment for stock photography submission:

```bash
python stock_photo_evaluator.py /photos http://localhost:11434/api/generate
```

### Technical Analysis

Quick single-file technical check:

```bash
python technical_analysis.py /path/to/image.NEF --format json
```

### CSV to PostgreSQL

Upload results to a database:

```bash
python csv_to_postgres.py results.csv --db-url postgres://localhost:5432/photos
```

## Running Tests

```bash
pytest
```

## License

MIT License
