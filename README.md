# Image Evaluator

AI-powered image quality assessment with EXIF metadata embedding.

## How It Works

1. **Context classification (Ollama-only step)**  
   Every image is resized to 1024 px and sent to `qwen3-vl:8b` (via Ollama or a remote endpoint) purely to decide which of the 10 photographic profiles best fits the scene (landscape, portrait_highkey, sports_action, etc.). This same model can optionally be reused later to rewrite titles/descriptions/keywords.

2. **Profile-specific weighting**  
   Each profile stores its own weight table covering five PyIQA models:
   - `clipiqa+_vitL14_512`
   - `laion_aes`
   - `musiq-ava`
   - `musiq-paq2piq`
   - `maniqa` (any of the maniqa variants)

   In addition to the weighted models, we always compute a sixth signal (`pyiqa_diff_z`) that measures disagreement between models and acts as a stability penalty/bonus.

3. **Multi-model PyIQA scoring**  
   After the profile is known, the image is resized to 2048 px and evaluated by all five models plus one disagreement metric (`pyiqa_diff_z`). The calibrated scores are converted to z-scores, blended with the profile’s weights, and adjusted with technical rules (sharpness, clipping, color cast, brightness) pulled from the same profile.

4. **Metadata & stock suitability**  
   The weighted composite score, profile name, technical metrics, and keywords are written into EXIF (JPEG/PNG via piexif, RAW/TIFF via exiftool). CSV output mirrors everything. Optional add-ons:
   - `--ollama-metadata` rewrites title/description/keywords with the vision LLM.
   - `--stock-eval` asks the LLM for commercial viability, release concerns, rejection risk, and a final stock recommendation (EXCELLENT/GOOD/MARGINAL/REJECT).

## Features

- **Profile-aware PyIQA pipeline**: Automatic context detection feeds 10 tailored weighting/rule sets.
- **Multi-model fusion**: clipiqa+_vitL14_512, laion_aes, musiq-ava, maniqa, musiq-paq2piq, plus a model disagreement metric.
- **Metadata & stock summaries**: Composite score, profile, warnings, and keywords written directly to EXIF, with optional Ollama rewrites plus stock suitability scoring (commercial viability, release risk, rejection risk).
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

# Process a single image
python image_eval_embed.py process --image /path/to/image.jpg

# Generate Ollama-driven titles/descriptions/keywords
python image_eval_embed.py process /path/to/images --ollama-metadata

# Add stock suitability scoring and recommendations
python image_eval_embed.py process /path/to/images --stock-eval
```

## Scoring & Metadata

PyIQA now performs all scoring. The vision LLM (Ollama) is used for context detection and, if `--ollama-metadata` is set, to regenerate titles/descriptions/keywords. Enable `--stock-eval` to have the same model generate commercial viability, release risk, rejection risk, and a final agency-ready recommendation. There is no longer an Ollama-only scoring mode.

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

Most flows only need the defaults. Add `--ollama-metadata` if you want the vision LLM to rewrite descriptions/keywords before embedding.

**PyIQA Options:**
```
--pyiqa-model NAME        Metric name (default: clipiqa+_vitL14_512)
--pyiqa-device DEVICE     cuda:0, cpu (default: auto)
--pyiqa-score-shift N     Score adjustment
--pyiqa-max-models N      Max models in VRAM (default: 1)
```

**Ollama / Context Options:**
```
--model NAME              LLM model (default: qwen3-vl:8b)
--context-host-url URL    Alternate Ollama endpoint for context classification
--ollama-metadata         Use Ollama to rewrite title/description/keywords
--stock-eval              Run stock photography suitability scoring (commercial/release)
--context NAME            Manual context (e.g., landscape, portrait)
--no-context-classification  Skip auto context detection
```

**Recommended Vision Model:**  
For profile determination, `ingu627/Qwen2.5-VL-7B-Instruct-Q5_K_M` provides excellent classification accuracy with good performance. Install via:
```bash
ollama pull ingu627/Qwen2.5-VL-7B-Instruct-Q5_K_M
```

**General:**
```
--workers N               Parallel workers (default: 4)
--csv PATH                Output CSV path
--image PATH              Process a single image instead of scanning a folder
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

The evaluator uses 13 photography profiles with tailored thresholds:

| Profile | Use Case |
|---------|----------|
| `studio_photography` | Controlled studio shots (strictest) |
| `macro_food` | Food photography |
| `macro_nature` | Nature macro (insects, flowers, textures) |
| `portrait_neutral` | Standard portraits |
| `portrait_highkey` | Bright, airy portraits |
| `landscape` | Nature and scenery |
| `street_documentary` | Street and documentary |
| `sports_action` | Sports and fast action |
| `wildlife_animal` | Animals as main subject (pets, birds, mammals) |
| `night_artificial_light` | Concerts, neon, city lights, nightlife |
| `night_natural_light` | Stars, Milky Way, aurora, moonlight |
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

### IQA Calibration Builder

Generate stable PyIQA calibration parameters from a scored reference set (the default points to `/root/gurushots_facebook_challenge_winners/all.csv`). This writes `iqa_calibration.json` with per-model mean/std and optional percentile support.

```bash
python calibrate_iqa.py \
  --input /root/gurushots_facebook_challenge_winners/all.csv \
  --output iqa_calibration.json \
  --overwrite
```

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
