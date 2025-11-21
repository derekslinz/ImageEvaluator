# Image Evaluator

## AI-Powered Image Metadata Embedding Tool

This tool processes images in folders, evaluates them using AI vision models (via Ollama), and embeds the resulting metadata into the images' EXIF data.

### Features

#### Core Functionality
- 🖼️ **Multi-format support**: JPEG, PNG, DNG, NEF, TIF/TIFF
- 🤖 **AI evaluation**: Uses Ollama vision models for intelligent scoring
- 📝 **Metadata embedding**: Score, title, description, and keywords
- 🔄 **Recursive processing**: Processes all subdirectories
- 💾 **Smart backups**: Creates backups before modifying files

#### Performance
- ⚡ **Parallel processing**: Process multiple images concurrently (configurable workers)
- 📊 **Progress tracking**: Real-time progress bar with tqdm
- 📈 **Statistics**: Avg/min/max scores, distribution histogram
- 📄 **CSV export**: Timestamped reports for all processed images

#### Quality & Safety
- ✅ **Score validation**: Ensures scores are 1-100, extracts numbers from malformed responses
- 🔁 **Auto-retry**: Exponential backoff for failed API calls (up to 3 attempts)
- 🔍 **Image validation**: Detects and skips corrupted files
- ✓ **Verification**: Optional metadata verification after embedding
- 🗂️ **Backup directory**: Store backups separately from originals
- ↩️ **Rollback**: Restore from backups if needed

#### Flexibility
- 🎛️ **Configurable model**: Use any Ollama vision model
- 📝 **Custom prompts**: Load evaluation criteria from files
- 🎯 **Selective processing**: Filter by file type, min score, skip existing
- 👁️ **Dry-run mode**: Preview what would be processed without changes

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
#### Example Output
```bash
python /Users/lderek/ImageEvaluator/image_eval_embed.py ~/Pictures/Lightroom\ Saved\ Photos/embed http://10.0.1.50:11434/api/generate 
Response for tongue.jpg: {"model":"llama3.2-vision","created_at":"2025-01-24T15:55:58.61357571Z","response":"{\"score\": \"\u003e92\", \"title\": \"Woman with Tongue Out\", \"description\": \"A woman is holding on to a wooden pole while sticking her tongue out.\", \"keywords\": \"woman, blonde, pole, stick, tongue, mouth, hand, fingers, green eyes, red lips, white tank top\"}","done":true,"done_reason":"stop","context":[128006,882,128007,271,58,1931,12,15,60,128256,271,83445,420,2217,323,9993,264,35876,5573,1990,220,16,12,1041,11075,555,459,16945,16865,315,279,10512,596,11156,4367,11,37637,14638,323,28697,11,90039,37637,14638,2033,13,1789,1855,2217,11,471,1664,8627,12400,4823,36051,287,311,279,2768,5497,512,1,12618,794,366,396,14611,1472,690,1101,471,264,53944,2316,3196,389,279,2217,1701,912,810,1109,220,1399,5885,11,264,4096,315,279,2217,1701,912,810,1109,220,1049,5885,323,709,311,220,717,9959,16570,9681,11,32783,49454,660,11,2085,82961,128009,128006,78191,128007,271,5018,12618,794,30057,6083,498,330,2150,794,330,96149,449,51491,361,4470,498,330,4789,794,330,32,5333,374,10168,389,311,264,23162,26078,1418,38072,1077,25466,704,10684,330,30095,794,330,22803,11,27117,11,26078,11,9396,11,25466,11,11013,11,1450,11,19779,11,6307,6548,11,2579,23726,11,4251,13192,1948,9388],"total_duration":1853019841,"load_duration":92021277,"prompt_eval_count":116,"prompt_eval_duration":166000000,"eval_count":67,"eval_duration":929000000}
Metadata received for tongue.jpg: {'score': '>92', 'title': 'Woman with Tongue Out', 'description': 'A woman is holding on to a wooden pole while sticking her tongue out.', 'keywords': 'woman, blonde, pole, stick, tongue, mouth, hand, fingers, green eyes, red lips, white tank top'}
Embedding score in User Comment: b'ASCII\x00\x00\x00>92'
User Comment metadata successfully embedded in /Users/lderek/Pictures/Lightroom Saved Photos/embed/tongue.jpg
Embedding Title: b'W\x00o\x00m\x00a\x00n\x00 \x00w\x00i\x00t\x00h\x00 \x00T\x00o\x00n\x00g\x00u\x00e\x00 \x00O\x00u\x00t\x00'
Title metadata successfully embedded in /Users/lderek/Pictures/Lightroom Saved Photos/embed/tongue.jpg
Embedding Description: b'A\x00 \x00w\x00o\x00m\x00a\x00n\x00 \x00i\x00s\x00 \x00h\x00o\x00l\x00d\x00i\x00n\x00g\x00 \x00o\x00n\x00 \x00t\x00o\x00 \x00a\x00 \x00w\x00o\x00o\x00d\x00e\x00n\x00 \x00p\x00o\x00l\x00e\x00 \x00w\x00h\x00i\x00l\x00e\x00 \x00s\x00t\x00i\x00c\x00k\x00i\x00n\x00g\x00 \x00h\x00e\x00r\x00 \x00t\x00o\x00n\x00g\x00u\x00e\x00 \x00o\x00u\x00t\x00.\x00'
Description metadata successfully embedded in /Users/lderek/Pictures/Lightroom Saved Photos/embed/tongue.jpg
Embedding Keywords: b'w\x00o\x00m\x00a\x00n\x00,\x00 \x00b\x00l\x00o\x00n\x00d\x00e\x00,\x00 \x00p\x00o\x00l\x00e\x00,\x00 \x00s\x00t\x00i\x00c\x00k\x00,\x00 \x00t\x00o\x00n\x00g\x00u\x00e\x00,\x00 \x00m\x00o\x00u\x00t\x00h\x00,\x00 \x00h\x00a\x00n\x00d\x00,\x00 \x00f\x00i\x00n\x00g\x00e\x00r\x00s\x00,\x00 \x00g\x00r\x00e\x00e\x00n\x00 \x00e\x00y\x00e\x00s\x00,\x00 \x00r\x00e\x00d\x00 \x00l\x00i\x00p\x00s\x00,\x00 \x00w\x00h\x00i\x00t\x00e\x00 \x00t\x00a\x00n\x00k\x00 \x00t\x00o\x00p\x00'
Keywords metadata successfully embedded in /Users/lderek/Pictures/Lightroom Saved Photos/embed/tongue.jpg
```
#### Example Results
```bash
 exiftool -XPTitle -XPKeywords -UserComment tongue.jpg
XP Title                        : Woman with Tongue Out and Pool Cue
XP Keywords                     : woman tongue out, blonde hair, pool cue, white tank top, braided ponytail, eyes closed, tongue out
User Comment                    : 95
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
