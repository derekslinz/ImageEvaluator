# README

## Image Metadata Embedding Tool

This tool processes images in a specified folder, evaluates them using an external API, and embeds the resulting metadata into the images' EXIF data.

### Features

- Reads images from a specified directory.
- Sends images to an external API for evaluation.
- Embeds the returned metadata (score, title, keywords) into the image's EXIF data.
- Supports JPEG and PNG formats.
- Creates backups of original images before modifying them.

### Requirements

- Python 3.6 or higher
- Required libraries:
  - `Pillow`
  - `requests`
  - `piexif`
  - `pydantic`
  - `colorama`

You can install the required libraries using pip:

```bash
pip install Pillow requests piexif pydantic colorama
```

### Usage

1. **Clone the repository** or download image_eval_embedded.py.
2. Star ollama with 'ollama server' and make sure you have the ollama3.2-vision model available.
3. **Run the script** from the command line with the following syntax:

   ```bash
   python image_eval_embedded.py <folder_path> <ollama_host_url>
   ```

   - `<folder_path>`: Path to the folder containing images.
   - `<ollama_host_url>`: Full URL of your Ollama API endpoint in this format: http://localhost:11434/api/generate
  

#### Example
### Example syntax
```bash
python image_eval_embedded.py /path/to/images http://localhost:8000/api/evaluate](http://localhost:11434/api/generate
```
#### Exmaple Output
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

### License

This project is licensed under the MIT License. See the LICENSE file for details.
