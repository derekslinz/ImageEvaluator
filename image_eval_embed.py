import argparse
import base64
import json  # Import json for parsing the response
import logging
import os
from typing import Dict

import piexif
import requests
from PIL import Image
from colorama import Fore
from pydantic import BaseModel, field_validator, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sanitize_string(s: str) -> str:
    """Sanitize string for Exif compatibility."""
    return s.replace('\x00', '').replace('\n', ' ').replace('\r', ' ')  # Remove null bytes and newlines


class Metadata(BaseModel):
    total_score: int
    technical_quality: int
    creative_appeal: int
    monetization_potential: int
    maximum_potential_score: int
    title: str
    description: str
    keywords: str

    class Config:
        extra = 'allow'  # Allow extra fields without raising an error

    @field_validator('keywords')
    def validate_keywords(cls, v):
        keyword_list = v.split(',')
        # Truncate to 12 keywords if there are more than 12
        if len(keyword_list) > 12:
            keyword_list = keyword_list[:12]
        return ','.join(keyword_list).strip()  # Join back to a string and strip whitespace


def embed_metadata(image_path: str, metadata: Dict):
    try:
        # Validate metadata
        validated_metadata = Metadata(**metadata)

        # Prepare Exif data with sanitized strings
        exif_dict = {
            piexif.ExifIFD.UserComment: sanitize_string(validated_metadata.model_dump_json()).encode('utf-8'),
            piexif.ImageIFD.ImageDescription: sanitize_string(validated_metadata.description).encode('utf-8'),
            piexif.ImageIFD.XPTitle: sanitize_string(validated_metadata.title).encode('utf-8'),
            piexif.ImageIFD.XPKeywords: sanitize_string(validated_metadata.keywords).encode('utf-8')
        }

        exif_bytes = piexif.dump(exif_dict)

        # Backup original image by appending .original suffix
        backup_image_path = f"{os.path.splitext(image_path)[0]}.original{os.path.splitext(image_path)[1]}"
        os.rename(image_path, backup_image_path)  # Rename original image to backup

        # Open the backup image and save with new metadata
        img = Image.open(backup_image_path)
        img.save(image_path, exif=exif_bytes)
        print(Fore.GREEN + f"Metadata successfully embedded in {image_path}" + Fore.RESET)

    except (ValidationError, Exception) as e:
        logger.error(f"Error embedding metadata in {image_path}: {e}")


def process_images_in_folder(folder_path, ollama_host_url):
    headers = {'Content-Type': 'application/json'}
    results = []  # To store processing results

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            payload = {
                "model": "llama3.2-vision",
                "stream": False,
                "images": [encoded_image],
                "prompt": "Evaluate this image and assign a numerical score between 1-100 composed of sub scores for technical quality, creative appeal, and monetization potential. For each image, return well-formatted JSON adhering to the following pattern:\n\"total_score\": 82,\n\"technical_quality\": 85,\n\"creative_appeal\": 80,\n\"monetization_potential\": 78,\n\"maximum_potential_score\": 90\nNormalize all scores to a 100-point scale and respond with the score values and score names. You will also return a descriptive title based on the image using no more than 140 characters, a description of the image using no more than 140 words and between 5 and 20 relevant keyword tags, comma separated, without hashtags",
                "format": {
                    "type": "object",
                    "properties": {
                        "total_score": {"type": "integer"},
                        "technical_quality": {"type": "integer"},
                        "creative_appeal": {"type": "integer"},
                        "monetization_potential": {"type": "integer"},
                        "maximum_potential_score": {"type": "integer"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "keywords": {"type": "string"}
                    },
                    "required": [
                        "total_score",
                        "technical_quality",
                        "creative_appeal",
                        "monetization_potential",
                        "maximum_potential_score",
                        "title",
                        "description",
                        "keywords"
                    ]
                }
            }

            try:
                response = requests.post(ollama_host_url, json=payload, headers=headers)
                if response.status_code == 200:
                    print(f"Response for {filename}: {response.text}")
                    response_data = response.json()  # Get the full response
                    response_metadata = json.loads(response_data['response'])  # Extract and parse the 'response' field

                    # Embed metadata after processing the image
                    embed_metadata(image_path, response_metadata)

                    results.append((filename, response_metadata))  # Store successful result

                else:
                    print(f"Failed to process {filename}: {response.status_code}")
                    results.append((filename, None))  # Store failure result
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {filename}: {e}")
                results.append((filename, None))  # Store failure result

    return results  # Return processing results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images in a specified folder.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    parser.add_argument('ollama_host_url', type=str, help='Full url of your ollama API endpoint ')  # Added URL argument
    args = parser.parse_args()

    # Validate folder path
    if not os.path.exists(args.folder_path):
        logger.error(f"The folder path '{args.folder_path}' does not exist.")
        exit(1)
    if not os.path.isdir(args.folder_path):
        logger.error(f"The path '{args.folder_path}' is not a directory.")
        exit(1)
    if not any(filename.endswith(('.jpg', '.jpeg', '.png')) for filename in os.listdir(args.folder_path)):
        logger.error(f"No image files found in the directory '{args.folder_path}'.")
        exit(1)

    results = process_images_in_folder(args.folder_path, args.ollama_host_url)  # Pass URL to function

    # Count successful and failed image processing
    success_count = sum(1 for _, result in results if result is not None)
    failure_count = len(results) - success_count

    print(f"Successfully processed images: {success_count}")
    print(f"Failed to process images: {failure_count}")
