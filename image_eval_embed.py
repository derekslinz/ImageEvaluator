import argparse
import base64
import json  # Import json for parsing the response
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import piexif
import piexif.helper
import requests
from PIL import Image
from colorama import Fore
from pydantic import BaseModel, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sanitize_string(s: str) -> str:
    """Sanitize string for Exif compatibility."""
    return s.replace('\x00', '').replace('\n', ' ').replace('\r', ' ')  # Remove null bytes and newlines


class Metadata(BaseModel):
    score: int  # Changed from str to int
    title: str
    description: str
    keywords: str

    class Config:
        extra = 'allow'  # Allow extra fields without raising an error

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


def embed_metadata(image_path: str, metadata: Dict):
    try:
        # Open the image to access its EXIF data
        with Image.open(image_path) as img:
            exif_data = img.info.get("exif", b"")
            if exif_data:
                exif_dict = piexif.load(exif_data)
            else:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}  # Create a new EXIF structure

            user_comment = piexif.helper.UserComment.dump(metadata.get('score', ''))
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
            print(f"Embedding score in User Comment: {user_comment}")

            # Embedding Title
            title = metadata.get('title', '').encode('utf-16le')
            exif_dict["0th"][piexif.ImageIFD.XPTitle] = title
            print(f"Embedding Title: {title}")

            # Embedding Description
            description = metadata.get('description', '').encode('utf-16le')
            exif_dict["0th"][piexif.ImageIFD.XPComment] = description
            print(f"Embedding Description: {description}")

            # Embedding Keywords
            keywords = metadata.get('keywords', '').encode('utf-16le')
            exif_dict["0th"][piexif.ImageIFD.XPKeywords] = keywords
            print(f"Embedding Keywords: {keywords}")

            # Prepare Exif data with sanitized strings
            exif_bytes = piexif.dump(exif_dict)

            # Backup original image by appending .original suffix
            backup_image_path = f"{os.path.splitext(image_path)[0]}.original{os.path.splitext(image_path)[1]}"
            if os.path.exists(image_path):  # Ensure the file exists before renaming
                os.rename(image_path, backup_image_path)

            # Save with new metadata
            img.save(image_path, exif=exif_bytes)
            print(Fore.GREEN + f"Metadata successfully embedded in {image_path}" + Fore.RESET)

    except Exception as e:
        logger.error(f"Error embedding metadata in {image_path}: {e}")


def process_images_in_folder(folder_path: str, ollama_host_url: str) -> List[Tuple[str, Optional[Dict]]]:
    headers = {'Content-Type': 'application/json'}
    results = []  # To store processing results

    for filename in os.listdir(folder_path):
        # Skip files with the '.original' suffix
        if filename.endswith('.original'):
            continue

        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)

            # Check if UserComment exists
            if has_user_comment(image_path):
                print(f"Skipping {filename} due to existing UserComment.")
                continue

            with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            payload = {
                "model": "llama3.2-vision",
                "stream": False,
                "images": [encoded_image],
                "prompt": "You are a discerning photography critic. You specialize in assessing individual professional photographs. Your evaluation is objective and quantitative, focusing on technical quality, creativity, aesthetic appeal and monetization potential. You provide a numerical score on a scale of 1 to 100 for each image you see. Technical quality is evaluated based on exposure, focus, color balance, composition, crop and aspect ratio. Creativity is assessed considering the images unique perspective, mood, subject matter and emotional impact. Aesthetic appeal is assessed based on how appealing the image is in purely aesthetic terms to a general audience. You will weight aesthetic appeal double in your scoring. For each image, return well-formatted JSON adhering to the following pattern:\n\"score\": <int>. You will also return a descriptive title based on the image using no more than 60 characters, a description of the image using no more than 200 characters and up to 12 relevant keyword tags, comma separated, without hashtags",
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

            try:
                response = requests.post(ollama_host_url, json=payload, headers=headers)
                if response.status_code == 200:
                    print(f"Response for {filename}: {response.text}")
                    response_data = response.json()  # Get the full response

                    # Extract and parse the 'response' field as JSON
                    response_metadata = json.loads(response_data['response'])  # This should already be a dict

                    # Print the metadata for debugging
                    print(f"Metadata received for {filename}: {response_metadata}")

                    # Proceed to embed metadata regardless of type checks
                    try:
                        embed_metadata(image_path, response_metadata)
                        results.append((filename, response_metadata))  # Store successful result
                    except Exception as e:
                        logger.error(f"Error embedding metadata for {filename}: {e}")
                        results.append((filename, None))  # Store failure result

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
        sys.exit(1)
    if not os.path.isdir(args.folder_path):
        logger.error(f"The path '{args.folder_path}' is not a directory.")
        sys.exit(1)
    if not any(filename.endswith(('.jpg', '.jpeg', '.png')) for filename in os.listdir(args.folder_path)):
        logger.error(f"No image files found in the directory '{args.folder_path}'.")
        sys.exit(1)

    results = process_images_in_folder(args.folder_path, args.ollama_host_url)  # Pass URL to function

    # Count successful and failed image processing
    success_count = sum(1 for _, result in results if result is not None)
    failure_count = len(results) - success_count

    print(f"Successfully processed images: {success_count}")
    print(f"Failed to process images: {failure_count}")
