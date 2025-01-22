import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict

import piexif
import piexif.helper
from PIL import Image
from ollama import Client
from pydantic import BaseModel, field_validator, ValidationError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
client = Client(host='http://10.0.1.50:11434')


class ImageScore(BaseModel):
    total_score: int
    technical_quality: int
    creative_appeal: int
    monetization_potential: int
    maximum_potential_score: int

    @field_validator('technical_quality', 'creative_appeal', 'monetization_potential', 'total_score',
                     'maximum_potential_score')
    def check_score_range(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('Score must be between 1 and 100')
        return v


class ImageProcessor:
    def __init__(self, model_name: str = 'llama3.2-vision'):
        self.model_name = model_name
        self.ollama_host = os.getenv('OLLAMA_HOST', '127.0.0.1:11434')  # Provide a default if needed

    async def _embed_metadata(self, image_path: Path, scores: Dict) -> None:
        try:
            with Image.open(image_path) as img:
                exif_dict = piexif.load(img.info.get("exif", b""))
                scores_str = json.dumps(scores, indent=4, ensure_ascii=False)

                exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(scores_str)
                exif_bytes = piexif.dump(exif_dict)
                img.save(image_path, exif=exif_bytes)

            logger.info(f"Scores successfully embedded in the UserComment field of image: {image_path}")

            green_text = f"\033[92m{scores_str}\033[0m"
            print(f"Embedded values for {image_path}:\n{green_text}")

        except Exception as e:
            logger.error(f"Error embedding metadata for image {image_path}: {str(e)}")
            raise

    async def _get_image_score(self, image_path: Path) -> ImageScore:
        try:
            prompt = {
                "model": "llama3.2-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Evaluate this image and assign a numerical score between 1-100 composed of sub scores "
                            "for technical quality, creative appeal, and monetization potential. For each image, "
                            "return well-formatted JSON adhering to the following pattern:\n\"total_score\": 82,\n"
                            "\"technical_quality\": 85,\n\"creative_appeal\": 80,\n\"monetization_potential\": 78,\n"
                            "\"maximum_potential_score\": 90\nNormalize all scores to a 100-point scale and respond "
                            "with only the score values and score names. Do not include any additional text."
                        ),
                        "images": [str(image_path)]
                    }
                ],
                "format": {
                    "type": "object",
                    "properties": {
                        "total_score": {"type": "integer"},
                        "technical_quality": {"type": "integer"},
                        "creative_appeal": {"type": "integer"},
                        "monetization_potential": {"type": "integer"},
                        "maximum_potential_score": {"type": "integer"}
                    },
                    "required": [
                        "total_score",
                        "technical_quality",
                        "creative_appeal",
                        "monetization_potential",
                        "maximum_potential_score"
                    ]
                }
            }

            response = await self._query_ollama(json.dumps(prompt), image_path)
            print(f"Raw response from Ollama: {response}")

            return self.parse_response(response)

        except ValidationError as ve:
            logger.error(f"Validation error while processing image {image_path}: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error evaluating image score for image {image_path}: {str(e)}")
            raise

    def parse_response(self, response: str) -> ImageScore:
        try:
            data = json.loads(response)
            scores = {
                "technical_quality": data["technical_quality"],
                "creative_appeal": data["creative_appeal"],
                "monetization_potential": data["monetization_potential"],
                "maximum_potential_score": data.get("maximum_potential_score", 100),
                "total_score": data["total_score"]
            }
        except (json.JSONDecodeError, KeyError):
            scores = {}
            patterns = {
                "technical_quality": r"Technical Quality.*?:?\s*(\d+)",
                "creative_appeal": r"Creative Appeal.*?:?\s*(\d+)",
                "monetization_potential": r"Monetization Potential.*?:?\s*(\d+)",
                "maximum_potential_score": r"Maximum Potential Score.*?(\d+)",
                "total_score": r"Total Score.*?(\d+)|Final Score.*?(\d+)"
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    scores[key] = int(match.group(1))
                else:
                    json_match = re.search(r'\"' + key + r'\"\s*:\s*(\d+)', response)
                    if json_match:
                        scores[key] = int(json_match.group(1))
                    else:
                        raise ValueError(f"Could not find {key} in the response.")

        return ImageScore(**scores)

    async def _query_ollama(self, prompt: str, image_path: Path) -> str:
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [str(image_path)],
                        'options': {
                            'num_gpu': 1
                        }
                    }],

                )
            )
            print(f"Raw response from Ollama: {response}")
            logger.debug(f"Raw response from Ollama: {response}")

            if not response:
                raise ValueError("Received empty response from Ollama.")

            if 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid response structure from Ollama.")

            return response['message']['content']
        except Exception as e:
            logger.error(f"Error querying Ollama for image {image_path}: {str(e)}")
            raise


async def process_images_in_folder(folder_path: Path):
    processor = ImageProcessor()
    for image_path in folder_path.glob('*.jpg'):  # Adjust the pattern for other image formats if needed
        try:
            logger.info(f"Processing image: {image_path}")
            scores = await processor._get_image_score(image_path)
            await processor._embed_metadata(image_path, scores.model_dump())
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = Path(sys.argv[1])

    if not folder_path.is_dir():
        print(f"The path {folder_path} is not a valid directory.")
        sys.exit(1)

    asyncio.run(process_images_in_folder(folder_path))
