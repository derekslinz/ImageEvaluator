import asyncio
import json
import logging
from pathlib import Path
from typing import Dict

import ollama
import piexif
import piexif.helper
from PIL import Image
from pydantic import BaseModel, validator, ValidationError

logger = logging.getLogger(__name__)


class ImageScore(BaseModel):
    total_score: int
    technical_quality: int
    creative_appeal: int
    monetization_potential: int
    maximum_potential_score: int

    @validator('technical_quality', 'creative_appeal', 'monetization_potential', 'total_score',
               'maximum_potential_score')
    def check_score_range(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('Score must be between 1 and 100')
        return v


class ImageProcessor:
    def __init__(self, model_name: str = 'llama3.2-vision'):
        self.model_name = model_name

    async def _embed_metadata(self, image_path: Path, scores: Dict) -> None:
        """Embed scores as metadata into the image's metadata."""
        try:
            with Image.open(image_path) as img:
                exif_dict = piexif.load(img.info.get("exif", b""))
                scores_str = json.dumps(scores, indent=4, ensure_ascii=False)

                exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(scores_str)
                exif_bytes = piexif.dump(exif_dict)
                img.save(image_path, exif=exif_bytes)

            logger.info(f"Scores successfully embedded in the UserComment field of image: {image_path}")

        except Exception as e:
            logger.error(f"Error embedding metadata for {image_path}: {str(e)}")
            raise

    async def _get_image_score(self, image_path: Path) -> ImageScore:
        """Retrieve image scores from Ollama."""
        try:
            # Updated prompt with the new structure
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

            # Query Ollama with the updated prompt
            response = await self._query_ollama(json.dumps(prompt), image_path)
            logger.debug(f"Raw score response: {response}")

            # Parse the JSON response
            response_data = json.loads(response)
            required_keys = [
                "technical_quality", "creative_appeal", "monetization_potential",
                "total_score", "maximum_potential_score"
            ]
            if not all(key in response_data for key in required_keys):
                raise ValueError("Invalid JSON structure for image score.")

            return ImageScore(**response_data)

        except ValidationError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error evaluating image score for image {image_path}: {str(e)}")
            raise

    async def _query_ollama(self, prompt: str, image_path: Path) -> str:
        """Send a query to Ollama with an image and expect structured output."""
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model=self.model_name,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [str(image_path)],
                        'options': {
                            'num_gpu': 1
                        }
                    }]
                )
            )
            logger.debug(f"Raw response from Ollama: {response}")
            if not response or 'message' not in response or 'content' not in response['message']:
                raise ValueError("Invalid or empty response from Ollama.")
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            raise
