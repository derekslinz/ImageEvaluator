    import asyncio
    import json
    import logging
    from pathlib import Path
    from typing import Dict, List

    import piexif
    from PIL import Image
    from ollama import Client
    from pydantic import BaseModel

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    client = Client(host='http://10.0.1.50:11434')


    class ImageDescription(BaseModel):
        description: str


    class ImageTags(BaseModel):
        tags: List[str]


    class ImageTitle(BaseModel):
        title: str


    class ImageProcessor:
        def __init__(self, model_name: str = 'llama3.2-vision'):
            self.model_name = model_name
            self.ollama_host = os.getenv('OLLAMA_HOST', '127.0.0.1:11434')  # Provide a default if needed

        async def _embed_metadata(self, image_path: Path, title: str, description: str, tags: List[str]) -> None:
            try:
                with Image.open(image_path) as img:
                    exif_dict = piexif.load(img.info.get("exif", b""))

                    truncated_title = title[:120]
                    title_str = json.dumps(truncated_title, indent=4, ensure_ascii=False)

                    cleaned_tags = clean_tags(tags)
                    tags_str = ', '.join(cleaned_tags)

                    truncated_description = ' '.join(description.split()[:120])

                    exif_dict["0th"][piexif.ImageIFD.XPTitle] = title_str.encode('utf-8')
                    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = truncated_description.encode('utf-8')
                    exif_dict["0th"][piexif.ImageIFD.XPKeywords] = tags_str.encode('utf-8')

                    exif_bytes = piexif.dump(exif_dict)
                    img.save(image_path, exif=exif_bytes)

                logger.info(f"Metadata (description, tags, title) embedded in image: {image_path}")

                white_description = f"\033[97m{truncated_description}\033[0m"
                yellow_title = f"\033[93m{truncated_title}\033[0m"
                green_tags = f"\033[92m{tags_str}\033[0m"

                print(f"Embedded description for {image_path}:\n{white_description}")
                print(f"Embedded title for {image_path}:\n{yellow_title}")
                print(f"Embedded tags for {image_path}:\n{green_tags}")

            except Exception as e:
                logger.error(f"Error embedding metadata for {image_path}: {str(e)}")
                raise

        async def process_image(self, image_path: Path) -> Dict:
            try:
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                logger.info(f"Getting description for image: {image_path}")
                description_response = await self._get_description(image_path)
                logger.debug(f"Received description: {description_response.description}")

                logger.info(f"Getting tags for image: {image_path}")
                tags_response = await self._get_tags(image_path)
                logger.debug(f"Received tags: {tags_response.tags}")

                logger.info(f"Getting title for image: {image_path}")
                title_response = await self._get_title(image_path)
                logger.debug(f"Received title: {title_response.title}")

                await self._embed_metadata(image_path, title_response.title, description_response.description,
                                           tags_response.tags)

                return {
                    "description": description_response.description,
                    "tags": tags_response.tags,
                    "title": title_response.title if title_response.title else "",
                    "is_processed": True
                }
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                raise

        async def _get_description(self, image_path: Path) -> ImageDescription:
            try:
                response = await self._query_ollama(
                    "Describe this image in one or two sentences. Do not exceed 120 words. Respond in JSON format with a single key: 'description'.",
                    image_path
                )
                logger.debug(f"Raw description response: {response}")

                try:
                    response_data = json.loads(response)
                    if not isinstance(response_data, dict) or 'description' not in response_data:
                        raise ValueError("Invalid JSON structure.")
                except json.JSONDecodeError:
                    logger.warning(f"Received plain text instead of JSON. Converting: {response}")
                    response_data = {"description": response.strip()}

                return ImageDescription(**response_data)

            except Exception as e:
                logger.error(f"Error getting description for image {image_path}: {str(e)}")
                raise

        async def _get_tags(self, image_path: Path) -> ImageTags:
            try:
                response = await self._query_ollama(
                    "List 5-10 relevant tags for this image. Include both objects, artistic style, type of image, color, etc. Respond in JSON format with a single key: 'tags' containing a list of strings.",
                    image_path
                )
                logger.debug(f"Raw tags response: {response}")

                try:
                    response_data = json.loads(response)
                    if not isinstance(response_data, dict) or 'tags' not in response_data or not isinstance(
                            response_data['tags'], list):
                        raise ValueError("Invalid JSON structure for tags.")
                except json.JSONDecodeError:
                    logger.warning(f"Received plain text instead of JSON for tags. Converting: {response}")
                    tags_list = [tag.strip() for tag in response.split(",") if tag.strip()]
                    response_data = {"tags": tags_list}

                return ImageTags(**response_data)

            except Exception as e:
                logger.error(f"Error getting tags for image {image_path}: {str(e)}")
                raise

        async def _get_title(self, image_path: Path) -> ImageTitle:
            try:
                response = await self._query_ollama(
                    (
                        "Give this photo a descriptive title no more than 120 characters long. Return the title as"
                        "a key value pair in well formatted json: \"title\": \"Descriptive Title\""
                    ),
                    image_path
                )
                logger.debug(f"Raw title response: {response}")

                try:
                    response_data = json.loads(response)
                    if not isinstance(response_data, dict) or 'title' not in response_data:
                        raise ValueError("Invalid JSON structure for title.")
                except json.JSONDecodeError:
                    logger.warning(f"Received plain text instead of JSON for title. Converting: {response}")
                    response_data = {
                        "title": response.strip()
                    }

                # Check if the title contains the phrase 'I cannot'
                if 'I cannot' in response_data['title']:
                    response_data['title'] = 'Untitled'

                return ImageTitle(**response_data)

            except Exception as e:
                logger.error(f"Error extracting title for image {image_path}: {str(e)}")
                raise

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
                        server_ip=self.server_ip  # Use the specified server IP
                    )
                )
                logger.debug(f"Raw response from Ollama: {response}")
                if not response or 'message' not in response or 'content' not in response['message']:
                    raise ValueError("Invalid or empty response from Ollama.")
                return response['message']['content']
            except Exception as e:
                logger.error(f"Error querying Ollama: {str(e)}")
                raise


    def clean_tags(tags: list) -> list:
        seen = set()
        unique_tags = []
        for tag in tags:
            cleaned_tag = ''.join(char for char in tag if char.isalnum() or char.isspace()).strip()
            if cleaned_tag not in seen:
                seen.add(cleaned_tag)
                unique_tags.append(cleaned_tag)
        return unique_tags[:10]


    async def process_images_in_folder(folder_path: Path, server_ip: str):
        processor = ImageProcessor(server_ip=server_ip)
        image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))

        if not image_files:
            logger.error(f"No image files found in directory: {folder_path}")
            return

        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    img.verify()

                logger.info(f"Processing image: {image_path}")
                await processor.process_image(image_path)
            except (IOError, SyntaxError) as e:
                logger.error(f"File {image_path} is not a valid image: {e}")
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")


    if __name__ == "__main__":
        import sys

        if len(sys.argv) != 2:
            print("Usage: python keyword_embedder.py <folder_path>")
            sys.exit(1)

        folder_path = Path(sys.argv[1])
