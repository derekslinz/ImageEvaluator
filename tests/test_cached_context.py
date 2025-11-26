import os
import sys

import piexif
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from image_eval_embed import CONTEXT_PROFILE_MAP, PROFILE_CONFIG, read_cached_context


def _make_image_with_description(tmp_path, description: str):
    img = Image.new("RGB", (1, 1), (255, 255, 255))
    exif_dict = {
        "0th": {piexif.ImageIFD.ImageDescription: description},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "Interop": {},
    }
    exif_bytes = piexif.dump(exif_dict)
    path = tmp_path / "with_description.jpg"
    img.save(path, exif=exif_bytes)
    return path


def test_cached_context_ignores_unknown_description(tmp_path):
    image_path = _make_image_with_description(tmp_path, "OLYMPUS DIGITAL CAMERA")
    assert read_cached_context(str(image_path)) is None


def test_cached_context_uses_known_profile(tmp_path):
    image_path = _make_image_with_description(tmp_path, "landscape")
    assert read_cached_context(str(image_path)) == "landscape"


def test_cached_context_uses_alias_mapping(tmp_path):
    alias = "travel"
    assert alias in CONTEXT_PROFILE_MAP  # guard alias presence
    mapped = CONTEXT_PROFILE_MAP[alias]
    assert mapped in PROFILE_CONFIG  # ensure mapping is valid

    image_path = _make_image_with_description(tmp_path, alias)
    assert read_cached_context(str(image_path)) == mapped
