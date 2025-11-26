import pytest

import image_eval_embed
from image_eval_embed import extract_exif_metadata


@pytest.mark.parametrize("helper_result", [
    {},
    {
        "iso": 200,
        "aperture": "f/2.8",
        "shutter_speed": "1/125s",
        "focal_length": "85mm",
        "camera_make": "Canon",
        "camera_model": "EOS R5",
        "lens_model": "RF 85mm f1.2"
    },
])
def test_extract_exif_metadata_merges_defaults(monkeypatch, helper_result):
    """extract_exif_metadata should always return the canonical key set."""
    captured_path = {}

    def fake_pil_extractor(path):
        captured_path["value"] = path
        return helper_result

    monkeypatch.setattr(
        image_eval_embed,
        "_extract_pil_exif_metadata",
        fake_pil_extractor,
    )

    result = extract_exif_metadata("/tmp/example.jpg")

    assert captured_path["value"] == "/tmp/example.jpg"
    expected_keys = {
        'iso', 'aperture', 'shutter_speed', 'focal_length',
        'camera_make', 'camera_model', 'lens_model'
    }
    assert set(result.keys()) == expected_keys

    for key in expected_keys:
        if key in helper_result:
            assert result[key] == helper_result[key]
        else:
            assert result[key] is None


def test_extract_exif_metadata_handles_helper_errors(monkeypatch):
    """If the helper raises, the top-level function should return empty defaults."""
    def fail(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(image_eval_embed, "_extract_pil_exif_metadata", fail)
    result = extract_exif_metadata("/tmp/broken.jpg")

    assert result == {
        'iso': None,
        'aperture': None,
        'shutter_speed': None,
        'focal_length': None,
        'camera_make': None,
        'camera_model': None,
        'lens_model': None
    }
