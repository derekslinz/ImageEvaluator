import os
import sys

import pytest
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from image_eval_embed import _decode_exif_string


def test_decode_exif_string_handles_utf8_bytes():
    assert _decode_exif_string("café".encode("utf-8")) == "café"


def test_decode_exif_string_strips_null_bytes_from_string():
    assert _decode_exif_string("hello\x00") == "hello"


def test_decode_exif_string_converts_int_and_none():
    assert _decode_exif_string(123) == "123"
    assert _decode_exif_string(None) == ""


def test_decode_exif_string_ignores_invalid_utf8_bytes():
    # Leading invalid bytes should be ignored while preserving valid trailing content
    assert _decode_exif_string(b"\xff\xfecontext") == "context"
