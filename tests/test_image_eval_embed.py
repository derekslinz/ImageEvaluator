import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from image_eval_embed import sanitize_string, Metadata


def test_sanitize_string_removes_null_and_newlines():
    raw = "Hello\x00World\nNew\rLine"
    result = sanitize_string(raw)
    assert "\x00" not in result
    assert "\n" not in result
    assert "\r" not in result
    assert result == "HelloWorld New Line"


def test_metadata_keywords_truncated_to_twelve():
    keywords = ",".join(f"kw{i}" for i in range(15))
    md = Metadata(score=1, title="t", description="d", keywords=keywords)
    result_keywords = md.keywords.split(',')
    assert len(result_keywords) == 12
    assert result_keywords == [f"kw{i}" for i in range(12)]
