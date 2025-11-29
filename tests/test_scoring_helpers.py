import os
import sys
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from image_eval_embed import (
    ImageResolutionTooSmallError,
    analyze_image_technical,
    assess_technical_metrics,
    categorize_score,
    compute_disagreement_z,
    compute_metric_z_scores,
    compute_post_process_potential,
    count_technical_flags,
    is_technically_warned,
    map_context_to_profile,
    normalize_weights,
    prompt_for_image_folder,
    validate_score,
)


def test_map_context_to_profile_alias_and_unknown():
    assert map_context_to_profile("Travel") == "landscape"
    assert map_context_to_profile("street_documentary") == "street_documentary"
    assert map_context_to_profile("made_up_label") == "studio_photography"


def test_normalize_weights_handles_negative_and_zero_totals():
    normalized = normalize_weights({"a": 2, "b": -3, "c": 0})
    assert normalized["a"] == pytest.approx(1.0)
    assert normalized["b"] == pytest.approx(0.0)
    assert normalized["c"] == pytest.approx(0.0)

    balanced = normalize_weights({"x": 0, "y": 0})
    assert balanced["x"] == pytest.approx(0.5)
    assert balanced["y"] == pytest.approx(0.5)


def test_compute_metric_z_scores_known_and_unknown_metrics():
    z_scores, percentiles, fused_scores = compute_metric_z_scores({
        "clipiqa_z": 80.40125,
        "unknown_metric": 10,
    })
    expected = (80.40125 - 70.40125) / 9.290555155864835
    assert z_scores["clipiqa_z"] == pytest.approx(expected, abs=1e-6)
    assert z_scores["unknown_metric"] == 0.0


def test_compute_disagreement_z_returns_negative_range():
    z = {"clipiqa_z": 1.0, "laion_aes_z": -2.0, "maniqa_z": 0.5}
    assert compute_disagreement_z(z) == pytest.approx(-3.0)
    assert compute_disagreement_z({}) == 0.0


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (92, "elite"),
        (87, "award_ready"),
        (78, "portfolio"),
        (66, "solid"),
        (58, "needs_work"),
        (40, "critical"),
    ],
)
def test_categorize_score(score, label):
    assert categorize_score(score) == label


def test_validate_score_accepts_multiple_formats():
    assert validate_score(50) == 50
    assert validate_score("Score 72/100") == 72
    assert validate_score("200") is None
    assert validate_score("no digits here") is None


def test_assess_technical_metrics_flags_expected_conditions():
    metrics = {
        "sharpness": 20.0,
        "histogram_clipping_highlights": 8.0,
        "histogram_clipping_shadows": 6.0,
        "color_cast": "warm/red",
        "noise_score": 70.0,
    }
    warnings = assess_technical_metrics(metrics, context="studio_photography")
    assert warnings == [
        "Sharpness critically low (20.0)",
        "Highlight clipping 8.0% reduces tonal range",
        "Shadow clipping 6.0% removes shadow detail",
        "Color cast detected: warm/red",
        "High noise level (score 70.0/100)",
    ]


def test_compute_post_process_potential_combines_adjustments():
    metrics = {
        "sharpness": 70.0,  # bonus
        "histogram_clipping_highlights": 1.0,  # bonus
        "histogram_clipping_shadows": 1.0,
        "noise_score": 70.0,  # high penalty
        "color_cast": "warm/red",  # penalty
    }
    score = compute_post_process_potential(metrics, context="studio_photography")
    assert score == 57  # 70 base +5 sharpness +5 clipping -15 noise -8 color


def test_prompt_for_image_folder_accepts_user_input(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "/tmp/images")
    assert prompt_for_image_folder("/fallback") == "/tmp/images"


def test_prompt_for_image_folder_uses_default_on_empty(monkeypatch):
    responses = iter(["", "   ", "/final"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    assert prompt_for_image_folder("/fallback") == "/fallback"


def test_prompt_for_image_folder_non_interactive(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit):
        prompt_for_image_folder("/fallback")


def test_analyze_image_technical_enforces_min_long_edge(tmp_path, monkeypatch):
    image_path = tmp_path / "tiny.jpg"
    Image.new('RGB', (800, 600)).save(image_path)

    @contextmanager
    def fake_open(_):
        img = Image.open(image_path)
        try:
            yield img
        finally:
            img.close()

    monkeypatch.setattr("image_eval_embed.open_image_for_analysis", fake_open)

    with pytest.raises(ImageResolutionTooSmallError):
        analyze_image_technical(str(image_path))


def test_analyze_image_technical_reports_sharpness_and_noise(tmp_path):
    width, height = 900, 900
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    noise = np.random.default_rng(123).integers(0, 30, size=(height, width), dtype=np.uint8)
    channel = np.clip(gradient + noise, 0, 255).astype(np.uint8)
    rgb = np.stack([channel, np.flipud(channel), channel], axis=2)

    image_path = tmp_path / "textured.jpg"
    Image.fromarray(rgb, mode="RGB").save(image_path)

    metrics = analyze_image_technical(str(image_path))

    assert metrics["status"] == "success"
    assert metrics["sharpness"] > 0.0
    assert metrics["noise_score"] > 0.0


def test_analyze_image_technical_falls_back_on_degenerate_cv2(monkeypatch, tmp_path):
    width, height = 900, 900
    gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    image_path = tmp_path / "cv2_zeroed.jpg"
    Image.fromarray(np.stack([gradient, gradient, gradient], axis=2), mode="RGB").save(image_path)

    class DummyCV2:
        CV_64F = 0
        CV_32F = 0
        INTER_AREA = 0

        @staticmethod
        def Laplacian(arr, dtype):
            return np.zeros_like(arr, dtype=np.float64)

        @staticmethod
        def resize(arr, dsize=None, fx=1.0, fy=1.0, interpolation=None):
            target_h = max(1, int(arr.shape[0] * fy))
            target_w = max(1, int(arr.shape[1] * fx))
            return np.zeros((target_h, target_w), dtype=arr.dtype)

        @staticmethod
        def GaussianBlur(arr, ksize, sigmaX=0.0, sigmaY=0.0):
            return np.zeros_like(arr, dtype=np.float32)

        @staticmethod
        def Sobel(arr, ddepth, dx, dy, ksize):
            return np.zeros_like(arr, dtype=np.float32)

    fallback_calls = {"count": 0}

    def fake_fallback(gray_array):
        fallback_calls["count"] += 1
        return 12.5, 0.01, 55.0

    monkeypatch.setattr("image_eval_embed.cv2", DummyCV2())
    monkeypatch.setattr("image_eval_embed.CV2_AVAILABLE", True)
    monkeypatch.setattr("image_eval_embed._compute_sharpness_noise_fallback", fake_fallback)

    metrics = analyze_image_technical(str(image_path))

    assert fallback_calls["count"] == 1
    assert metrics["sharpness"] == pytest.approx(12.5)
    assert metrics["noise_sigma"] == pytest.approx(0.01)
    assert metrics["noise_score"] == pytest.approx(55.0)


def test_count_technical_flags_identifies_critical_and_warn():
    """Test that count_technical_flags correctly counts critical vs warn flags."""
    # No issues - clean image (sharpness above warn threshold of ~21)
    clean = {
        'sharpness': 35.0,  # Above warn threshold (~21)
        'noise_score': 10.0,
        'histogram_clipping_highlights': 1.0,
        'histogram_clipping_shadows': 2.0,
        'color_cast_delta': 5.0,
    }
    critical, warn = count_technical_flags(clean)
    assert critical == 0
    assert warn == 0
    
    # One critical sharpness issue (below ~13)
    critical_sharpness = {
        'sharpness': 10.0,  # Below critical threshold (~13)
        'noise_score': 10.0,
        'histogram_clipping_highlights': 1.0,
        'histogram_clipping_shadows': 2.0,
        'color_cast_delta': 5.0,
    }
    critical, warn = count_technical_flags(critical_sharpness)
    assert critical == 1
    assert warn == 0
    
    # One warn-level issue (sharpness between warn ~21 and critical ~13)
    warn_sharpness = {
        'sharpness': 17.0,  # Between warn (~21) and critical (~13)
        'noise_score': 10.0,
        'histogram_clipping_highlights': 1.0,
        'histogram_clipping_shadows': 2.0,
        'color_cast_delta': 5.0,
    }
    critical, warn = count_technical_flags(warn_sharpness)
    assert critical == 0
    assert warn == 1


def test_is_technically_warned_requires_critical_or_two_warns():
    """Test that is_technically_warned follows the 1 critical OR 2+ warns rule."""
    # Clean image - not warned (sharpness above warn threshold)
    clean = {
        'sharpness': 35.0,  # Above warn threshold (~21)
        'noise_score': 10.0,
        'histogram_clipping_highlights': 1.0,
        'histogram_clipping_shadows': 2.0,
        'color_cast_delta': 5.0,
    }
    assert is_technically_warned(clean) is False
    
    # One warn only - not warned
    one_warn = {
        'sharpness': 17.0,  # warn level (between 13-21)
        'noise_score': 10.0,
        'histogram_clipping_highlights': 1.0,
        'histogram_clipping_shadows': 2.0,
        'color_cast_delta': 5.0,
    }
    assert is_technically_warned(one_warn) is False
    
    # One critical - warned
    one_critical = {
        'sharpness': 10.0,  # critical level (below ~13)
        'noise_score': 10.0,
        'histogram_clipping_highlights': 1.0,
        'histogram_clipping_shadows': 2.0,
        'color_cast_delta': 5.0,
    }
    assert is_technically_warned(one_critical) is True
    
    # Two warns (different metrics) - warned
    two_warns = {
        'sharpness': 17.0,  # warn level (sharpness)
        'noise_score': 45.0,  # warn level (above ~41.6)
        'histogram_clipping_highlights': 1.0,
        'histogram_clipping_shadows': 2.0,
        'color_cast_delta': 5.0,
    }
    assert is_technically_warned(two_warns) is True
