import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from image_eval_embed import (
    assess_technical_metrics,
    categorize_score,
    compute_disagreement_z,
    compute_metric_z_scores,
    compute_post_process_potential,
    map_context_to_profile,
    normalize_weights,
    prompt_for_image_folder,
    validate_score,
)


def test_map_context_to_profile_alias_and_unknown():
    assert map_context_to_profile("Travel") == "landscape"
    assert map_context_to_profile("street_documentary") == "street_documentary"
    assert map_context_to_profile("made_up_label") == "stock_product"


def test_normalize_weights_handles_negative_and_zero_totals():
    normalized = normalize_weights({"a": 2, "b": -3, "c": 0})
    assert normalized["a"] == pytest.approx(1.0)
    assert normalized["b"] == pytest.approx(0.0)
    assert normalized["c"] == pytest.approx(0.0)

    balanced = normalize_weights({"x": 0, "y": 0})
    assert balanced["x"] == pytest.approx(0.5)
    assert balanced["y"] == pytest.approx(0.5)


def test_compute_metric_z_scores_known_and_unknown_metrics():
    z_scores = compute_metric_z_scores({
        "clipiqa_z": 80.40125,
        "unknown_metric": 10,
    })
    expected = (80.40125 - 70.40125) / 9.290555155864835
    assert z_scores["clipiqa_z"] == pytest.approx(expected)
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
    warnings = assess_technical_metrics(metrics, context="stock_product")
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
    score = compute_post_process_potential(metrics, context="stock_product")
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
