"""
Unified profile configuration for context-aware image evaluation.

Each profile defines:
- name: Human-readable profile name
- model_weights: Z-score weights for PyIQA metrics
- rules: Technical thresholds for penalties/bonuses
- post_process: Post-processing potential parameters
- noise: Noise thresholds (warn, high, penalties)
"""

PROFILE_CONFIG = {
    # 1. Stock / Product
    "stock_product": {
        "name": "Stock/Product Photography",
        "model_weights": {
            "clipiqa_z": 0.20,
            "laion_aes_z": 0.25,
            "musiq_ava_z": 0.25,
            "maniqa_z": 0.50,
            "musiq_paq2piq_z": 0.35,
            "pyiqa_diff_z": 0.40,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 60.0,
                "critical_threshold": 30.0,
                "soft_penalty": 10.0,
                "critical_penalty": 30.0,
            },
            "clipping": {
                "warn_pct": 5.0,
                "hard_pct": 20.0,
                "warn_penalty": 8.0,
                "hard_penalty": 16.0,
                "bonus_pct": 0.5,
                "bonus_points": 5.0,
            },
            "color_cast": {
                "threshold": 15.0,
                "penalty": 8.0,
            },
            "brightness": {
                "min_ok": 200.0,
                "max_ok": 245.0,
                "mild_penalty": 8.0,
                "strong_penalty": 16.0,
            },
            "noise": {
                "warn": 30.0,
                "high": 60.0,
                "warn_penalty": 8.0,
                "high_penalty": 18.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 35.0,
            "sharpness_soft": 60.0,
        },
    },

    # 2. Macro / Food
    "macro_food": {
        "name": "Macro/Food Photography",
        "model_weights": {
            "clipiqa_z": 0.15,
            "laion_aes_z": 0.20,
            "musiq_ava_z": 0.25,
            "maniqa_z": 0.55,
            "musiq_paq2piq_z": 0.30,
            "pyiqa_diff_z": 0.50,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 65.0,
                "critical_threshold": 35.0,
                "soft_penalty": 10.0,
                "critical_penalty": 30.0,
            },
            "clipping": {
                "warn_pct": 3.0,
                "hard_pct": 15.0,
                "warn_penalty": 8.0,
                "hard_penalty": 16.0,
            },
            "color_cast": {
                "threshold": 18.0,
                "penalty": 6.0,
            },
            "brightness": {
                "min_ok": 180.0,
                "max_ok": 240.0,
                "mild_penalty": 6.0,
                "strong_penalty": 15.0,
            },
            "noise": {
                "warn": 30.0,
                "high": 55.0,
                "warn_penalty": 8.0,
                "high_penalty": 16.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 40.0,
            "sharpness_soft": 65.0,
        },
    },

    # 3. Portrait (Neutral / Studio)
    "portrait_neutral": {
        "name": "Portrait (Neutral/Studio)",
        "model_weights": {
            "clipiqa_z": 0.45,
            "laion_aes_z": 0.45,
            "musiq_ava_z": 0.35,
            "maniqa_z": 0.35,
            "musiq_paq2piq_z": 0.25,
            "pyiqa_diff_z": 0.25,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 55.0,
                "critical_threshold": 25.0,
                "soft_penalty": 12.0,
                "critical_penalty": 32.0,
            },
            "clipping": {
                "warn_pct": 6.0,
                "hard_pct": 25.0,
                "warn_penalty": 6.0,
                "hard_penalty": 14.0,
                "bonus_pct": 1.0,
                "bonus_points": 3.0,
            },
            "color_cast": {
                "threshold": 12.0,
                "penalty": 10.0,
            },
            "brightness": {
                "min_ok": 100.0,
                "max_ok": 200.0,
                "mild_penalty": 6.0,
                "strong_penalty": 14.0,
            },
            "noise": {
                "warn": 35.0,
                "high": 65.0,
                "warn_penalty": 8.0,
                "high_penalty": 16.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 30.0,
            "sharpness_soft": 55.0,
        },
    },

    # 4. High-Key Portrait
    "portrait_highkey": {
        "name": "High-Key Portrait",
        "model_weights": {
            "clipiqa_z": 0.50,
            "laion_aes_z": 0.55,
            "musiq_ava_z": 0.40,
            "maniqa_z": 0.20,
            "musiq_paq2piq_z": 0.15,
            "pyiqa_diff_z": 0.15,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 50.0,
                "critical_threshold": 20.0,
                "soft_penalty": 8.0,
                "critical_penalty": 24.0,
            },
            "clipping": {
                "warn_pct": 12.0,
                "hard_pct": 40.0,
                "warn_penalty": 4.0,
                "hard_penalty": 12.0,
                "bonus_pct": 2.0,
                "bonus_points": 2.0,
            },
            "color_cast": {
                "threshold": 18.0,
                "penalty": 4.0,
            },
            "brightness": {
                "min_ok": 180.0,
                "max_ok": 255.0,
                "mild_penalty": 4.0,
                "strong_penalty": 10.0,
            },
            "noise": {
                "warn": 40.0,
                "high": 70.0,
                "warn_penalty": 6.0,
                "high_penalty": 12.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 25.0,
            "sharpness_soft": 50.0,
        },
    },

    # 5. Landscape / Nature
    "landscape": {
        "name": "Landscape/Nature",
        "model_weights": {
            "clipiqa_z": 0.35,
            "laion_aes_z": 0.35,
            "musiq_ava_z": 0.35,
            "maniqa_z": 0.40,
            "musiq_paq2piq_z": 0.30,
            "pyiqa_diff_z": 0.30,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 60.0,
                "critical_threshold": 30.0,
                "soft_penalty": 10.0,
                "critical_penalty": 28.0,
            },
            "clipping": {
                "warn_pct": 4.0,
                "hard_pct": 20.0,
                "warn_penalty": 8.0,
                "hard_penalty": 16.0,
                "bonus_pct": 0.5,
                "bonus_points": 4.0,
            },
            "color_cast": {
                "threshold": 18.0,
                "penalty": 5.0,
            },
            "brightness": {
                "min_ok": 100.0,
                "max_ok": 200.0,
                "mild_penalty": 6.0,
                "strong_penalty": 14.0,
            },
            "noise": {
                "warn": 30.0,
                "high": 60.0,
                "warn_penalty": 8.0,
                "high_penalty": 16.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 35.0,
            "sharpness_soft": 60.0,
        },
    },

    # 6. Street / Documentary
    "street_documentary": {
        "name": "Street/Documentary",
        "model_weights": {
            "clipiqa_z": 0.45,
            "laion_aes_z": 0.40,
            "musiq_ava_z": 0.35,
            "maniqa_z": 0.25,
            "musiq_paq2piq_z": 0.20,
            "pyiqa_diff_z": 0.20,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 45.0,
                "critical_threshold": 20.0,
                "soft_penalty": 6.0,
                "critical_penalty": 18.0,
            },
            "clipping": {
                "warn_pct": 10.0,
                "hard_pct": 45.0,
                "warn_penalty": 4.0,
                "hard_penalty": 10.0,
                "bonus_pct": 2.0,
                "bonus_points": 2.0,
            },
            "color_cast": {
                "threshold": 22.0,
                "penalty": 4.0,
            },
            "brightness": {
                "min_ok": 50.0,
                "max_ok": 200.0,
                "mild_penalty": 4.0,
                "strong_penalty": 10.0,
            },
            "noise": {
                "warn": 40.0,
                "high": 75.0,
                "warn_penalty": 4.0,
                "high_penalty": 10.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 25.0,
            "sharpness_soft": 45.0,
        },
    },

    # 7. Sports / Action / Wildlife
    "sports_action": {
        "name": "Sports/Action/Wildlife",
        "model_weights": {
            "clipiqa_z": 0.30,
            "laion_aes_z": 0.30,
            "musiq_ava_z": 0.25,
            "maniqa_z": 0.45,
            "musiq_paq2piq_z": 0.35,
            "pyiqa_diff_z": 0.40,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 55.0,
                "critical_threshold": 25.0,
                "soft_penalty": 10.0,
                "critical_penalty": 28.0,
            },
            "clipping": {
                "warn_pct": 8.0,
                "hard_pct": 35.0,
                "warn_penalty": 6.0,
                "hard_penalty": 14.0,
                "bonus_pct": 1.0,
                "bonus_points": 3.0,
            },
            "color_cast": {
                "threshold": 20.0,
                "penalty": 6.0,
            },
            "brightness": {
                "min_ok": 80.0,
                "max_ok": 220.0,
                "mild_penalty": 5.0,
                "strong_penalty": 12.0,
            },
            "noise": {
                "warn": 40.0,
                "high": 70.0,
                "warn_penalty": 6.0,
                "high_penalty": 12.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 30.0,
            "sharpness_soft": 55.0,
        },
    },

    # 8. Concert / Night / City at Night
    "concert_night": {
        "name": "Concert/Night/Low-Light",
        "model_weights": {
            "clipiqa_z": 0.50,
            "laion_aes_z": 0.50,
            "musiq_ava_z": 0.40,
            "maniqa_z": 0.20,
            "musiq_paq2piq_z": 0.15,
            "pyiqa_diff_z": 0.20,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 40.0,
                "critical_threshold": 15.0,
                "soft_penalty": 4.0,
                "critical_penalty": 12.0,
            },
            "clipping": {
                "warn_pct": 20.0,
                "hard_pct": 70.0,
                "warn_penalty": 2.0,
                "hard_penalty": 8.0,
                "bonus_pct": 3.0,
                "bonus_points": 2.0,
            },
            "color_cast": {
                "threshold": 30.0,
                "penalty": 0.0,
            },
            "brightness": {
                "min_ok": 20.0,
                "max_ok": 120.0,
                "mild_penalty": 3.0,
                "strong_penalty": 8.0,
            },
            "noise": {
                "warn": 50.0,
                "high": 80.0,
                "warn_penalty": 4.0,
                "high_penalty": 10.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 20.0,
            "sharpness_soft": 40.0,
        },
    },

    # 9. Architecture / Real Estate
    "architecture_realestate": {
        "name": "Architecture/Real Estate",
        "model_weights": {
            "clipiqa_z": 0.30,
            "laion_aes_z": 0.30,
            "musiq_ava_z": 0.30,
            "maniqa_z": 0.50,
            "musiq_paq2piq_z": 0.40,
            "pyiqa_diff_z": 0.40,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 65.0,
                "critical_threshold": 30.0,
                "soft_penalty": 10.0,
                "critical_penalty": 30.0,
            },
            "clipping": {
                "warn_pct": 4.0,
                "hard_pct": 20.0,
                "warn_penalty": 8.0,
                "hard_penalty": 16.0,
                "bonus_pct": 0.5,
                "bonus_points": 4.0,
            },
            "color_cast": {
                "threshold": 15.0,
                "penalty": 8.0,
            },
            "brightness": {
                "min_ok": 120.0,
                "max_ok": 220.0,
                "mild_penalty": 6.0,
                "strong_penalty": 14.0,
            },
            "noise": {
                "warn": 30.0,
                "high": 55.0,
                "warn_penalty": 8.0,
                "high_penalty": 18.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 35.0,
            "sharpness_soft": 65.0,
        },
    },

    # 10. Fine Art / Creative / Experimental
    "fineart_creative": {
        "name": "Fine Art/Creative/Experimental",
        "model_weights": {
            "clipiqa_z": 0.60,
            "laion_aes_z": 0.60,
            "musiq_ava_z": 0.40,
            "maniqa_z": 0.05,
            "musiq_paq2piq_z": 0.05,
            "pyiqa_diff_z": 0.10,
        },
        "rules": {
            "sharpness": {
                "soft_threshold": 25.0,
                "critical_threshold": 10.0,
                "soft_penalty": 2.0,
                "critical_penalty": 10.0,
            },
            "clipping": {
                "warn_pct": 25.0,
                "hard_pct": 90.0,
                "warn_penalty": 2.0,
                "hard_penalty": 8.0,
                "bonus_pct": 5.0,
                "bonus_points": 2.0,
            },
            "color_cast": {
                "threshold": 25.0,
                "penalty": 0.0,
            },
            "brightness": {
                "min_ok": 0.0,
                "max_ok": 255.0,
                "mild_penalty": 0.0,
                "strong_penalty": 0.0,
            },
            "noise": {
                "warn": 60.0,
                "high": 90.0,
                "warn_penalty": 2.0,
                "high_penalty": 6.0,
            },
        },
        "post_process": {
            "base": 70,
            "sharpness_heavy": 15.0,
            "sharpness_soft": 25.0,
        },
    },
}


def get_profile(context: str) -> dict:
    """Get profile config with fallback to stock_product."""
    return PROFILE_CONFIG.get(context, PROFILE_CONFIG["stock_product"])


def get_profile_name(context: str) -> str:
    """Get human-readable profile name."""
    return get_profile(context).get("name", context.replace("_", " ").title())


# =============================================================================
# PyIQA Calibration Parameters
# =============================================================================
# Baseline statistics derived from a reference dataset of professional images.
# Used to compute z-scores for profile-weighted scoring.

PYIQA_BASELINE_STATS = {
    "clipiqa_z": {"mean": 70.40125, "std": 9.290555155864835},
    "laion_aes_z": {"mean": 57.107083333333335, "std": 3.5580589033519696},
    "musiq_ava_z": {"mean": 56.75416666666667, "std": 4.285875947678478},
    "maniqa_z": {"mean": 49.132083333333334, "std": 8.258054895659281},
    "musiq_paq2piq_z": {"mean": 74.73625, "std": 3.1990914237483117},
}

# Scoring transformation parameters
PROFILE_SCORE_CENTER = 70.0  # Base score before adjustments
PROFILE_SCORE_STD_SCALE = 12.0  # Points per standard deviation

# Default shift values for PyIQA models to align with 0-100 scale
DEFAULT_PYIQA_SHIFTS = {
    "clipiqa+_vitl14_512": 14.0,
    "maniqa": 14.0,
    "maniqa-kadid": 14.0,
    "maniqa-pipal": 14.0,
}


def get_baseline_stats(metric_key: str) -> dict:
    """Get baseline mean/std for a metric, with safe fallback."""
    return PYIQA_BASELINE_STATS.get(metric_key, {"mean": 50.0, "std": 10.0})


def get_default_pyiqa_shift(model_name: str) -> float:
    """Get default score shift for a PyIQA model."""
    return DEFAULT_PYIQA_SHIFTS.get(model_name.lower(), 0.0)

