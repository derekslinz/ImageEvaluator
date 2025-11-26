"""
Profile configuration for image quality evaluation.

This module defines evaluation profiles for different photography genres.
Each profile contains model weights and rule thresholds tuned for specific use cases.

Model Weights Note:
    The model_weights in each profile are relative importance weights, NOT probability
    distributions. They do NOT need to sum to 1.0. During scoring, these weights are
    applied to z-scored model outputs and combined. Higher weights mean that model's
    opinion matters more for that profile. The weights are normalized internally
    when computing final scores.
    
    For example, in "macro_food", maniqa_z has weight 0.55 while clipiqa_z has 0.15,
    meaning technical quality (maniqa) is ~3.7x more important than semantic quality
    (clipiqa) for food photography evaluation.

Available Profiles:
    - stock_product: Commercial/stock photography
    - macro_food: Close-up food photography
    - portrait_neutral: Studio/neutral portraits
    - portrait_highkey: Bright, airy portraits
    - landscape: Landscape/nature photography
    - street_documentary: Street/documentary style
    - sports_action: Sports/action/wildlife
    - concert_night: Low-light/concert photography
    - architecture_realestate: Architecture/real estate
    - fineart_creative: Fine art/experimental
"""

from typing import Dict, Any, Optional, List


def get_available_profiles() -> List[str]:
    """Return a list of available profile names.
    
    Returns:
        List of valid profile name strings.
    """
    return list(PROFILE_CONFIG.keys())


def get_profile(name: str) -> Optional[Dict[str, Any]]:
    """Get a profile configuration by name.
    
    Args:
        name: The profile name (e.g., 'stock_product', 'landscape').
        
    Returns:
        The profile configuration dict if found, None otherwise.
    """
    return PROFILE_CONFIG.get(name)


def validate_profile_name(name: str) -> bool:
    """Check if a profile name is valid.
    
    Args:
        name: The profile name to validate.
        
    Returns:
        True if the profile name exists, False otherwise.
    """
    return name in PROFILE_CONFIG


PROFILE_CONFIG: Dict[str, Dict[str, Any]] = {
    # 1. Stock / Product
    "stock_product": {
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
        },
    },

    # 2. Macro / Food
    "macro_food": {
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
        },
    },

    # 3. Portrait (Neutral / Studio)
    "portrait_neutral": {
        "model_weights": {
            # Balanced; aesthetics and technical both matter
            "clipiqa_z": 0.45,
            "laion_aes_z": 0.45,
            "musiq_ava_z": 0.35,
            "maniqa_z": 0.35,
            "musiq_paq2piq_z": 0.25,
            "pyiqa_diff_z": 0.25,
        },
        "rules": {
            "sharpness": {
                # High sharpness for face/eyes
                "soft_threshold": 55.0,
                "critical_threshold": 25.0,
                "soft_penalty": 12.0,
                "critical_penalty": 32.0,
            },
            "clipping": {
                # Moderate clipping tolerance
                "warn_pct": 6.0,
                "hard_pct": 25.0,
                "warn_penalty": 6.0,
                "hard_penalty": 14.0,
            },
            "color_cast": {
                # Sensitive to skin tone casts
                "threshold": 12.0,
                "penalty": 10.0,
            },
            "brightness": {
                # Medium brightness
                "min_ok": 100.0,
                "max_ok": 200.0,
                "mild_penalty": 6.0,
                "strong_penalty": 14.0,
            },
        },
    },

    # 4. High-Key Portrait
    "portrait_highkey": {
        "model_weights": {
            # More weight on aesthetic models
            "clipiqa_z": 0.50,
            "laion_aes_z": 0.55,
            "musiq_ava_z": 0.40,
            "maniqa_z": 0.20,
            "musiq_paq2piq_z": 0.15,
            "pyiqa_diff_z": 0.15,
        },
        "rules": {
            "sharpness": {
                # Slightly more relaxed than neutral portrait
                "soft_threshold": 50.0,
                "critical_threshold": 20.0,
                "soft_penalty": 8.0,
                "critical_penalty": 24.0,
            },
            "clipping": {
                # Very tolerant of highlight clipping
                "warn_pct": 12.0,
                "hard_pct": 40.0,
                "warn_penalty": 4.0,
                "hard_penalty": 12.0,
            },
            "color_cast": {
                # Intentional warm bias accepted
                "threshold": 18.0,
                "penalty": 4.0,
            },
            "brightness": {
                # Very bright
                "min_ok": 180.0,
                "max_ok": 255.0,
                "mild_penalty": 4.0,
                "strong_penalty": 10.0,
            },
        },
    },

    # 5. Landscape / Nature
    "landscape": {
        "model_weights": {
            # Mix of aesthetics and technical
            "clipiqa_z": 0.35,
            "laion_aes_z": 0.35,
            "musiq_ava_z": 0.35,
            "maniqa_z": 0.40,
            "musiq_paq2piq_z": 0.30,
            "pyiqa_diff_z": 0.30,
        },
        "rules": {
            "sharpness": {
                # High overall sharpness
                "soft_threshold": 60.0,
                "critical_threshold": 30.0,
                "soft_penalty": 10.0,
                "critical_penalty": 28.0,
            },
            "clipping": {
                # Tight clipping control with a bonus for very clean highlights
                "warn_pct": 4.0,
                "hard_pct": 20.0,
                "warn_penalty": 8.0,
                "hard_penalty": 16.0,
                "bonus_pct": 0.5,
                "bonus_points": 4.0,
            },
            "color_cast": {
                # Some tolerance (golden/blue hour)
                "threshold": 18.0,
                "penalty": 5.0,
            },
            "brightness": {
                # Medium brightness
                "min_ok": 100.0,
                "max_ok": 200.0,
                "mild_penalty": 6.0,
                "strong_penalty": 14.0,
            },
        },
    },

    # 6. Street / Documentary
    "street_documentary": {
        "model_weights": {
            # Story/aesthetic > pure technical perfection
            "clipiqa_z": 0.45,
            "laion_aes_z": 0.40,
            "musiq_ava_z": 0.35,
            "maniqa_z": 0.25,
            "musiq_paq2piq_z": 0.20,
            "pyiqa_diff_z": 0.20,
        },
        "rules": {
            "sharpness": {
                # Tolerant of motion blur
                "soft_threshold": 45.0,
                "critical_threshold": 20.0,
                "soft_penalty": 6.0,
                "critical_penalty": 18.0,
            },
            "clipping": {
                # High clipping tolerance for contrasty scenes
                "warn_pct": 10.0,
                "hard_pct": 45.0,
                "warn_penalty": 4.0,
                "hard_penalty": 10.0,
            },
            "color_cast": {
                # Varied lighting moods accepted
                "threshold": 22.0,
                "penalty": 4.0,
            },
            "brightness": {
                # Wide brightness range
                "min_ok": 50.0,
                "max_ok": 200.0,
                "mild_penalty": 4.0,
                "strong_penalty": 10.0,
            },
        },
    },

    # 7. Sports / Action / Wildlife
    "sports_action": {
        "model_weights": {
            # Technical sharpness and micro-contrast matter
            "clipiqa_z": 0.30,
            "laion_aes_z": 0.30,
            "musiq_ava_z": 0.25,
            "maniqa_z": 0.45,
            "musiq_paq2piq_z": 0.35,
            "pyiqa_diff_z": 0.40,
        },
        "rules": {
            "sharpness": {
                # Sharp subject preferred
                "soft_threshold": 55.0,
                "critical_threshold": 25.0,
                "soft_penalty": 10.0,
                "critical_penalty": 28.0,
            },
            "clipping": {
                # Moderate clipping tolerance for stadium lights, etc.
                "warn_pct": 8.0,
                "hard_pct": 35.0,
                "warn_penalty": 6.0,
                "hard_penalty": 14.0,
            },
            "color_cast": {
                # Accepts messy lighting
                "threshold": 20.0,
                "penalty": 6.0,
            },
            "brightness": {
                # Wide brightness range
                "min_ok": 80.0,
                "max_ok": 220.0,
                "mild_penalty": 5.0,
                "strong_penalty": 12.0,
            },
        },
    },

    # 8. Concert / Night / Low-Light
    "concert_night": {
        "model_weights": {
            # Vibe/aesthetic heavy, technical less important
            "clipiqa_z": 0.50,
            "laion_aes_z": 0.50,
            "musiq_ava_z": 0.40,
            "maniqa_z": 0.20,
            "musiq_paq2piq_z": 0.15,
            "pyiqa_diff_z": 0.20,
        },
        "rules": {
            "sharpness": {
                # Very tolerant of blur
                "soft_threshold": 40.0,
                "critical_threshold": 15.0,
                "soft_penalty": 4.0,
                "critical_penalty": 12.0,
            },
            "clipping": {
                # Huge clipping tolerance for point lights
                "warn_pct": 20.0,
                "hard_pct": 70.0,
                "warn_penalty": 2.0,
                "hard_penalty": 8.0,
            },
            "color_cast": {
                # Color casts fully accepted
                "threshold": 30.0,
                "penalty": 0.0,
            },
            "brightness": {
                # Dark images
                "min_ok": 20.0,
                "max_ok": 120.0,
                "mild_penalty": 3.0,
                "strong_penalty": 8.0,
            },
        },
    },

    # 9. Architecture / Real Estate
    "architecture_realestate": {
        "model_weights": {
            # Very technical: edges, detail, neutral color
            "clipiqa_z": 0.30,
            "laion_aes_z": 0.30,
            "musiq_ava_z": 0.30,
            "maniqa_z": 0.50,
            "musiq_paq2piq_z": 0.40,
            "pyiqa_diff_z": 0.40,
        },
        "rules": {
            "sharpness": {
                # High sharpness for edges/details
                "soft_threshold": 65.0,
                "critical_threshold": 30.0,
                "soft_penalty": 10.0,
                "critical_penalty": 30.0,
            },
            "clipping": {
                # Moderate clipping; small bright windows OK
                "warn_pct": 4.0,
                "hard_pct": 20.0,
                "warn_penalty": 8.0,
                "hard_penalty": 16.0,
            },
            "color_cast": {
                # Color neutrality important
                "threshold": 15.0,
                "penalty": 8.0,
            },
            "brightness": {
                # Medium-bright
                "min_ok": 120.0,
                "max_ok": 220.0,
                "mild_penalty": 6.0,
                "strong_penalty": 14.0,
            },
        },
    },

    # 10. Fine Art / Creative / Experimental
    "fineart_creative": {
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
        },
    },
}