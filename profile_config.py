PROFILE_CONFIG = {
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
