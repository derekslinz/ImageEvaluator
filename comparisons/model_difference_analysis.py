#!/usr/bin/env python3
"""
model_difference_analysis.py

All-in-one analysis script for GuruShots top-10 images with multiple IQA models.

What it does:
- Loads 5 CSV files with model scores:
    * clipiqa+_vitL14_512_shift_14_gurushots.csv
    * laion_aes.csv
    * maniqa_calibrated.csv
    * musiq-ava.csv
    * musiq-paq2piq.csv
- Assumes all have at least:
    - file_path
    - either score or overall_score
- Uses the CLIP-IQA+ CSV as the "base" (keeps all its extra columns).
- Adds columns:
    * clipiqa, laion, maniqa, musiq_ava, musiq_paq
    * per-model z-scores: *_z
    * z_aesthetic = mean(laion_z, musiq_ava_z)
    * z_technical = mean(maniqa_z, musiq_paq_z, clipiqa_z)
    * disagreement_z = max(z) - min(z) across all five models
    * bucket = qualitative category based on the above
- Writes: gurushots_all_models_merged_labeled.csv
"""

import os
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Directory containing your CSVs.
# If the script lives in the same directory as the CSVs, this will work as-is.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "clipiqa": os.path.join(BASE_DIR, "clipiqa+_vitL14_512_shift_14_gurushots.csv"),
    "laion":   os.path.join(BASE_DIR, "laion_aes.csv"),
    "maniqa":  os.path.join(BASE_DIR, "maniqa_calibrated.csv"),
    "m_ava":   os.path.join(BASE_DIR, "musiq-ava.csv"),
    "m_paq":   os.path.join(BASE_DIR, "musiq-paq2piq.csv"),
}

OUTPUT_CSV = os.path.join(BASE_DIR, "gurushots_all_models_merged_labeled.csv")

# Bucket thresholds (z-score units)
Z_AESTH_HIGH = 1.0
Z_TECH_HIGH = 1.0
Z_AESTH_LOW = -0.5
Z_TECH_LOW = -0.5
DISAGREE_HIGH = 1.5
DISAGREE_VERY_HIGH = 2.0


# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------

def load_simple(path: str, score_col_name: str) -> pd.DataFrame:
    """
    Load a model CSV which may be either:
    - simple:  file_path, score, status
    - rich:    file_path, overall_score, technical_score, ..., etc.

    We auto-detect:
    - 'file_path' column (required)
    - score column: prefer 'score', else 'overall_score'
    """
    df = pd.read_csv(path)

    if "file_path" not in df.columns:
        raise ValueError(
            f"{path} must contain 'file_path'; got {df.columns.tolist()}"
        )

    if "score" in df.columns:
        score_col = "score"
    elif "overall_score" in df.columns:
        score_col = "overall_score"
    else:
        raise ValueError(
            f"{path} must contain either 'score' or 'overall_score'; "
            f"got {df.columns.tolist()}"
        )

    df = df[["file_path", score_col]].rename(columns={score_col: score_col_name})
    return df


def load_and_merge(paths: dict) -> pd.DataFrame:
    """
    Load all CSVs and merge into single DataFrame.
    Uses CLIP-IQA+ file as base, keeping all its columns.
    """
    # Base: CLIP-IQA+ rich table
    base = pd.read_csv(paths["clipiqa"])
    if "file_path" not in base.columns or "overall_score" not in base.columns:
        raise ValueError(
            "CLIP-IQA+ CSV must contain 'file_path' and 'overall_score'; "
            f"got {base.columns.tolist()}"
        )

    # Rename overall_score to clipiqa
    base = base.rename(columns={"overall_score": "clipiqa"})

    # Other models
    laion = load_simple(paths["laion"], "laion")
    maniqa = load_simple(paths["maniqa"], "maniqa")
    m_ava = load_simple(paths["m_ava"], "musiq_ava")
    m_paq = load_simple(paths["m_paq"], "musiq_paq")

    merged = (
        base
        .merge(laion, on="file_path", how="left")
        .merge(maniqa, on="file_path", how="left")
        .merge(m_ava, on="file_path", how="left")
        .merge(m_paq, on="file_path", how="left")
    )

    return merged


def add_z_scores(df: pd.DataFrame, score_cols) -> pd.DataFrame:
    """Add z-score columns for each score column."""
    for c in score_cols:
        mean = df[c].mean()
        std = df[c].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[f"{c}_z"] = np.nan
        else:
            df[f"{c}_z"] = (df[c] - mean) / std
    return df


def label_row(row: pd.Series) -> str:
    """Assign a qualitative bucket label to one row based on z-scores."""
    z_a = row["z_aesthetic"]
    z_t = row["z_technical"]
    d = row["disagreement_z"]

    # Consensus high: strong on both, low disagreement
    if (z_a > Z_AESTH_HIGH) and (z_t > Z_TECH_HIGH) and (d < 1.0):
        return "consensus_top"

    # Artistic but technically controversial: aesthetics high, technical weak, big disagreement
    if (z_a > Z_AESTH_HIGH) and (z_t < Z_TECH_LOW) and (d > DISAGREE_HIGH):
        return "artistic_but_technically_controversial"

    # Technically strong but aesthetically controversial
    if (z_t > Z_TECH_HIGH) and (z_a < Z_AESTH_LOW) and (d > DISAGREE_HIGH):
        return "technical_but_aesthetically_controversial"

    # Wild disagreement (no consensus at all)
    if d > DISAGREE_VERY_HIGH:
        return "wild_disagreement"

    # Everything else
    return "consensus_or_mid"


def analyze_and_label() -> pd.DataFrame:
    """End-to-end: load, merge, compute metrics, label, and save CSV."""
    print("Loading and merging CSVs...")
    df = load_and_merge(PATHS)
    print(f"Merged shape: {df.shape}")

    score_cols = ["clipiqa", "laion", "maniqa", "musiq_ava", "musiq_paq"]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected score columns in merged df: {missing}")

    print("Computing per-model z-scores...")
    df = add_z_scores(df, score_cols)
    z_cols = [c + "_z" for c in score_cols]

    print("Computing aggregated aesthetic and technical scores...")
    df["z_aesthetic"] = df[["laion_z", "musiq_ava_z"]].mean(axis=1)
    df["z_technical"] = df[["maniqa_z", "musiq_paq_z", "clipiqa_z"]].mean(axis=1)

    print("Computing disagreement metric...")
    df["disagreement_z"] = df[z_cols].max(axis=1) - df[z_cols].min(axis=1)

    print("Assigning bucket labels...")
    df["bucket"] = df.apply(label_row, axis=1)

    print(f"Saving labeled dataset to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nBucket counts:")
    print(df["bucket"].value_counts().sort_index())

    print("\nMean z-scores by bucket:")
    print(
        df.groupby("bucket")[["z_aesthetic", "z_technical", "disagreement_z"]]
        .mean()
        .round(3)
    )

    return df


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    analyze_and_label()
