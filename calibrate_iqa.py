"""
Utility for generating PyIQA calibration statistics from a scored reference set.

The calibration produced here is intended to be stable; avoid overwriting the
output unless you deliberately want to recompute baselines from a new reference
set.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

IQA_COLUMNS = [
    "clipiqa_z",
    "laion_aes_z",
    "maniqa_z",
    "musiq_ava_z",
    "musiq_paq2piq_z",
]

DEFAULT_INPUT = Path("/root/gurushots_facebook_challenge_winners/all.csv")
DEFAULT_OUTPUT = Path("iqa_calibration.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-model IQA calibration statistics (mean, std, and optional "
            "empirical percentile support) from a scored reference CSV."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Path to the scored reference CSV containing PyIQA columns. "
            f"Default: {DEFAULT_INPUT}"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the calibration JSON. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=IQA_COLUMNS,
        help=(
            "IQA columns to calibrate. Defaults to the five PyIQA metrics "
            "produced by the evaluator."
        ),
    )
    parser.add_argument(
        "--no-sorted-values",
        dest="include_sorted_values",
        action="store_false",
        help="Do not store sorted values for empirical percentile lookup.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing calibration file.",
    )
    return parser.parse_args()


def extract_iqa_from_description(description: str, columns: Iterable[str]) -> Dict[str, Optional[float]]:
    """Extract IQA scores from the description field.
    
    The description contains patterns like:
    'clipiqa_z:73.6 (z=+0.35); laion_aes_z:46.9 (z=-2.52); ...'
    
    Returns a dict mapping column name to its calibrated value (or None if not found).
    """
    result: Dict[str, Optional[float]] = {col: None for col in columns}
    
    if not isinstance(description, str):
        return result
    
    # Pattern matches "model_name:value" where value is a float
    for col in columns:
        # Match patterns like "clipiqa_z:73.6" 
        pattern = rf'{re.escape(col)}:([0-9.]+)'
        match = re.search(pattern, description)
        if match:
            try:
                result[col] = float(match.group(1))
            except ValueError:
                pass
    
    return result


def extract_iqa_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Extract IQA columns from description field if not present as direct columns.
    
    If the IQA columns exist directly in the DataFrame, return as-is.
    Otherwise, parse them from the 'description' field.
    """
    columns_list = list(columns)
    available = set(df.columns)
    
    # Check if all columns are directly available
    missing = [col for col in columns_list if col not in available]
    
    if not missing:
        # All columns present directly
        return df
    
    # Need to extract from description
    if 'description' not in df.columns:
        raise ValueError(
            f"IQA columns {missing} not found and no 'description' field to parse from."
        )
    
    print(f"Extracting IQA scores from 'description' field for: {', '.join(missing)}")
    
    # Extract IQA values from each row's description
    extracted_data = df['description'].apply(
        lambda desc: extract_iqa_from_description(desc, columns_list)
    )
    
    # Convert to DataFrame and merge
    extracted_df = pd.DataFrame(extracted_data.tolist(), index=df.index)
    
    # Only add columns that were missing
    for col in missing:
        df[col] = extracted_df[col]
    
    return df


def validate_columns(df_columns: Iterable[str], required: Iterable[str]) -> List[str]:
    available = set(df_columns)
    missing = [col for col in required if col not in available]
    return missing


def compute_calibration(
    df: pd.DataFrame, columns: Iterable[str], include_sorted_values: bool = True
) -> Dict[str, Dict[str, object]]:
    calibration: Dict[str, Dict[str, object]] = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        values = series.to_numpy(dtype=float)
        if values.size == 0:
            mu = 0.0
            sigma = 0.0
            sorted_values: List[float] = []
        else:
            mu = float(values.mean())
            sigma = float(values.std(ddof=0))
            sorted_values = np.sort(values).tolist()

        entry: Dict[str, object] = {
            "mu": mu,
            "sigma": sigma,
            "count": int(values.size),
        }
        if include_sorted_values:
            entry["sorted_values"] = sorted_values
        calibration[col] = entry
    return calibration


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser()
    output_path = args.output.expanduser()

    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Calibration file already exists at {output_path}. "
            "Use --overwrite to regenerate it."
        )

    df = pd.read_csv(input_path)

    # Try to extract IQA columns from description field if not directly available
    try:
        df = extract_iqa_columns(df, args.columns)
    except ValueError as e:
        raise SystemExit(str(e))

    # Validate that we now have all required columns
    missing_columns = validate_columns(df.columns, args.columns)
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise SystemExit(
            f"Input CSV is missing required columns: {missing_list}. "
            "Ensure the file contains the expected PyIQA outputs."
        )

    calibration = compute_calibration(
        df=df, columns=args.columns, include_sorted_values=args.include_sorted_values
    )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_csv": str(input_path),
        "row_count": int(df.shape[0]),
        "iqa_models": calibration,
    }

    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote calibration for {len(calibration)} IQA models to {output_path}")


if __name__ == "__main__":
    main()
