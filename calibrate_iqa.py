"""
Utility for generating PyIQA calibration statistics from a scored reference set.

The calibration produced here is intended to be stable; avoid overwriting the
output unless you deliberately want to recompute baselines from a new reference
set.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

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
