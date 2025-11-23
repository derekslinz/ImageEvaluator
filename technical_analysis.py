#!/usr/bin/env python3
"""
Simple CLI to reuse the stock evaluator's technical analysis helper for a single file.
"""

import argparse
import json
import os
import sys

from stock_photo_evaluator import analyze_technical_quality


def main():
    parser = argparse.ArgumentParser(description="Print technical metadata for one image.")
    parser.add_argument("image_path", help="Path to the image to analyze")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    metrics = analyze_technical_quality(args.image_path)

    if args.format == "json":
        print(json.dumps(metrics, indent=2))
        return

    print(f"Technical metrics for {args.image_path}:")
    print(f"  Resolution: {metrics.get('megapixels', 0):.1f} MP ({metrics.get('dimensions')})")
    print(f"  Sharpness score: {metrics.get('sharpness', 0):.1f}/100")
    print(f"  Noise estimate: {metrics.get('noise', 0):.1f}")
    print(f"  Highlight clipping: {metrics.get('highlight_clip', 0)*100:.1f}%")
    print(f"  Shadow clipping: {metrics.get('shadow_clip', 0)*100:.1f}%")
    print(f"  DPI: {metrics.get('dpi', 0)}")

    notes = metrics.get("notes")
    if notes:
        print("  Notes:")
        for note in notes:
            print(f"    - {note}")


if __name__ == "__main__":
    main()
