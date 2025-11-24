#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/ImageEvaluator')

from image_eval_embed import extract_exif_metadata

if len(sys.argv) < 2:
    print("Usage: python test_exif_extraction.py <image_file>")
    sys.exit(1)

image_path = sys.argv[1]
print(f"Testing EXIF extraction with: {image_path}\n")

metadata = extract_exif_metadata(image_path)

print("Extracted metadata:")
for key, value in metadata.items():
    print(f"  {key}: {value}")
