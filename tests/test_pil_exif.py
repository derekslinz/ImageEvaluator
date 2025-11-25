#!/usr/bin/env python3
from PIL import Image
import sys

if len(sys.argv) < 2:
    print("Usage: python test_pil_exif.py <image_file>")
    sys.exit(1)

image_path = sys.argv[1]
print(f"Testing PIL EXIF with: {image_path}")

try:
    with Image.open(image_path) as img:
        print(f"\nImage format: {img.format}")
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        
        exif = img.getexif()
        print(f"\nEXIF data available: {bool(exif)}")
        
        if exif:
            print(f"Number of EXIF tags: {len(exif)}")
            
            # Common EXIF tags
            tag_names = {
                34855: 'ISOSpeedRatings',
                33437: 'FNumber',
                33434: 'ExposureTime',
                37386: 'FocalLength',
                271: 'Make',
                272: 'Model',
                42036: 'LensModel'
            }
            
            print("\nRequested EXIF tags:")
            for tag_id, tag_name in tag_names.items():
                value = exif.get(tag_id)
                print(f"  {tag_name} ({tag_id}): {value} (type: {type(value).__name__})")
            
            # Show first 20 tags
            print("\nFirst 20 EXIF tags:")
            for i, (tag_id, value) in enumerate(exif.items()):
                if i >= 20:
                    break
                print(f"  Tag {tag_id}: {value} (type: {type(value).__name__})")
        else:
            print("No EXIF data found")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
