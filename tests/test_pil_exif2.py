#!/usr/bin/env python3
from PIL import Image
from PIL.ExifTags import TAGS
import sys

if len(sys.argv) < 2:
    print("Usage: python test_pil_exif2.py <image_file>")
    sys.exit(1)

image_path = sys.argv[1]
print(f"Testing PIL EXIF with: {image_path}")

try:
    with Image.open(image_path) as img:
        print(f"\nImage format: {img.format}")
        
        # Try to get EXIF from tag_v2
        exif = img.getexif()
        print(f"\nEXIF from getexif(): {len(exif)} tags")
        
        # Try to get IFD (Image File Directory) data
        if hasattr(exif, 'get_ifd'):
            print("\nTrying to get IFD data...")
            try:
                # EXIF IFD (tag 34665)
                exif_ifd = exif.get_ifd(0x8769)
                print(f"EXIF IFD: {len(exif_ifd)} tags")
                
                # Show relevant tags from EXIF IFD
                important_tags = {
                    34855: 'ISOSpeedRatings',
                    33437: 'FNumber', 
                    33434: 'ExposureTime',
                    37386: 'FocalLength',
                    42036: 'LensModel'
                }
                
                print("\nImportant EXIF IFD tags:")
                for tag_id, tag_name in important_tags.items():
                    value = exif_ifd.get(tag_id)
                    if value is not None:
                        print(f"  {tag_name} ({tag_id}): {value} (type: {type(value).__name__})")
                    
            except Exception as e:
                print(f"Error getting IFD: {e}")
        
        # Alternative: use _getexif() if available (deprecated but may work)
        if hasattr(img, '_getexif'):
            print("\nTrying deprecated _getexif()...")
            exif_data = img._getexif()
            if exif_data:
                print(f"Found {len(exif_data)} tags")
                for tag_id in [34855, 33437, 33434, 37386, 42036]:
                    if tag_id in exif_data:
                        tag_name = TAGS.get(tag_id, tag_id)
                        print(f"  {tag_name} ({tag_id}): {exif_data[tag_id]}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
