#!/usr/bin/env python3
from PIL import Image
import sys

if len(sys.argv) < 2:
    print("Usage: python test_lens_tags.py <image_file>")
    sys.exit(1)

image_path = sys.argv[1]
print(f"Testing lens tags with: {image_path}\n")

try:
    with Image.open(image_path) as img:
        exif = img.getexif()
        
        # Try to get EXIF IFD
        exif_ifd = None
        try:
            if hasattr(exif, 'get_ifd'):
                exif_ifd = exif.get_ifd(0x8769)
        except Exception:
            pass
        
        # Common lens tags to check
        lens_tags = {
            42036: 'LensModel (42036)',
            42035: 'LensMake (42035)',
            42034: 'LensSpecification (42034)',
            42033: 'LensSerialNumber (42033)',
            # Additional possible tags
            50: 'SubIFDOffset',
            254: 'NewSubfileType',
        }
        
        print("Checking main EXIF:")
        for tag_id, tag_name in lens_tags.items():
            value = exif.get(tag_id)
            if value is not None:
                print(f"  {tag_name}: {value}")
        
        print("\nChecking EXIF IFD:")
        if exif_ifd:
            for tag_id, tag_name in lens_tags.items():
                value = exif_ifd.get(tag_id)
                if value is not None:
                    print(f"  {tag_name}: {value}")
        
        # Show all tags containing 'lens' in various forms
        print("\nAll EXIF tags (first 50):")
        for i, (tag_id, value) in enumerate(exif.items()):
            if i >= 50:
                break
            print(f"  Tag {tag_id}: {repr(value)[:100]}")
        
        if exif_ifd:
            print("\nAll EXIF IFD tags (first 50):")
            for i, (tag_id, value) in enumerate(exif_ifd.items()):
                if i >= 50:
                    break
                print(f"  Tag {tag_id}: {repr(value)[:100]}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
