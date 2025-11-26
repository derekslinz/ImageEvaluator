#!/usr/bin/env python3
import rawpy
import sys

if len(sys.argv) < 2:
    print("Usage: python test_rawpy.py <nef_file>")
    sys.exit(1)

image_path = sys.argv[1]
print(f"Testing rawpy with: {image_path}")

try:
    raw = rawpy.imread(image_path)
    print(f"\nrawpy version: {rawpy.__version__}")
    print(f"\nObject type: {type(raw)}")
    
    # List all non-private attributes
    attrs = [attr for attr in dir(raw) if not attr.startswith('_')]
    print(f"\nAvailable attributes ({len(attrs)}):")
    for attr in attrs:
        try:
            value = getattr(raw, attr, None)
            if not callable(value):
                print(f"  {attr}: {value} (type: {type(value).__name__})")
        except Exception as e:
            print(f"  {attr}: ERROR - {e}")
    
    # Try specific metadata attributes
    print("\n\nTrying specific metadata attributes:")
    for attr in ['iso_speed', 'iso', 'aperture', 'f_number', 'shutter', 'exposure_time', 
                 'focal_length', 'camera_maker', 'camera', 'camera_model', 'lens']:
        try:
            value = getattr(raw, attr, 'NOT_FOUND')
            print(f"  {attr}: {value}")
        except Exception as e:
            print(f"  {attr}: ERROR - {e}")
    
    raw.close()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
