#!/usr/bin/env python3
"""
Convert raw framebuffer binary data to PNG image.
The raw data is expected to be RGB format (3 bytes per pixel),
with rows ordered bottom-to-top (OpenGL convention).
"""

import sys
import numpy as np
from PIL import Image


def convert_raw_to_png(input_file, output_file, width, height):
    """
    Convert raw RGB framebuffer data to PNG.

    Args:
        input_file: Path to raw binary file
        output_file: Path for output PNG file
        width: Image width in pixels
        height: Image height in pixels
    """
    # Calculate expected file size
    expected_size = width * height * 3

    # Read raw binary data
    with open(input_file, 'rb') as f:
        raw_data = f.read()

    if len(raw_data) != expected_size:
        print(f"Warning: Expected {expected_size} bytes, got {len(raw_data)} bytes")

    # Convert to numpy array and reshape
    data = np.frombuffer(raw_data, dtype=np.uint8)
    data = data.reshape((height, width, 3))

    # Flip vertically (convert from OpenGL's bottom-to-top to PNG's top-to-bottom)
    data = np.flipud(data)

    # Create PIL image and save as PNG
    img = Image.fromarray(data, 'RGB')
    img.save(output_file, 'PNG')
    print(f"Converted {input_file} to {output_file}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_raw_to_png.py <input.bin> <output.png> <width> <height>")
        print("Example: python convert_raw_to_png.py frame.bin frame.png 640 480")
        sys.exit(1)

    # Height: 617, Width: 1147
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 1147
    height = int(sys.argv[4]) if len(sys.argv) > 4 else 617

    try:
        convert_raw_to_png(input_file, output_file, width, height)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()