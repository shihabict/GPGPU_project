import argparse
import numpy as np
from PIL import Image
import os

def convert_raw_to_image(raw_path, meta_path, output_image=None):
    # Check if metadata file exists
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file {meta_path} not found.")
        return

    # Read metadata
    with open(meta_path, "r") as f:
        width, height = map(int, f.read().split())

    # Read raw data
    raw_data = np.fromfile(raw_path, dtype=np.uint8)

    # Reshape based on metadata
    img_array = raw_data.reshape((height, width))

    # Convert to PIL image
    img = Image.fromarray(img_array, mode="L")

    # Set output image path
    if output_image is None:
        output_image = os.path.splitext(raw_path)[0] + "_converted.jpg"

    img.save(output_image)
    print(f"Saved converted image to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a raw image back to JPG.")
    parser.add_argument("raw_path", help="Path to the raw image file")
    parser.add_argument("meta_path", help="Path to the metadata file")
    parser.add_argument("--output", help="Path to save the output image", default=None)

    args = parser.parse_args()
    convert_raw_to_image(args.raw_path, args.meta_path, args.output)

