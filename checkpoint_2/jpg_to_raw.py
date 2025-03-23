import argparse
from PIL import Image
import numpy as np
import os

def convert_to_raw(image_path, output_raw=None):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img, dtype=np.uint8)

    # Set output raw file path
    if output_raw is None:
        output_raw = os.path.splitext(image_path)[0] + ".raw"

    # Save raw data
    img_array.tofile(output_raw)

    # Save image dimensions
    metadata_file = output_raw + ".meta"
    with open(metadata_file, "w") as f:
        f.write(f"{img.width} {img.height}")

    print(f"Saved raw image to {output_raw}")
    print(f"Saved metadata to {metadata_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image to raw format.")
    parser.add_argument("image_path", help="Path to the input image (JPG or PNG)")
    parser.add_argument("--output", help="Path to save the raw file", default=None)

    args = parser.parse_args()
    convert_to_raw(args.image_path, args.output)

