import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path


def adaptive_thresholding(image, block_size, C):
    # Ensure the block size is odd
    if block_size % 2 == 0:
        block_size += 1

    # image dimensions extraction
    height, width = image.shape

    # empty output image
    output = np.zeros_like(image)

    # Padding to handle border pixels
    pad = block_size // 2
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # adaptive thresholding
    for y in range(height):
        for x in range(width):
            # local neighborhood extraction
            neighborhood = padded_image[y:y + block_size, x:x + block_size]
            # mean of the neighborhood
            mean = np.mean(neighborhood)
            # threshold computation
            threshold = mean - C
            # pixel classification
            if image[y, x] > threshold:
                output[y, x] = 255  # Foreground
            else:
                output[y, x] = 0  # Background

    return output


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python adaptive_threshold.py <input_image_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Create directories
    os.makedirs("output_images", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # Get the base filename
    input_filename = Path(input_path).stem

    # load image in grayscale mode
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image not empty
    if image is None:
        print(f"Error: Could not load image at {input_path}")
        sys.exit(1)

    # Define parameters for adaptive thresholding
    block_size = 11  # Size of the local neighborhood
    C = 2  # Constant subtracted from the mean

    # Start timing
    start_time = time.time()

    # Apply adaptive thresholding
    thresholded_image = adaptive_thresholding(image, block_size, C)

    # Stop timing
    end_time = time.time()

    # Calculate the elapsed time
    execution_time = end_time - start_time

    # Define output paths
    output_image_path = f"output_images/output_{input_filename}_host.jpg"
    metrics_path = f"metrics/metrics_{input_filename}_host.txt"

    # Save the output image
    cv2.imwrite(output_image_path, thresholded_image)

    # Save execution time to file
    with open(metrics_path, "w") as file:
        file.write(f"Execution Time: {execution_time:.4f} seconds\n")

    print(f"Output Image saved to: {output_image_path}")



if __name__ == "__main__":
    main()
