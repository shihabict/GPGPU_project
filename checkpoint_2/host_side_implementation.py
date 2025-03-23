import cv2
import numpy as np
import time  # Import the time module

def adaptive_thresholding(image, block_size, C):

    # Ensure the block size is odd
    if block_size % 2 == 0:
        block_size += 1

    # Get image dimensions
    height, width = image.shape

    # Create an empty output image
    output = np.zeros_like(image)

    # Pad the image to handle border pixels
    pad = block_size // 2
    padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # Apply adaptive thresholding
    for y in range(height):
        for x in range(width):
            # Extract the local neighborhood
            neighborhood = padded_image[y:y + block_size, x:x + block_size]
            # Compute the mean of the neighborhood
            mean = np.mean(neighborhood)
            # Compute the threshold
            threshold = mean - C
            # Classify the pixel
            if image[y, x] > threshold:
                output[y, x] = 255  # Foreground
            else:
                output[y, x] = 0  # Background

    return output

# Load an image in grayscale mode
image = cv2.imread("detection.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# Define parameters for adaptive thresholding
block_size = 11  # Size of the local neighborhood (must be odd)
C = 2  # Constant subtracted from the mean

# Start timing
start_time = time.time()

# Apply adaptive thresholding
thresholded_image = adaptive_thresholding(image, block_size, C)

# Stop timing
end_time = time.time()

# Calculate the elapsed time
execution_time = end_time - start_time

# Save the output image
cv2.imwrite("image_adpt_python.jpg", thresholded_image)

# Save execution time to file
with open("host_side_metrics.txt", "w") as file:
    file.write(f"Execution Time: {execution_time} seconds\n")

print("Adaptive Thresholding completed and output image saved.")
print(f"Execution Time: {execution_time} seconds")
