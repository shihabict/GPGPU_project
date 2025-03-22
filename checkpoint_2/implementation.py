import cv2
import numpy as np
import time

def adaptive_threshold(image, neighborhood_size=1):
    """
    Perform adaptive thresholding on a grayscale image.
    
    Args:
        image (numpy.ndarray): Input grayscale image.
        neighborhood_size (int): Size of the neighborhood (e.g., 1 for 3x3 neighborhood).
    
    Returns:
        numpy.ndarray: Binary output image after adaptive thresholding.
    """
    height, width = image.shape
    output = np.zeros_like(image)

    # Iterate over each pixel
    for y in range(height):
        for x in range(width):
            # Define the neighborhood boundaries
            y_start = max(0, y - neighborhood_size)
            y_end = min(height, y + neighborhood_size + 1)
            x_start = max(0, x - neighborhood_size)
            x_end = min(width, x + neighborhood_size + 1)

            # Extract the neighborhood
            neighborhood = image[y_start:y_end, x_start:x_end]

            # Compute the mean of the neighborhood
            threshold = np.mean(neighborhood)

            # Apply thresholding
            if image[y, x] > threshold:
                output[y, x] = 255  # Foreground
            else:
                output[y, x] = 0  # Background

    return output

# Load image
image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Could not load image!")

# Measure execution time
start_time = time.time()  # Start timer

# Perform adaptive thresholding
output = adaptive_threshold(image, neighborhood_size=1)

# Measure execution time
end_time = time.time()  # End timer
execution_time = end_time - start_time

# Save output image
cv2.imwrite('output_image_python.png', output)

# Print execution time
print(f"Execution Time: {execution_time * 1000:.2f} ms")

print("Output image saved as output_image_python.png")
