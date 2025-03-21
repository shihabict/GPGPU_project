import cv2
import numpy as np

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
image = cv2.imread("../data/traffic-less-but-more-speeding.jpg", cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# Define parameters for adaptive thresholding
block_size = 11  # Size of the local neighborhood (must be odd)
C = 2  # Constant subtracted from the mean

# Apply adaptive thresholding
thresholded_image = adaptive_thresholding(image, block_size, C)

# Convert the thresholded image to 3 channels (to match the original image if it's not grayscale)
thresholded_image_bgr = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

# Convert the original grayscale image to 3 channels (for side-by-side display)
if len(image.shape) == 2:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
else:
    image_bgr = image

# Concatenate the original and thresholded images side by side
side_by_side = np.hstack((image_bgr, thresholded_image_bgr))

# Resize the concatenated image to fit the screen
scale_percent = 50  # Resize to 50% of the original size
width = int(side_by_side.shape[1] * scale_percent / 100)
height = int(side_by_side.shape[0] * scale_percent / 100)
resized_side_by_side = cv2.resize(side_by_side, (width, height))

# Create a resizable window
cv2.namedWindow("Original vs Adaptive Thresholded", cv2.WINDOW_NORMAL)

# Display the resized images side by side
cv2.imshow("Original vs Adaptive Thresholded", resized_side_by_side)

# Save the output image
cv2.imwrite("output_thresholded_image.jpg", thresholded_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.destroyAllWindows()
