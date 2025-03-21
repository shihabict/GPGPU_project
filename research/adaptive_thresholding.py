import cv2
import numpy as np

# KNN
KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # detectShadows=True: exclude shadow areas from the objects you detected

# MOG2
MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)  # exclude shadow areas from the objects you detected

# Choose your subtractor
bg_subtractor = MOG2_subtractor

# Read the image
image = cv2.imread("resources/image.jpg")  # Replace with your image path

# Apply background subtraction to the image
foreground_mask = bg_subtractor.apply(image)

# Apply adaptive thresholding to the foreground mask
adaptive_threshold = cv2.adaptiveThreshold(
    foreground_mask,  # Input image (foreground mask)
    255,  # Maximum value to use with THRESH_BINARY
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method (Gaussian-weighted sum)
    cv2.THRESH_BINARY,  # Threshold type
    11,  # Block size (size of a pixel neighborhood)
    2,  # Constant subtracted from the mean
)

# Dilation to expand or thicken regions of interest in the image
dilated = cv2.dilate(adaptive_threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

# Find contours
contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around detected objects
for contour in contours:
    # If the area exceeds a certain value, draw a bounding box
    if cv2.contourArea(contour) > 50:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

# Save the results
cv2.imwrite("foreground_mask.jpg", foreground_mask)
cv2.imwrite("adaptive_threshold.jpg", adaptive_threshold)
cv2.imwrite("detection.jpg", image)

# Display the results (optional)
cv2.imshow("Foreground Mask", foreground_mask)
cv2.imshow("Adaptive Threshold", adaptive_threshold)
cv2.imshow("Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()