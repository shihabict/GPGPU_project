import cv2

# Initialize video capture and MOG2 background subtractor
video = cv2.VideoCapture("data/los_angeles.mp4")
mog2 = cv2.createBackgroundSubtractorMOG2()

# Define the desired width and height for the resized frames
display_width = 640  # Adjust as needed
display_height = 360  # Adjust as needed

while True:
    ret, frame = video.read()
    if not ret:
        break

        # Resize the original frame
    frame = cv2.resize(frame, (display_width, display_height))
    # Apply MOG2 to get the foreground mask
    mask = mog2.apply(frame)

    # Clean up the mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        if cv2.contourArea(contour) < 500: # Avoid small contours
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Object Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()