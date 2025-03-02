import cv2
import numpy as np

# Create a MOG2 background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Capture video
cap = cv2.VideoCapture('data/los_angeles.mp4')

# Define the desired width and height for the resized frames
display_width = 640  # Adjust as needed
display_height = 360  # Adjust as needed

# Get the original video's frame rate and dimensions
fps = int(cap.get(cv2.CAP_PROP_FPS))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter objects to save the output videos
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files

# VideoWriter for the concatenated video (side-by-side layout)
out_combined = cv2.VideoWriter('mog2_combined_video.mp4', fourcc, fps, (display_width * 2, display_height))

# VideoWriter for the background subtraction video (foreground mask only)
out_fgmask = cv2.VideoWriter('mog2_fgmask_video.mp4', fourcc, fps, (display_width, display_height), isColor=False)

while True:
    ret, frame = cap.read()

    if frame is None:
        break

    # Resize the original frame
    frame_resized = cv2.resize(frame, (display_width, display_height))

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Optional: Apply morphological operations to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Resize the foreground mask
    fgmask_resized = cv2.resize(fgmask, (display_width, display_height))

    # Convert the grayscale mask to a 3-channel image to match the original frame
    fgmask_colored = cv2.cvtColor(fgmask_resized, cv2.COLOR_GRAY2BGR)

    # Concatenate the original frame and the foreground mask side by side
    combined_frame = cv2.hconcat([frame_resized, fgmask_colored])

    # Write the combined frame to the concatenated video file
    out_combined.write(combined_frame)

    # Write the foreground mask to the background subtraction video file
    out_fgmask.write(fgmask_resized)

    # Display the combined frame
    cv2.imshow('Input Video | Foreground Mask', combined_frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Press 'Esc' to exit
        break

# Release everything when done
cap.release()
out_combined.release()
out_fgmask.release()
cv2.destroyAllWindows()