import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


vid = cv2.VideoCapture('data/los_angeles.mp4')

# frame size calculation
frames = []
frame_count = 0

while True:
    ret, frame = vid.read()
    if frame is not None:
        frames.append(frame)
        frame_count += 1
    else:
        break
frames = np.array(frames)

print("Number of frames extracted is {}".format(frame_count))
print("array dimensions will be (num_frames, image_width, image_height, num_channels)")
print("Shape of frames is {}".format(frames.shape))

# data model
gmm = GaussianMixture(n_components = 2)
# initialize a dummy background image with all zeros
background = np.zeros(shape=(frames.shape[1:]))
print("Shape of dummy background image is {}".format(background.shape))

for i in tqdm(range(frames.shape[1])):
    for j in range(frames.shape[2]):
        for k in range(frames.shape[3]):
            X = frames[:, i, j, k]
            X = X.reshape(X.shape[0], 1)
            gmm.fit(X)
            means = gmm.means_
            covars = gmm.covariances_
            weights = gmm.weights_
            idx = np.argmax(weights)
            # background[i][j][k] = int(means[idx])
            background[i][j][k] = int(means[idx].item())

# Store the result onto disc
cv2.imwrite('background.png', background)