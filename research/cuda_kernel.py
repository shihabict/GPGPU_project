import cv2
import numpy as np
import cupy as cp
from tqdm import tqdm

# Load video
vid = cv2.VideoCapture('los_angeles.mp4')

frames = []
frame_count = 0

while True:
    ret, frame = vid.read()
    if frame is not None:
        frames.append(frame)
        frame_count += 1
    else:
        break
vid.release()

frames = np.array(frames, dtype=np.uint8)  # Convert to numpy array

num_frames, height, width, channels = frames.shape
print(f"Number of frames: {num_frames}, Frame Shape: {frames.shape}")

# Transfer data to GPU
frames_gpu = cp.asarray(frames)  # Copy frames to GPU
background_gpu = cp.zeros((height, width, channels), dtype=cp.uint8)

# CUDA kernel source code
cuda_kernel = r'''
extern "C" __global__ void background_subtraction(unsigned char *frames, unsigned char *background, int num_frames, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || c >= channels) return;

    int pixel_idx = (y * width + x) * channels + c;
    float mean = 0.0f;

    for (int f = 0; f < num_frames; f++) {
        mean += frames[f * width * height * channels + pixel_idx];
    }
    mean /= num_frames;

    background[pixel_idx] = static_cast<unsigned char>(mean);
}
'''

# Compile CUDA kernel
module = cp.RawModule(code=cuda_kernel)
kernel = module.get_function("background_subtraction")

# Define grid/block dimensions
block_size = (16, 16, 1)  # Threads per block
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1],
    channels
)

# Run CUDA kernel
kernel(
    ((width + 15) // 16, (height + 15) // 16, channels),  # Grid size (3D)
    (16, 16, 1),  # Block size (3D)
    (frames_gpu, background_gpu, num_frames, width, height, channels)
)


# Copy result back to CPU
background = cp.asnumpy(background_gpu)

# Save background image
cv2.imwrite('background_cuda.png', background)

print("Background image saved as background_cuda.png")
