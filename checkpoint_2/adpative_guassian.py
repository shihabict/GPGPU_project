import cv2
import numpy as np
import cupy as cp
import time

# Load image
image = cv2.imread('detection.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Could not load image!")

height, width = image.shape
print(f"Image Shape: {image.shape}")

# Transfer data to GPU
image_gpu = cp.asarray(image)  # Copy image to GPU
output_gpu = cp.zeros((height, width), dtype=cp.uint8)

# CUDA kernel source code for Gaussian-based adaptive thresholding
cuda_kernel = r'''
extern "C" __global__ void adaptive_threshold(const unsigned char *input, unsigned char *output, int width, int height, int neighborhood_size, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int i = -neighborhood_size; i <= neighborhood_size; i++) {
        for (int j = -neighborhood_size; j <= neighborhood_size; j++) {
            int idx_x = x + j;
            int idx_y = y + i;

            // Handle edge cases by clamping
            idx_x = max(0, min(idx_x, width - 1));
            idx_y = max(0, min(idx_y, height - 1));

            // Compute Gaussian weight
            float weight = expf(-(i * i + j * j) / (2 * sigma * sigma));
            sum += input[idx_y * width + idx_x] * weight;
            weight_sum += weight;
        }
    }

    float threshold = sum / weight_sum;

    // Apply thresholding
    if (input[y * width + x] > threshold) {
        output[y * width + x] = 255; // Foreground
    } else {
        output[y * width + x] = 0;   // Background
    }
}
'''

# Compile CUDA kernel
module = cp.RawModule(code=cuda_kernel)
kernel = module.get_function("adaptive_threshold")

# Define grid/block dimensions
block_size = (16, 16)  # Threads per block (2D)
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1]
)

# Set neighborhood size (e.g., 3x3 neighborhood)
neighborhood_size = 1

# Set Gaussian sigma (standard deviation)
sigma = 1.0  # Adjust this value to control the Gaussian spread

# Measure execution time
start_time = time.time()  # Start timer

# Run CUDA kernel
kernel(
    grid_size,  # Grid size (2D)
    block_size,  # Block size (2D)
    (image_gpu, output_gpu, width, height, neighborhood_size, sigma)
)

# Synchronize to ensure the kernel has finished
cp.cuda.stream.get_current_stream().synchronize()

# Measure execution time
end_time = time.time()  # End timer
execution_time = end_time - start_time

# Copy result back to CPU
output = cp.asnumpy(output_gpu)

# Save output image
cv2.imwrite('output_image_cuda_gaussian.png', output)

# Print execution time
print(f"Execution Time: {execution_time * 1000:.2f} ms")

print("Output image saved as output_image_cuda_gaussian.png")
