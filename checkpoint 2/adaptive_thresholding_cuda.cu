#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define RADIUS 1 // For a 3x3 neighborhood

// CUDA kernel for adaptive thresholding
__global__ void adaptiveThresholdKernel(const unsigned char* input_image, unsigned char* output_image,
                                       int image_width, int image_height) {
    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the pixel is within the image bounds
    if (x >= image_width || y >= image_height) return;

    // Shared memory for neighborhood
    __shared__ int shared_mem[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    // Load pixel and its neighborhood into shared memory
    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int idx_x = x + j;
            int idx_y = y + i;

            // Handle edge cases by clamping
            idx_x = max(0, min(idx_x, image_width - 1));
            idx_y = max(0, min(idx_y, image_height - 1));

            shared_mem[threadIdx.y + i + RADIUS][threadIdx.x + j + RADIUS] = input_image[idx_y * image_width + idx_x];
        }
    }

    // Synchronize threads to ensure shared memory is loaded
    __syncthreads();

    // Compute threshold (mean of neighborhood)
    int sum = 0;
    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            sum += shared_mem[threadIdx.y + i + RADIUS][threadIdx.x + j + RADIUS];
        }
    }
    int threshold = sum / ((2 * RADIUS + 1) * (2 * RADIUS + 1));

    // Apply thresholding
    if (input_image[y * image_width + x] > threshold) {
        output_image[y * image_width + x] = 255; // Foreground
    } else {
        output_image[y * image_width + x] = 0;   // Background
    }
}

int main() {
    // Load image using OpenCV
    std::string image_path = "input_image.jpg"; // Replace with your image path
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    int image_width = image.cols;
    int image_height = image.rows;

    // Allocate host memory for input and output images
    unsigned char* h_input = image.data;
    unsigned char* h_output = new unsigned char[image_width * image_height];

    // Allocate device memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, image_width * image_height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, image_width * image_height * sizeof(unsigned char));

    // Copy input image to device
    cudaMemcpy(d_input, h_input, image_width * image_height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((image_width + blockSize.x - 1) / blockSize.x, (image_height + blockSize.y - 1) / blockSize.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Launch kernel
    adaptiveThresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, image_width, image_height);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output image back to host
    cudaMemcpy(h_output, d_output, image_width * image_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save output image
    cv::Mat output_image(image_height, image_width, CV_8UC1, h_output);
    cv::imwrite("output_image.jpg", output_image);

    // Save performance metrics to a text file
    std::ofstream metrics_file("metrics.txt");
    metrics_file << "Execution Time: " << milliseconds << " ms" << std::endl;
    metrics_file.close();

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    delete[] h_output;

    std::cout << "Adaptive Thresholding completed successfully!" << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    return 0;
}