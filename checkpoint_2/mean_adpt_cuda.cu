#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define RADIUS 1 // For a 3x3 neighborhood

// Adaptive threshold kernel
__global__ void adaptiveThresholdKernel(const unsigned char* input, unsigned char* output, int width, int height, int blockSize, int C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pad = blockSize / 2;
    int sum = 0;
    int count = 0;

    // Loop through the blockSize area to calculate the local neighborhood sum
    for (int i = -pad; i <= pad; ++i) {
        for (int j = -pad; j <= pad; ++j) {
            int xi = x + i;
            int yj = y + j;

            if (xi >= 0 && xi < width && yj >= 0 && yj < height) {
                sum += input[yj * width + xi];
                count++;
            }
        }
    }

    int mean = sum / count;
    int threshold = mean - C;

    output[y * width + x] = (input[y * width + x] > threshold) ? 255 : 0;
}

// Function to read image dimensions from metadata file
bool readMetaFile(const std::string& meta_file_path, int& image_width, int& image_height) {
    std::ifstream meta_file(meta_file_path);
    if (!meta_file.is_open()) {
        std::cerr << "Error: Could not open metadata file!" << std::endl;
        return false;
    }

    // Read the width and height from the metadata file
    meta_file >> image_width >> image_height;

    meta_file.close();
    return true;
}

// Function to load raw image from file
unsigned char* loadRawImage(const std::string& raw_image_path, int image_width, int image_height) {
    // Open raw image file in binary mode
    std::ifstream file(raw_image_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open raw image file!" << std::endl;
        return nullptr;
    }

    // Allocate memory for the image
    unsigned char* image_data = new unsigned char[image_width * image_height];

    // Read image data into the array
    file.read(reinterpret_cast<char*>(image_data), image_width * image_height * sizeof(unsigned char));
    file.close();

    return image_data;
}

int main() {
    // Paths to the raw image and metadata files
    std::string raw_image_path = "detection.raw"; // Replace with your raw image path
    std::string meta_file_path = "detection.raw.meta"; // Replace with your metadata file path

    // Load image dimensions from metadata file
    int image_width = 0;
    int image_height = 0;
    if (!readMetaFile(meta_file_path, image_width, image_height)) {
        return -1;
    }

    // Load raw image using the dimensions from metadata
    unsigned char* h_input = loadRawImage(raw_image_path, image_width, image_height);
    if (h_input == nullptr) {
        return -1;
    }

    // Allocate host memory for output image
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

    // Threshold constant (C) and block size (to adjust the neighborhood area)
    int C = 10;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Launch the adaptive threshold kernel
    adaptiveThresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, image_width, image_height, BLOCK_SIZE, C);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output image back to host
    cudaMemcpy(h_output, d_output, image_width * image_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save output image (as raw file or any other format)
    std::ofstream output_file("output_cuda_adpt_image.raw", std::ios::binary);
    output_file.write(reinterpret_cast<char*>(h_output), image_width * image_height * sizeof(unsigned char));
    output_file.close();

    // Save performance metrics to a text file
    std::ofstream metrics_file("cuda_performance_metrics.txt");
    metrics_file << "Execution Time: " << milliseconds << " ms" << std::endl;
    metrics_file.close();

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    delete[] h_input;
    delete[] h_output;

    std::cout << "Adaptive Thresholding completed successfully!" << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    return 0;
}

