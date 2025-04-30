#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <sys/stat.h>  // For directory creation
#include <algorithm>   // For string manipulation

#define BLOCK_SIZE 16
#define RADIUS 1 // For a 3x3 neighborhood

__global__ void adaptiveThresholdKernel(const unsigned char* input, unsigned char* output, int width, int height, int blockSize, int C) {
    // Thread and block coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    //int sharedSize = blockSize + 2 * RADIUS;
    __shared__ unsigned char tile[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    // Global coordinates of the shared memory element
    int sharedX = tx + RADIUS;
    int sharedY = ty + RADIUS;

    // Load the central data
    if (x < width && y < height)
        tile[sharedY][sharedX] = input[y * width + x];
    else
        tile[sharedY][sharedX] = 0;

    // Load the halo (padding) data
    if (tx < RADIUS) {
        // Left
        int x_left = x - RADIUS;
        tile[sharedY][tx] = (x_left >= 0 && y < height) ? input[y * width + x_left] : 0;
        // Right
        int x_right = x + BLOCK_SIZE;
        tile[sharedY][sharedX + RADIUS] = (x_right < width && y < height) ? input[y * width + x_right] : 0;
    }
    if (ty < RADIUS) {
        // Top
        int y_top = y - RADIUS;
        tile[ty][sharedX] = (y_top >= 0 && x < width) ? input[y_top * width + x] : 0;
        // Bottom
        int y_bottom = y + BLOCK_SIZE;
        tile[sharedY + RADIUS][sharedX] = (y_bottom < height && x < width) ? input[y_bottom * width + x] : 0;
    }

    // Corners
    if (tx < RADIUS && ty < RADIUS) {
        int x_corner = x - RADIUS;
        int y_corner = y - RADIUS;
        tile[ty][tx] = (x_corner >= 0 && y_corner >= 0) ? input[y_corner * width + x_corner] : 0;

        x_corner = x + BLOCK_SIZE;
        tile[ty][sharedX + RADIUS] = (x_corner < width && y_corner >= 0) ? input[y_corner * width + x_corner] : 0;

        y_corner = y + BLOCK_SIZE;
        tile[sharedY + RADIUS][tx] = (x_corner >= 0 && y_corner < height) ? input[y_corner * width + x_corner] : 0;

        tile[sharedY + RADIUS][sharedX + RADIUS] = (x_corner < width && y_corner < height) ? input[y_corner * width + x_corner] : 0;
    }

    __syncthreads();  // Ensure all shared memory is loaded

    // Perform local mean filtering from shared memory
    if (x < width && y < height) {
        int sum = 0;
        int count = 0;

        for (int j = -RADIUS; j <= RADIUS; ++j) {
            for (int i = -RADIUS; i <= RADIUS; ++i) {
                sum += tile[sharedY + j][sharedX + i];
                count++;
            }
        }

        int mean = sum / count;
        int threshold = mean - C;
        output[y * width + x] = (tile[sharedY][sharedX] > threshold) ? 255 : 0;
    }
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


// Helper function to create directory if it doesn't exist
void createDirectory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0777);
    }
}

// Helper function to extract base name from path
std::string getBaseName(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    size_t last_dot = path.find_last_of(".");

    if (last_slash == std::string::npos) last_slash = 0;
    else last_slash++;

    if (last_dot == std::string::npos || last_dot < last_slash) {
        return path.substr(last_slash);
    }
    return path.substr(last_slash, last_dot - last_slash);
}

int main(int argc, char* argv[]) {
    // Paths to the raw image and metadata files
//     std::string raw_image_path = "detection.raw";
//     std::string meta_file_path = "detection.raw.meta";
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <raw_image_path> <meta_file_path>" << std::endl;
        return -1;
    }

    std::string raw_image_path = argv[1];
    std::string meta_file_path = argv[2];

    // Get base name for output files
    std::string base_name = getBaseName(raw_image_path);

    // Create output directory
    createDirectory("output_images");

    // Create output directory
    createDirectory("metrices");

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
   // cudaMemcpy(d_input, h_input, image_width * image_height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    float h2d_ms = 0.0f;
    cudaEvent_t h2d_start, h2d_stop;
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    cudaEventRecord(h2d_start);

    cudaMemcpy(d_input, h_input, image_width * image_height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);
    cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop);
    std::cout << "Host to Device Copy Time: " << h2d_ms << " ms" << std::endl;

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((image_width + blockSize.x - 1) / blockSize.x, (image_height + blockSize.y - 1) / blockSize.y);

    // Threshold constant (C) and block size (to adjust the neighborhood area)
    int C = 2;
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
    //cudaMemcpy(h_output, d_output, image_width * image_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    float d2h_ms = 0.0f;
    cudaEvent_t d2h_start, d2h_stop;
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);
    cudaEventRecord(d2h_start);

    cudaMemcpy(h_output, d_output, image_width * image_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);
    cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop);
    std::cout << "Device to Host Copy Time: " << d2h_ms << " ms" << std::endl;

//     // Save output image (as raw file or any other format)
//     std::ofstream output_file("output_cuda_adpt_image.raw", std::ios::binary);
//     output_file.write(reinterpret_cast<char*>(h_output), image_width * image_height * sizeof(unsigned char));
//     output_file.close();
    // Save output image with base name in output_images directory
    std::string output_image_path = "output_images/" + base_name + "_adpt_threshold_cuda.raw";
    std::ofstream output_file(output_image_path, std::ios::binary);
    output_file.write(reinterpret_cast<char*>(h_output), image_width * image_height * sizeof(unsigned char));
    output_file.close();

//     // Save performance metrics to a text file
//     std::ofstream metrics_file("cuda_performance_metrics.txt");
//     metrics_file << "Execution Time: " << milliseconds << " ms" << std::endl;
//     metrics_file.close();
    // Save performance metrics with base name
    std::string metrics_path = "metrices/" + base_name + "_performance_metrics_cuda.txt";
    std::ofstream metrics_file(metrics_path);
    metrics_file << "Input Image: " << base_name << "\n";
    metrics_file << "Execution Time: " << milliseconds << " ms\n";
    metrics_file << "Host to Device Copy Time: " << h2d_ms << " ms\n";
    metrics_file << "Device to Host Copy Time: " << d2h_ms << " ms\n";

    metrics_file << "Output Image: " << output_image_path << std::endl;
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


