#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <algorithm> 

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

// Function to read image dimensions from metadata
bool readMetaFile(const std::string& meta_file_path, int& image_width, int& image_height) {
    std::ifstream meta_file(meta_file_path);
    if (!meta_file.is_open()) {
        std::cerr << "Error: Could not open metadata file!" << std::endl;
        return false;
    }

    // Read the width and height
    meta_file >> image_width >> image_height;

    meta_file.close();
    return true;
}

// load raw image from file
unsigned char* loadRawImage(const std::string& raw_image_path, int image_width, int image_height) {
    // Open raw image
    std::ifstream file(raw_image_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open raw image file!" << std::endl;
        return nullptr;
    }

    // Allocate memory for the image
    unsigned char* image_data = new unsigned char[image_width * image_height];

    // image data into the array
    file.read(reinterpret_cast<char*>(image_data), image_width * image_height * sizeof(unsigned char));
    file.close();

    return image_data;
}


// function to create directory
void createDirectory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0777);
    }
}

//function to extract base name from path
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
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <raw_image_path> <meta_file_path>" << std::endl;
        return -1;
    }

    std::string raw_image_path = argv[1];
    std::string meta_file_path = argv[2];

    // base name for output files
    std::string base_name = getBaseName(raw_image_path);

    // Create output directory
    createDirectory("output_images");

    // Create metrics directory
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

    // host memory allocation for the output image
    unsigned char* h_output = new unsigned char[image_width * image_height];

    // device memory allocation
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, image_width * image_height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, image_width * image_height * sizeof(unsigned char));

    // input image to device
    cudaMemcpy(d_input, h_input, image_width * image_height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //  block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((image_width + blockSize.x - 1) / blockSize.x, (image_height + blockSize.y - 1) / blockSize.y);

    // Threshold constant c 
    int C = 2;
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timing
    cudaEventRecord(start);

    // launch adaptive threshold kernel 
    adaptiveThresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, image_width, image_height, BLOCK_SIZE, C);

    // kernel errors checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //  elapsed time calculation
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // output image back to host
    cudaMemcpy(h_output, d_output, image_width * image_height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    std::string output_image_path = "output_images/" + base_name + "_adpt_threshold_cuda.raw";
    std::ofstream output_file(output_image_path, std::ios::binary);
    output_file.write(reinterpret_cast<char*>(h_output), image_width * image_height * sizeof(unsigned char));
    output_file.close();

    // performance metrics with base name
    std::string metrics_path = "metrices/" + base_name + "_performance_metrics_cuda.txt";
    std::ofstream metrics_file(metrics_path);
    metrics_file << "Input Image: " << base_name << "\n";
    metrics_file << "Execution Time: " << milliseconds << " ms\n";
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


