#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <algorithm>

#define BLOCK_SIZE 16
#define RADIUS 1

// adaptive threshold kernel
__global__ void adaptiveThresholdKernel(const unsigned char* input, unsigned char* output, int width, int height, int blockSize, int C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pad = blockSize / 2;
    int sum = 0;
    int count = 0;

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

bool readMetaFile(const std::string& meta_file_path, int& image_width, int& image_height) {
    std::ifstream meta_file(meta_file_path);
    if (!meta_file.is_open()) {
        std::cerr << "Error: Could not open metadata file!" << std::endl;
        return false;
    }

    meta_file >> image_width >> image_height;

    meta_file.close();
    return true;
}

//load raw image from file
unsigned char* loadRawImage(const std::string& raw_image_path, int image_width, int image_height) {

    std::ifstream file(raw_image_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open raw image file!" << std::endl;
        return nullptr;
    }

    // Memory allocation for the image
    unsigned char* image_data = new unsigned char[image_width * image_height];

    // Image to the array
    file.read(reinterpret_cast<char*>(image_data), image_width * image_height * sizeof(unsigned char));
    file.close();

    return image_data;
}

void createDirectory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0777);
    }
}

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
        std::cerr << "Usage: " << argv[0] << " <raw_image_path> <meta_file_path>" << std::endl;
        return -1;
    }

    std::string raw_image_path = argv[1];
    std::string meta_file_path = argv[2];

    std::string base_name = getBaseName(raw_image_path);

    createDirectory("output_images");

    createDirectory("metrices");

    int image_width = 0;
    int image_height = 0;
    if (!readMetaFile(meta_file_path, image_width, image_height)) {
        return -1;
    }

    unsigned char* h_input = loadRawImage(raw_image_path, image_width, image_height);
    if (h_input == nullptr) {
        return -1;
    }

    unsigned char* h_output = new unsigned char[image_width * image_height];

    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, image_width * image_height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, image_width * image_height * sizeof(unsigned char));

    // Copy input image to device
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

    // block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((image_width + blockSize.x - 1) / blockSize.x, (image_height + blockSize.y - 1) / blockSize.y);

    int C = 2;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch the adaptive threshold kernel
    adaptiveThresholdKernel<<<gridSize, blockSize>>>(d_input, d_output, image_width, image_height, BLOCK_SIZE, C);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

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
    std::string output_image_path = "output_images/" + base_name + "_adpt_threshold_cuda.raw";
    std::ofstream output_file(output_image_path, std::ios::binary);
    output_file.write(reinterpret_cast<char*>(h_output), image_width * image_height * sizeof(unsigned char));
    output_file.close();

    std::string metrics_path = "metrices/" + base_name + "_performance_metrics_cuda.txt";
    std::ofstream metrics_file(metrics_path);
    metrics_file << "Input Image: " << base_name << "\n";
    metrics_file << "Execution Time: " << milliseconds << " ms\n";
    metrics_file << "Host to Device Copy Time: " << h2d_ms << " ms\n";
    metrics_file << "Device to Host Copy Time: " << d2h_ms << " ms\n";

    metrics_file << "Output Image: " << output_image_path << std::endl;
    metrics_file.close();

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    std::cout << "Adaptive Thresholding completed successfully!" << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    return 0;
}


