#include <cuda_runtime.h>

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

extern "C" void adaptiveThresholdCUDA(const unsigned char* input, unsigned char* output, int width, int height, int blockSize, int C) {
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    adaptiveThresholdKernel<<<grid, block>>>(d_input, d_output, width, height, blockSize, C);

    cudaMemcpy(output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
