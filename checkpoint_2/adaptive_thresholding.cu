// adaptive_threshold.cu
extern "C" {
    __global__ void adaptiveThresholdKernel(const unsigned char* input_image, unsigned char* output_image,
                                           int image_width, int image_height, int neighborhood_size) {
        // Calculate pixel coordinates
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Check if the pixel is within the image bounds
        if (x >= image_width || y >= image_height) return;

        // Compute the mean of the neighborhood
        int sum = 0;
        int count = 0;
        for (int i = -neighborhood_size; i <= neighborhood_size; i++) {
            for (int j = -neighborhood_size; j <= neighborhood_size; j++) {
                int idx_x = x + j;
                int idx_y = y + i;

                // Handle edge cases by clamping
                idx_x = max(0, min(idx_x, image_width - 1));
                idx_y = max(0, min(idx_y, image_height - 1));

                sum += input_image[idx_y * image_width + idx_x];
                count++;
            }
        }

        int threshold = sum / count;

        // Apply thresholding
        if (input_image[y * image_width + x] > threshold) {
            output_image[y * image_width + x] = 255; // Foreground
        } else {
            output_image[y * image_width + x] = 0;   // Background
        }
    }
}
