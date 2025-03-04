#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>

#define MAX_COMPONENTS 2

__global__ void gmm_background_kernel(const unsigned char* frames, float* background, int num_frames, int height, int width, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Column index
    int k = blockIdx.z * blockDim.z + threadIdx.z; // Channel index

    if (i < height && j < width && k < channels) {
        // Extract intensity values for this pixel across all frames
        float intensity_values[num_frames];
        for (int f = 0; f < num_frames; f++) {
            intensity_values[f] = (float)frames[f * height * width * channels + i * width * channels + j * channels + k];
        }

        // Fit a simple GMM (assuming 2 components)
        float means[MAX_COMPONENTS] = {0.0f};
        float weights[MAX_COMPONENTS] = {0.5f, 0.5f}; // Equal weights for simplicity
        float variances[MAX_COMPONENTS] = {1.0f, 1.0f}; // Initial variances

        // Expectation-Maximization (EM) steps (simplified)
        for (int iter = 0; iter < 10; iter++) { // Fixed number of iterations
            float responsibilities[num_frames][MAX_COMPONENTS];

            // E-step: Compute responsibilities
            for (int f = 0; f < num_frames; f++) {
                float sum = 0.0f;
                for (int c = 0; c < MAX_COMPONENTS; c++) {
                    responsibilities[f][c] = weights[c] * expf(-0.5f * powf((intensity_values[f] - means[c]), 2) / sqrtf(2 * M_PI * variances[c]);
                    sum += responsibilities[f][c];
                }
                for (int c = 0; c < MAX_COMPONENTS; c++) {
                    responsibilities[f][c] /= sum;
                }
            }

            // M-step: Update means, variances, and weights
            for (int c = 0; c < MAX_COMPONENTS; c++) {
                float sum_resp = 0.0f;
                float sum_resp_x = 0.0f;
                float sum_resp_x2 = 0.0f;

                for (int f = 0; f < num_frames; f++) {
                    sum_resp += responsibilities[f][c];
                    sum_resp_x += responsibilities[f][c] * intensity_values[f];
                    sum_resp_x2 += responsibilities[f][c] * powf(intensity_values[f], 2);
                }

                means[c] = sum_resp_x / sum_resp;
                variances[c] = (sum_resp_x2 / sum_resp) - powf(means[c], 2);
                weights[c] = sum_resp / num_frames;
            }
        }

        // Select the mean of the most probable component
        int max_idx = (weights[0] > weights[1]) ? 0 : 1;
        background[i * width * channels + j * channels + k] = means[max_idx];
    }
}
