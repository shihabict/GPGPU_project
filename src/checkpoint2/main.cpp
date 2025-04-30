#include <opencv2/opencv.hpp>
#include <iostream>

extern "C" void adaptiveThresholdCUDA(const unsigned char* input, unsigned char* output, int width, int height, int blockSize, int C);

int main() {
    // Load an image in grayscale mode
    cv::Mat image = cv::imread("detection.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Define parameters for adaptive thresholding
    int blockSize = 11;  // Size of the local neighborhood (must be odd)
    int C = 2;           // Constant subtracted from the mean

    // Create an output image
    cv::Mat output(image.size(), image.type());

    // Apply adaptive thresholding using CUDA
    adaptiveThresholdCUDA(image.ptr(), output.ptr(), image.cols, image.rows, blockSize, C);

    // Save the output image as a PNG file
    cv::imwrite("output_thresholded_image_cuda.png", output);

    std::cout << "Output image saved as 'output_thresholded_image_cuda.png'" << std::endl;

    return 0;
}
