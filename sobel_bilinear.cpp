#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <limits>
#include <random>
#include <cassert>
#include <cstdint>

using namespace std;

void applySobelEdgeDetection(const std::string& inputImagePath) {
    // Load the image in grayscale
    cv::Mat image = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);

    // Error handling
    if (image.empty()) {
        std::cerr << "Error: Image not found or could not be loaded: " << inputImagePath << std::endl;
        return;
    }

    // Define Sobel kernels
    cv::Mat sobelX = (cv::Mat_<float>(3, 3) << -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);
    cv::Mat sobelY = (cv::Mat_<float>(3, 3) << 1, 2, 1,
        0, 0, 0,
        -1, -2, -1);

    // Prepare matrices for the gradient images
    cv::Mat gradX, gradY;
    cv::Mat absGradX, absGradY, sobelImage;

    // Apply Sobel X kernel
    cv::filter2D(image, gradX, CV_32F, sobelX);
    cv::convertScaleAbs(gradX, absGradX);

    // Apply Sobel Y kernel
    cv::filter2D(image, gradY, CV_32F, sobelY);
    cv::convertScaleAbs(gradY, absGradY);

    // Combine
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, sobelImage);

    // Show the resulting image
    cv::imshow("Sobel Edges", sobelImage);
    cv::imshow("Original Image", image);

    // Wait for any keystroke
    cv::waitKey(0);
}

int bilinear_interpolation(const std::string& inputImagePath, const std::string& outputImagePath, int scale_x, int scale_y, bool show_image) {

    cv::Mat image;

    image = cv::imread(inputImagePath, cv::IMREAD_COLOR);

    int rows = image.rows;
    int cols = image.cols;
    int new_rows = rows / scale_y;
    int new_cols = cols / scale_x;
    int type = image.type();

    // Initiate a matrix of new_row x new_col
    cv::Mat new_image(new_rows, new_cols, type);

    // Compute the scaling factors
    float xScale = static_cast<float>(image.cols) / new_cols;
    float yScale = static_cast<float>(image.rows) / new_rows;

    for (int y = 0; y < new_rows; ++y) {
        for (int x = 0; x < new_cols; ++x) {
            // Compute the corresponding position in the original image
            float srcX = x * xScale;
            float srcY = y * yScale;

            int x1 = static_cast<int>(srcX);
            int y1 = static_cast<int>(srcY);
            int x2 = x1 + 1;
            int y2 = y1 + 1;

            // Ensure coordinates are within bounds
            if (x2 >= image.cols) x2 = image.cols - 1;
            if (y2 >= image.rows) y2 = image.rows - 1;

            // Compute the fractional parts
            float dx = srcX - x1;
            float dy = srcY - y1;

            // Get the values at the four surrounding points
            cv::Vec3b Q11 = image.at<cv::Vec3b>(y1, x1);
            cv::Vec3b Q21 = image.at<cv::Vec3b>(y1, x2);
            cv::Vec3b Q12 = image.at<cv::Vec3b>(y2, x1);
            cv::Vec3b Q22 = image.at<cv::Vec3b>(y2, x2);
            

            // Perform bilinear interpolation
            cv::Vec3b interpolatedValue;
            for (int c = 0; c < 3; ++c) {
                interpolatedValue[c] = static_cast<uchar>(
                    (1 - dx) * (1 - dy) * Q11[c] +
                    dx * (1 - dy) * Q21[c] +
                    (1 - dx) * dy * Q12[c] +
                    dx * dy * Q22[c]
                    );
            }
            
            // Set the value in the destination image
            new_image.at<cv::Vec3b>(y, x) = interpolatedValue;


        }

    }

    // Optionally display the images
    if (show_image) {
        cv::imshow("Original Image", image);
        cv::imshow("Resized Image", new_image);
        cv::waitKey(0);

    }
    

    if (cv::imwrite(outputImagePath, new_image)) {
        std::cout << "Image saved successfully to " << outputImagePath << std::endl;
    }
    else {
        std::cerr << "Error saving image!" << std::endl;
        return -1;
    }

    return 0;

}

int main() {

	auto inputImg = "C:\\Users\\bhara\\Downloads\\redcar.jpg";
	auto outputImg = "C:\\Users\\bhara\\Downloads\\redcar_resized.jpg";

    applySobelEdgeDetection(inputImg);

    // The amount by which to scale down image
    int scale_x = 4;
    int scale_y = 4;
    bool show_image = false;
    bilinear_interpolation(inputImg, outputImg, scale_x, scale_y, show_image);

    return 0;

}
