#include <string.h>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include "OnnxModel.h"

/*
== = INPUT INFO == =
Input 0: input
Shape : [-1, 3, 224, 224]
Type : float32

== = OUTPUT INFO == =
Output 0 : xyz_x21
Shape : [-1, 63]
Type : float32

Output 1 : hand_score
Shape : [-1, 1]
Type : float32

Output 2 : lefthand_0_or_righthand_1
Shape : [-1, 1]
Type : float32
*/




int main() {
    Onnx_loader MediaPipe_model();
    cv::VideoCapture capture(0);

    if (!capture.isOpened()) {
        std::cerr << "Unable to connect camera " << std::endl;
        return -1;
    }
    std::cout << "Successfully opened video." << std::endl;
    std::cout << "Backend: " << capture.getBackendName() << std::endl;
    return 0;
}