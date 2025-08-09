#include <string.h>
#include <filesystem>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include "OnnxModel.h"
#include "box_visualizer.h"
#include "Mouse_event.h"

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

using namespace std;


int main() {

    clock_t start, finish;
    double duration;

    start = clock();
    
    Onnx_loader MediaPipe_model;
    BOX_DRAWING box_visualizer;
    Mouse_event event_control;

    std::cout << "3. ONNX 환경 생성..." << std::endl;

    cv::VideoCapture capture(0);
    capture.set(cv::CAP_PROP_FPS, 40);
    capture.set(cv::CAP_PROP_BRIGHTNESS, 0.3);


    

    if (!capture.isOpened()) {
        std::cerr << "Unable to connect camera " << std::endl;
        return -1;
    }
    std::cout << "Successfully opened video." << std::endl;
    std::cout << "Backend: " << capture.getBackendName() << std::endl;
    
    cv::Mat img;
    cv::Mat flipped_img;
    Onnx_Outputs result;

    while (1)
    {   
        start = clock();

        capture >> img;
        cv::flip(img, flipped_img, 1);
        MediaPipe_model.get_data(img);
        box_visualizer.updateImage(flipped_img);

        result = MediaPipe_model.pred_pose();
        box_visualizer.updatehandpos(result);
        event_control.updatehandpos(result);
           

        flipped_img = box_visualizer.process();
        event_control.process();
        imshow("camera img", flipped_img);



        finish = clock();

        duration = (double)CLOCKS_PER_SEC/ (finish - start);
        cout << duration << "FPS" << endl;

        if (cv::waitKey(1) == 27)
            break;
       
        
    }
    
    return 0;
}