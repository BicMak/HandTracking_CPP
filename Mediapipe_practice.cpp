#include <string.h>
#include <filesystem>
#include <ctime>
#include <thread>
#include <future>


#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <iostream>
//#include "OnnxModel.h"
#include "OnnxYolo.h"
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
    Yolo_loader Yolo_model;
    BOX_DRAWING box_visualizer;
    Mouse_event event_control;


    cv::VideoCapture capture(0);
    capture.set(cv::CAP_PROP_FPS, 60);
    capture.set(cv::CAP_PROP_BRIGHTNESS, 0.5);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 640);

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

        auto start_camera = std::chrono::high_resolution_clock::now();
        capture >> img;
        cv::flip(img, flipped_img, 1);
        auto end_camera = std::chrono::high_resolution_clock::now();
        auto camera_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_camera - start_camera).count();

        auto img_up_future = std::async(std::launch::async, [&]() {
            box_visualizer.updateImage(flipped_img);
            });

        auto pipe_future = std::async(std::launch::async, [&]() {
            MediaPipe_model.get_data(img);
            return MediaPipe_model.pred_pose();
            });

        auto yolo_future = std::async(std::launch::async, [&]() {
            Yolo_model.get_data(img);
            return Yolo_model.pred_pose();
            });

        //std::cout << "✅ 비동기 추론 시작됨" << std::endl;

        // 결과 대기
        auto start_inference = std::chrono::high_resolution_clock::now();
        img_up_future.get();
        auto pipe_result = pipe_future.get();
        auto yolo_result = yolo_future.get();
        auto end_inference = std::chrono::high_resolution_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference).count();

        box_visualizer.updatehandpos(pipe_result);
        event_control.updatehandpos(pipe_result,yolo_result);
           

        flipped_img = box_visualizer.process();
        event_control.process();
        



        finish = clock();

        double fps = (double)CLOCKS_PER_SEC/ (finish - start);


        std::ostringstream duration;
        duration << std::fixed << std::setprecision(2) << fps;
        std::string contents = duration.str() + "FPS";

        cv::putText(flipped_img, contents, cv::Point(10, 50),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 1);
        cv::putText(flipped_img, std::to_string(yolo_result.class_id), cv::Point(30, 100),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 1);

        imshow("camera img", flipped_img);

        if (cv::waitKey(1) == 27)
            break;
       
        
        std::cout << "📹 카메라: " << camera_time << "ms | "
            << "🧠 추론: " << inference_time << "ms | " << std::endl;
    }


    
    return 0;
}