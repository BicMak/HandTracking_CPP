
#include <windows.h>
#include <string.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


#include "OnnxModel.h"


struct normal_point_locset {
    float x_point;
    float y_point;
};

class Mouse_event {
private:
    //스크린 정보
    int screen_width;
    int screen_height;
    cv::Mat img;
    //스크린 정보
    std::vector<normal_point_locset> hand_normal_loc;
    int hand_direction;
    float hand_score;
    
    POINT cursorPos;



public:
    Mouse_event() {
        GetCursorPos(&cursorPos);
        screen_width = GetSystemMetrics(SM_CXSCREEN);
        screen_height = GetSystemMetrics(SM_CYSCREEN);
    }

    void updatehandpos(Onnx_Outputs onnx_data) {
        normal_point_locset temp;

        hand_normal_loc.clear();

        hand_direction = (int)onnx_data.hand_type[0];
        hand_score = onnx_data.hand_score[0];

        // Normalize Point coordinates
        for (int i = 0; i < 21; i++) {
            int idx1 = 3 * i;
            int idx2 = 3 * i + 1;
            temp.x_point = (1 - onnx_data.landmarks[idx1] / 224);
            temp.y_point = (onnx_data.landmarks[idx2] / 224);
            hand_normal_loc.push_back(temp);
        }

    }

    void drawbox() {

    }

    void process() {
        int mouse_x = screen_width * hand_normal_loc[8].x_point;
        int mouse_y = screen_width * hand_normal_loc[8].y_point;
        SetCursorPos(mouse_x, mouse_y);
    }

};