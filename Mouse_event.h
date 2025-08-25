#pragma once

#include <windows.h>
#include <string.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "OnnxModel.h"
#include "OnnxYolo.h"

struct normal_point_locset {
    cv::Vec2f vec;
};

class Mouse_event {
private:
    //스크린 정보
    int screen_width;
    int screen_height;
    cv::Mat img;
    
    //좌표 정보
    std::vector<normal_point_locset> hand_normal_loc;
    int hand_id;
    float hand_score;
    POINT cursorPos;
    std::vector<double> tip_dist;

    //event code
    std::string envet_code;
    bool left_click_flag;
    double fist_start_time;



public:
    Mouse_event() {
        GetCursorPos(&cursorPos);
        screen_width = GetSystemMetrics(SM_CXSCREEN);
        screen_height = GetSystemMetrics(SM_CYSCREEN);
    }

    void updatehandpos(Onnx_Outputs onnx_data,
                       Detection onnx_yolodata) {
        normal_point_locset temp;

        hand_normal_loc.clear();

        hand_score = onnx_yolodata.confidence;
        hand_id = onnx_yolodata.class_id;

        // Normalize Point coordinates
        for (int i = 0; i < 21; i++) {
            int idx1 = 3 * i;
            int idx2 = 3 * i + 1;
            temp.vec =  cv::Vec2f((1 - onnx_data.landmarks[idx1] / 224),
                                  (onnx_data.landmarks[idx2] / 224) );
            hand_normal_loc.push_back(temp);
        }
    }

    void mouse_moving() {

        static float DEAD_ZONE = 3.0f;
        static float ALPHA = 0.25f;

        static float smooth_x = -1, smooth_y = -1;
        static int cursor_x, cursor_y;

        static int center_x = screen_width / 2;
        static int center_y = screen_height / 2;
        
        float offset_x = (hand_normal_loc[8].vec[0]-0.5)*2* screen_width;
        float offset_y = (hand_normal_loc[8].vec[1]-0.5)*2* screen_height;

        float raw_x = center_x + offset_x;
        float raw_y = center_y + offset_y;

        if (smooth_x < 0) {
            smooth_x = raw_x;
            smooth_y = raw_y;
            cursor_x = (int)raw_x;
            cursor_y = (int)raw_y;
            SetCursorPos(cursor_x, cursor_y);
            return;
        }

        // 스무딩 적용
        smooth_x = smooth_x * (1.0f - ALPHA) + raw_x * ALPHA;
        smooth_y = smooth_y * (1.0f - ALPHA) + raw_y * ALPHA;

        // 데드존 체크
        float dx = smooth_x - cursor_x;
        float dy = smooth_y - cursor_y;
        float distance = sqrt(dx * dx + dy * dy);

        if (distance > DEAD_ZONE) {
            cursor_x = (int)smooth_x;
            cursor_y = (int)smooth_y;
            SetCursorPos(cursor_x, cursor_y);
        }
    }

    void mouse_lefton() {
        static int click_cnt = 0;
        if (left_click_flag == FALSE) {
            click_cnt += 1;
            if (click_cnt > 0) {
                left_click_flag = TRUE;
                mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
                click_cnt = 0;
            }
            
        }
        else {
            return;
        }
        
    }

    void mouse_leftoff() {
        static int click_cnt = 0;
        if (left_click_flag) {
            click_cnt += 1;
            if (click_cnt > 0) {
                left_click_flag = FALSE;
                mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
                click_cnt = 0;
            }
            
        }
        else {
            return;
        }
        
    }

    void process() {
        if (hand_score > 0.4) {
            switch (hand_id) {
            case 14:
                mouse_lefton();
                break;
            case 20:
                mouse_leftoff();
                break;
            case 11:
                mouse_moving();
                break;
                
            }
            
        }
    }

};

