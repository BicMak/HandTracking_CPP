#pragma once

#include <windows.h>
#include <string.h>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "OnnxModel.h"
#include "OnnxYolo.h"

using TimePoint = std::chrono::high_resolution_clock::time_point;

/**
* @brief Structure to store normalized hand landmark coordinates
*
* @details This structure holds a single normalized 2D point representing
* hand landmark positions. The coordinates are typically normalized to [0,1] range
* where (0,0) represents the top-left corner and (1,1) represents the bottom-right
* corner of the detection frame.
*
* @param vec Normalized 2D vector containing (x, y) coordinates of a hand landmark point
*/
struct normal_point_locset {
    cv::Vec2f vec;
};


/**
 * @brief Controls mouse cursor using Windows API
 *
 * This class receives MediaPipe hand landmark detection results
 * and translates them into mouse movements and clicks.
 *
 * @author Marcus Kim
 * @date 2025-08-31
 * @version 1.0
 */
class Mouse_event {
private:
    //screen infomation
    int screen_width;
    int screen_height;
    cv::Mat img;
    
    //vector point variable
    std::vector<normal_point_locset> hand_normal_loc;
    int hand_id;
    float hand_score;
    POINT cursorPos;
    std::vector<double> tip_dist;
    cv::Vec2f yolo_pivot;
    cv::Vec2f fingertip_pivot;

    cv::Vec2f bbox_curpose;
    cv::Vec2f starting_point;

    //event code
    std::string envet_code;
    cv::Vec2f current_pos;
    bool left_click_flag;
    TimePoint start_time = std::chrono::high_resolution_clock::now();;

    //curser Moving parameter
    float DEAD_ZONE = 3.0f;
    float ALPHA = 0.25f;
    float smooth_x = -1, smooth_y = -1;
    int center_x;
    int center_y;
    int cursor_x = 0;
    int cursor_y = 0;

public:
    /**
     * @brief Constructor for Mouse_event class
     *
     * Initializes the mouse event handler by:
     * - Getting current cursor position
     * - Retrieving main screen dimensions
     * - Calculating screen center coordinates for reference
     */
    Mouse_event() {
        GetCursorPos(&cursorPos);
        screen_width = GetSystemMetrics(SM_CXSCREEN);
        screen_height = GetSystemMetrics(SM_CYSCREEN);
        center_x = screen_width / 2;
        center_y = screen_height / 2;
    }

    /**
     * @brief Update hand position data from ONNX prediction results
     *
     * Processes MediaPipe hand landmark predictions and YOLO hand detection results:
     * 1. Extracts and stores YOLO bounding box center coordinates and hand class ID
     * 2. Normalizes MediaPipe joint coordinates (21 landmarks) with horizontal flip
     *    - Coordinates are normalized to [0,1] range from 224x224 input
     *    - X coordinates are horizontally flipped (1 - x/224)
     *
     * @param onnx_data MediaPipe prediction tensor containing 63 values (21 landmarks × 3)
     * @param onnx_yolodata HAGRID YOLO10s detection result with bbox and classification
     */
    void updatehandpos(Onnx_Outputs onnx_data,
                       Detection onnx_yolodata) {
        normal_point_locset temp;

        hand_normal_loc.clear();

        hand_score = onnx_yolodata.confidence;
        hand_id = onnx_yolodata.class_id;
        bbox_curpose[0] = (1-onnx_yolodata.x/640.0f);
        bbox_curpose[1] = onnx_yolodata.y / 640.0f;
        std::cout << bbox_curpose[0] << std::endl;
        std::cout << bbox_curpose[1] << std::endl;

        // Normalize Point coordinates
        for (int i = 0; i < 21; i++) {
            int idx1 = 3 * i;
            int idx2 = 3 * i + 1;
            temp.vec =  cv::Vec2f((1 - onnx_data.landmarks[idx1] / 224),
                                  (onnx_data.landmarks[idx2] / 224) );
            hand_normal_loc.push_back(temp);
        }
    }


    /**
     * @brief Update hand position data from ONNX prediction results
     *
     * Processes MediaPipe hand landmark predictions and YOLO hand detection results:
     * 1. Extracts and stores YOLO bounding box center coordinates and hand class ID
     *    - Yolo coordinates normalized to [0,1] range from 640x640 input
     *    - Yolo X coordinates are horizontally flipped (1 - x/640)
     * 2. Normalizes MediaPipe joint coordinates (21 landmarks) with horizontal flip
     *    - Coordinates are normalized to [0,1] range from 224x224 input
     *    - X coordinates are horizontally flipped (1 - x/224)
     *
     * @param onnx_data MediaPipe prediction tensor containing 63 values (21 landmarks × 3)
     * @param onnx_yolodata HAGRID YOLO10s detection result with bbox and classification
     */
    void mouse_moving() {
        
        int scailer = 2;
        float offset_x = (hand_normal_loc[8].vec[0] - 0.5f) * scailer * screen_width;
        float offset_y = (hand_normal_loc[8].vec[1] - 0.5f) * scailer * screen_height;
        float raw_x = center_x + offset_x;
        float raw_y = center_y + offset_y;

        // 🔧 드래그 중이 아닐 때만 YOLO 기준점 업데이트
        if (smooth_x < 0) {
            smooth_x = raw_x;
            smooth_y = raw_y;
            cursor_x = (int)std::clamp(raw_x, 0.0f,(float)screen_width);
            cursor_y = (int)std::clamp(raw_y, 0.0f, (float)screen_height);

            // 🔧 fingertip_pivot도 드래그 중이 아닐 때만 업데이트
            if (!left_click_flag) {
                fingertip_pivot = cv::Vec2f(hand_normal_loc[8].vec[0], hand_normal_loc[8].vec[1]);
            }

            SetCursorPos(cursor_x, cursor_y);
        }
        else {
            smooth_x = smooth_x * (1.0f - ALPHA) + raw_x * ALPHA;
            smooth_y = smooth_y * (1.0f - ALPHA) + raw_y * ALPHA;

            float dx = smooth_x - cursor_x;
            float dy = smooth_y - cursor_y;

            float distance = sqrt(dx * dx + dy * dy);

            if (distance > DEAD_ZONE) {
                cursor_x = (int)std::clamp(smooth_x, 0.0f, (float)screen_width);
                cursor_y = (int)std::clamp(smooth_y, 0.0f, (float)screen_height);

                // 🔧 fingertip_pivot도 드래그 중이 아닐 때만 업데이트
                if (!left_click_flag) {
                    fingertip_pivot = cv::Vec2f(hand_normal_loc[8].vec[0], hand_normal_loc[8].vec[1]);
                }

                SetCursorPos(cursor_x, cursor_y);
            }

        }
    }


    /**
     * @brief Control left mouse press with time-based click/drag classification
     *
     * Determines user intent based on gesture duration and executes accordingly:
     *
     * **Click Classification** (automatic completion):
     * - Detects brief gestures (< 0.5s duration)
     * - Automatically sends MOUSEEVENTF_LEFTUP in mouse_leftoff()
     * - Completes interaction when gesture is released quickly
     *
     * **Drag Classification** (progressive operation):
     * - Identifies sustained press gestures (≥ 0.5s duration)
     * - Enables continuous cursor tracking based on YOLO bbox center
     * - Maintains drag state until corresponding mouse_leftoff() call
     * - Updates cursor position in real-time during drag progression
     *
     * **Classification Logic**:
     * - Initial gesture: Always sends MOUSEEVENTF_LEFTDOWN
     * - After 500ms: Activates drag mode with cursor tracking with yolo bbox center pos
     * - Cursor follows hand movement using absolute positioning
     *
     * @note This approach uses time-based classification for simplicity
     * @warning 500ms delay before drag activation may feel slightly sluggish
     * @see mouse_leftoff() for drag completion and click finalization
     */
    void mouse_lefton() {
        float scailer = 1.5;
        auto current_time = std::chrono::high_resolution_clock::now();
        
        if (left_click_flag == FALSE) {
            left_click_flag = TRUE;
            starting_point = fingertip_pivot;  // 화면 픽셀 좌표로 저장됨
            start_time = std::chrono::high_resolution_clock::now();

            yolo_pivot = bbox_curpose;
            mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
        }
        else if (left_click_flag == TRUE) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time
            ).count();

            if (duration >= 500) {
                // 🔧 YOLO 바운딩박스 센터 기반 절대좌표 계산
                float offset_x = (bbox_curpose[0] - 0.5f) * scailer * screen_width;
                float offset_y = (bbox_curpose[1] - 0.5f) * scailer * screen_height;

                float raw_x = center_x + offset_x;
                float raw_y = center_y + offset_y;


                // 직접 커서 이동 
                cursor_x = (int)std::clamp(raw_x, 0.0f, (float)screen_width);
                cursor_y = (int)std::clamp(raw_y, 0.0f, (float)screen_height);
                SetCursorPos(cursor_x, cursor_y);
            }
        }
    }

    /**
     * @brief Complete mouse operations (both click and drag)
     *
     * Handles final LEFTUP event for all mouse interactions
     *
     * **Common Functionality**:
     * - Always resets left_click_flag to FALSE
     * - Provides operation completion feedback
     * - Safely handles calls when no active operation exists
     *
     * @note This function handles LEFTUP for both clicks and drags
     * @see mouse_lefton() for operation initiation and drag tracking
     */
    void mouse_leftoff() {
        if (left_click_flag) {
            left_click_flag = FALSE;
            mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
            std::cout << "🖱️ 드래그 종료" << std::endl;
        }

    }


    /**
    * @brief Process hand gesture recognition and execute corresponding mouse actions
    *
    * Main processing pipeline that interprets YOLO hand detection results and
    * triggers appropriate mouse control functions based on gesture classification:
    *
    * **Gesture-to-Action Mapping**:
    * - Class 11/19 (Pointing): Cursor movement via mouse_moving()
    * - Class 14 (Fist): Mouse press/drag initiation via mouse_lefton()
    * - Class 20 (Open Palm): Mouse release/operation completion via mouse_leftoff()
    *
    * **Quality Control**:
    * - Minimum confidence threshold: 0.4 (40%)
    * - Low confidence gestures are ignored to prevent false triggers
    * - Active drag operations are safely terminated if hand detection fails
    *
    * **Safety Features**:
    * - Automatic drag termination on detection loss
    * - Prevents stuck drag states from unstable hand tracking
    * - Graceful degradation when hand visibility is poor
    *
    * @note This function should be called once per frame in the main loop
    * @warning Confidence threshold (0.4) may need adjustment based on lighting conditions
    * @see mouse_moving(), mouse_lefton(), mouse_leftoff() for individual gesture handlers
    */
    void process() {
        if (hand_score > 0.4) {
            switch (hand_id) {
            case 11:  // Pointing - 마우스 이동
                mouse_moving();
                break;
            case 19:  // Pointing - 마우스 이동
                mouse_moving();
                break;
            case 14:  // Fist - 드래그
                mouse_lefton();
                break;
            case 20:  // Open Palm - 드래그 종료
                mouse_leftoff();
                break;
            default:
                break;
            }
        }
        else {
            if (left_click_flag) {
                std::cout << "⚠️ 손 인식 불안정으로 드래그 강제 종료" << std::endl;
                mouse_leftoff();
            }
        }
    }

};

