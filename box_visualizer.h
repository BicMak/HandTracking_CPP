

#include <string.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <algorithm>


#include "OnnxModel.h"

/**
 * @brief To save result hand landmark xy position
 *
 * @details
 * - x_point: land mark x axis location
 * - y_point: land mark y axis location
 */
struct point_locset {
    std::vector<int> x_point;
    std::vector<int> y_point;
};

/**
* @brief Visualizes bounding boxes and skeleton structures from inferred coordinate data
*
* Renders model predictions including skeletal structure and bounding box detection.
* Bounding boxes are generated using min/max values from 21 coordinate points,
* while skeletal structure is visualized by connecting joint points with lines.
*
* @author Marcus Kim
* @date 2025-08-25
* @version 1.0
*/
class BOX_DRAWING{
private:
    //Handmarker skeleton line index
    const std::array<std::array<int, 2>, 20> HAND_CONNECTIONS = { {
        {{0, 1}}, {{0, 5}}, {{0, 9}}, {{0, 13}}, {{0, 17}},
        {{1, 2}}, {{2, 3}}, {{3, 4}},
        {{5, 6}}, {{6, 7}}, {{7, 8}},
        {{9, 10}}, {{10, 11}}, {{11, 12}},
        {{13, 14}}, {{14, 15}}, {{15, 16}},
        {{17, 18}}, {{18, 19}}, {{19, 20}}
    } };

    //Handbox_color
    cv::Scalar LEFT_HAND_COLOR = cv::Scalar(0, 128, 128);    // 빨간색 (BGR 순서)
    cv::Scalar RIGHT_HAND_COLOR = cv::Scalar(0, 255, 0);   // 녹색
    cv::Scalar ROOTMARK_COLOR = cv::Scalar(255, 255, 0); 
    cv::Scalar LINE_COLOR = cv::Scalar(255, 0, 0);// 시안색  
    cv::Scalar TEXT_COLOR = cv::Scalar(255, 255, 255);     // 흰색

    //image size and matrix
    int img_width;
    int img_height;
    cv::Mat img;

    //predicted data
    point_locset hand_loc;
    int hand_direction;
    float hand_score;
    std::string hand_name; 



public:
    /**
    * @brief Updates the input image and retrieves its dimensions
    *
    * Stores the captured camera frame and extracts image width and height.
    * Validates that the input image is not empty before processing.
    *
    * @param input_img Captured image from camera (any size, BGR format, CV_8UC3)
    * @return None
    */
    void updateImage(const cv::Mat& input_img) {
        img = input_img;

        if (img.empty()) {
            std::cerr << "Error: You Need to update image" << std::endl;
        }
        else {
            img_width = img.cols;
            img_height = img.rows;

            std::cout << "Image updated successfully: "
                << img_width << "x" << img_height << std::endl;
        }
    }

    /**
    * @brief Updates detected hand landmarks from inference results
    *
    * Processes ONNX model inference output and extracts hand landmark predictions.
    * Converts output tensors to point structures and extracts only x,y coordinates.
    *
    * @param onnx_data MediaPipe inference result containing hand landmark data
    * @return None
    *
    * @pre updateImage() must be called first to set the base image
    */
    void updatehandpos(Onnx_Outputs onnx_data) {
        hand_loc.x_point.clear();
        hand_loc.y_point.clear();

        hand_direction = (int)onnx_data.hand_type[0];
        hand_score = onnx_data.hand_score[0];

        for (int i = 0; i < 21; i++) {
            int idx1 = 3 * i;
            int idx2 = 3 * i + 1;
            float temp_x = (1-onnx_data.landmarks[idx1]/224) * img_width;
            float temp_y = (onnx_data.landmarks[idx2]/224) * img_height;  
            hand_loc.x_point.push_back(int(temp_x));
            hand_loc.y_point.push_back(int(temp_y));
        }



    }

    /**
    * @brief Draws bounding box around detected hand
    *
    * Renders bounding box with color coding based on hand position (left or right).
    * Also displays hand label text above the bounding box.
    *
    * @param None
    * @return None
    *
    * @pre updatehandpos() must be called before drawing
    */
    void drawbox() {
        cv::Scalar color;
        std::string hand_text;
        if (hand_direction == 0) {
            color = LEFT_HAND_COLOR;
            hand_text = "LEFT_HAND";
        }
        else {
            color = RIGHT_HAND_COLOR;
            hand_text = "RIGHT_HAND";
        }
        auto x_min = std::min_element(hand_loc.x_point.begin(), hand_loc.x_point.end());
        auto x_max = std::max_element(hand_loc.x_point.begin(), hand_loc.x_point.end());
        auto y_min = std::min_element(hand_loc.y_point.begin(), hand_loc.y_point.end());
        auto y_max = std::max_element(hand_loc.y_point.begin(), hand_loc.y_point.end());


        int x1 = *x_min;  // 9
        int x2 = *x_max;  // 9
        int y1 = *y_min;  // 9
        int y2 = *y_max;  // 9

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        
        double fontScale = 0.5;
        int thickness = 1;

        cv::Size textSize = cv::getTextSize(hand_text, cv::FONT_HERSHEY_SIMPLEX, 
            fontScale, thickness, nullptr);
        cv::rectangle(img, cv::Rect(x1, y1 - textSize.height, textSize.width, textSize.height), color, -1);

        cv::putText(img, hand_text, cv::Point(x1, y1), 
                     cv::FONT_HERSHEY_SIMPLEX, fontScale, TEXT_COLOR, thickness);
    }

    /**
    * @brief Draws hand skeleton structure
    *
    * Renders hand skeleton by drawing joint points as circles and
    * connecting them with lines based on hand connection patterns.
    *
    * @param None
    * @return None
    *
    * @pre updatehandpos() must be called before drawing
    */
    void drawskeleton() {

        for (int i = 0; i < 21; i++) {
            int x_center = hand_loc.x_point[i];
            int y_center = hand_loc.y_point[i];
            circle(img, cv::Point(x_center, y_center), 5, ROOTMARK_COLOR, cv::FILLED);
        }

        for (auto line_pts : HAND_CONNECTIONS) {

            int x1 = hand_loc.x_point[line_pts[0]];
            int y1 = hand_loc.y_point[line_pts[0]];
            int x2 = hand_loc.x_point[line_pts[1]];
            int y2 = hand_loc.y_point[line_pts[1]];
          

            cv::Point Pt1 = cv::Point(x1, y1);
            cv::Point Pt2 = cv::Point(x2, y2);
            line(img, Pt1, Pt2, LINE_COLOR, 2);
        }
 
    }


    /**
    * @brief Processes and renders complete hand visualization
    *
    * Combines bounding box and skeleton drawing if hand confidence score
    * exceeds threshold (0.4). Returns the final rendered image.
    *
    * @param None
    * @return cv::Mat Rendered image with hand visualizations
    *
    * @pre updatehandpos() must be called before processing
    */
    cv::Mat process(){
        if (hand_score > 0.4) {
            this->drawbox();
            this->drawskeleton();
        }

        return img;
    }

};