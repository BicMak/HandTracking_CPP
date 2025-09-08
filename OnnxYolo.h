/*
* Todo : Onnx session_options 다시 확인해보기
*    
*    Environment: 전역 ONNX 런타임 설정
*    SessionOptions: 세션별 성능/동작 설정
*    Session: 실제 모델 로드 (옵션 적용됨)
*    MemoryInfo: 텐서 메모리 관리 (독립적)
*    
*   SetIntraOpNumThreads : 단일 노드 내부에서 병렬처리
*   SetInterOpNumThreads : 독립적인 연산을 병렬처리
*   --------------EXAMPLE---------------------------
*     신경망 그래프에서 독립적인 연산들
*      Conv1 → ReLU1 ↘
*                      → Concat → Output
*      Conv2 → ReLU2 ↗
*      InterOp = 2라면:
*      Thread A: Conv1 → ReLU1 실행
*      Thread B: Conv2 → ReLU2 실행 (동시에!)
*
*
* Todo : data input 확인해보기
*    1. 매 호출마다 메모리공간을 새로할당하여 변수를 생성하는 과정 삭제 (static이든 미리만들든)
*    2. cvt color와 normalize에 대한 최적화 연산하기
*/

#pragma once

#include <string>
#include <iostream>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

/**
 * @brief To save result predicted data and return at once
 *
 * @details
 * - x: bounding box x-center
 * - y: bounding box x-center
 * - w: bounding box width
 * - h: bounding box height
 * - confidence: detection confidence score (0.0 ~ 1.0)
 * - class_id: save classificated hand gesture
 */
struct Detection {
    float x, y, w, h;         
    float confidence;       
    int class_id;             
};

/**
 * @brief ONNX-based Hagrid YOLO v10 hand gesture detection class
 *
 * This class provides hand gesture detection functionality using the
 * Hagrid YOLO v10n model. It leverages ONNX Runtime for cross-platform
 * inference and supports real-time gesture recognition.
 * (https://github.com/hukenovs/hagrid)
 * 
 * @author Marcus Kim
 * @date 2025-08-25
 * @version 1.0
 */
class Yolo_loader {
private:
    const ORTCHAR_T* model_path = L"D:/cpp_project/Mediapipe_practice/models/yolo_hand_detection_Nx3x224x224.onnx";
    int input_widht = 640;
    int input_height = 640;
    Ort::SessionOptions session_options;
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;

    cv::Mat resized_mat;
    cv::Mat float_mat;


    Ort::MemoryInfo memory_info;
    std::vector<float> input_buffer;
    std::vector<int64_t> input_shape = { 1, 3, input_widht, input_height };
    std::vector<float> raw_output;
    

    std::vector<Detection> result_shape;

    /**
    * @brief Convert cv::Mat type to ONNX tensor input shape format
    *
    * Converts OpenCV's HWC (Height-Width-Channel) format to
    * NCHW (Batch-Channel-Height-Width) format required by ONNX model.
    *
    * @param image Preprocessed input image (anysize, CV_32F)
    * @return result NCHW format float vector (1x3x640x640, float)
    */
    std::vector<float> reshapeToNCHW(const cv::Mat& image) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        std::vector<float> result;
        result.reserve(3 * input_widht * input_height);

        for (int c = 0; c < 3; c++) {
            float* ptr = (float*)channels[c].data;
            result.insert(result.end(), ptr, ptr + input_widht * input_height);
        }
        return result;
    }


    void convertBGRToRGBFloat(const cv::Mat& bgr_src, cv::Mat& rgb_float_dst) {
        const uint8_t* src = bgr_src.ptr<uint8_t>();
        float* dst = rgb_float_dst.ptr<float>();
        const int total_pixels = bgr_src.rows * bgr_src.cols;
        const float scale = 1.0f / 255.0f;

        for (int i = 0; i < total_pixels; i++) {
            dst[i * 3 + 0] = src[i * 3 + 2] * scale;  // B→R
            dst[i * 3 + 1] = src[i * 3 + 1] * scale;  // G→G
            dst[i * 3 + 2] = src[i * 3 + 0] * scale;  // R→B
        }
    }
public:
    Yolo_loader() :
        env(ORT_LOGGING_LEVEL_WARNING, "HandDetect"),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        input_buffer(3 * input_height * input_widht) {

        resized_mat = cv::Mat(input_height, input_widht, CV_8UC3);
        float_mat = cv::Mat(input_height, input_widht, CV_32FC3);
        // SessionOptions 설정 (기존 코드 그대로)
        session_options.SetIntraOpNumThreads(8);
        session_options.SetInterOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        session_options.EnableCpuMemArena();
        session_options.EnableMemPattern();

        // ✅ Session 생성 (기존 코드 살짝 수정)
        session = std::make_unique<Ort::Session>(env, model_path, session_options);
    }

    /**
    * @brief Acquire input image and store in ONNX model input buffer
    *
    * Preprocesses input image to meet MediaPipe model requirements:
    * 1. Resize to 640x640
    * 2. Convert BGR → RGB
    * 3. Normalize [0,255] → [0,1]
    * 4. Transform HWC → NCHW format
    *
    * @param frame Captured image from camera (any size, BGR, CV_8UC3)
    * @return None (result stored in internal input_buffer)
    */
    void get_data(cv::Mat frame) {

        if (frame.empty()) {
            std::cerr << "ERROR: 입력 프레임이 비어있습니다!" << std::endl;
            return;
        }

        cv::resize(frame, resized_mat, cv::Size(input_widht, input_height));

        convertBGRToRGBFloat(resized_mat, float_mat);

        input_buffer = reshapeToNCHW(float_mat);
    }

    /**
     * @brief Perform inference on buffered image data
     *
     * Runs ONNX model inference and extracts hand landmark predictions:
     * 1. Create input tensor from buffer
     * 2. Execute model inference
     * 3. Extract and convert output tensors to float vectors
     *
     * @param None
     * @return Onnx_Outputs structure containing bboxes, class id, and confidence score 
     *
     * @pre get_data() must be called first to prepare input buffer
     */
    Detection pred_pose() {

            //std::cout << "추론 함수 호출!!" << std::endl;
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_buffer.data(),
                input_buffer.size(),
                input_shape.data(),
                input_shape.size()
            );
            //std::cout << "텐서 생성 완료" << std::endl;

            const char* input_names[] = { "images" };
            const char* output_names[] = { "output0" };  // 또는 실제 출력 이름

            //std::cout << "추론 시작..." << std::endl;
            auto results = session->Run(Ort::RunOptions{},
                input_names, &input_tensor, 1,
                output_names, 1);  // 이제 일치함
            //std::cout << "추론 완료!" << std::endl;

            auto result_size = results[0].GetTensorTypeAndShapeInfo().GetElementCount();
            float* result_ptr = results[0].GetTensorMutableData<float>();  // 타이포 수정
            raw_output.assign(result_ptr, result_ptr + result_size);

            Detection return_value = this->SupressNonmax(results);
            //std::cout << "Class: " << return_value.class_id << std::endl;
            return return_value;

    }

    /**
    * @brief Suppress non-maximum detections and return highest confidence bounding box
    *
    * Processes YOLO model inference results to find the detection with maximum confidence score.
    * Filters out detections below confidence threshold (0.3) and returns the best detection.
    *
    * @param results YOLO model inference output tensor [1, 300, 6] format
    *                Format: [x, y, w, h, confidence, class_id] for each detection
    * @return Detection structure containing best bounding box with highest confidence
    *
    * @pre Model inference must be completed and results tensor must be valid
    * @note Currently implements simplified NMS by selecting maximum confidence detection only
    */
    Detection SupressNonmax(std::vector<Ort::Value>& results) {
        // 출력 텐서에서 데이터 포인터 가져오기
        float* output_data = results[0].GetTensorMutableData<float>();

        // 텐서 shape 정보 가져오기
        auto tensor_info = results[0].GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        // YOLO 출력 형태: [1, 300, 6] 또는 [1, anchor_count, 6]
        int batch_size = static_cast<int>(shape[0]);      // 1
        int anchor_count = static_cast<int>(shape[1]);    // 300
        int detection_size = static_cast<int>(shape[2]);  // 6 (x, y, w, h, conf, class)

        Detection best_detection;
        best_detection.confidence = 0.0f;
        best_detection.class_id = -1;
        best_detection.x = best_detection.y = best_detection.w = best_detection.h = 0.0f;

        
        int best_idx = -1;
        // 모든 앵커 박스를 순회하면서 최고 confidence 찾기
        for (int i = 0; i < anchor_count; i++) {
            float confidence = output_data[i * detection_size + 4];
            if (confidence > 0.3f && confidence > best_detection.confidence) {
                best_detection.confidence = confidence;
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            int base_idx = best_idx * detection_size;
            best_detection.x = output_data[base_idx + 0];
            best_detection.y = output_data[base_idx + 1];
            best_detection.w = output_data[base_idx + 2];
            best_detection.h = output_data[base_idx + 3];
            // confidence는 이미 루프에서 설정했으므로 생략 가능
            best_detection.class_id = static_cast<int>(output_data[base_idx + 5]);  // int로 캐스팅
        }

        return best_detection;
    }
};


