#pragma once

#include <string>
#include <iostream>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


struct Detection {
    float x, y, w, h;           // bbox
    float confidence;           // 최고 클래스 확률
    int class_id;              // 클래스 ID
};

class Yolo_loader {
private:
    const ORTCHAR_T* model_path = L"D:/cpp_project/Mediapipe_practice/models/yolo_hand_detection_Nx3x224x224.onnx";
    int input_widht = 640;
    int input_height = 640;
    Ort::SessionOptions session_options;
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
    std::vector<float> input_buffer;
    std::vector<int64_t> input_shape = { 1, 3, input_widht, input_height };
    std::vector<float> raw_output;
    

    std::vector<Detection> result_shape;

    // 성능 최적화: 디버그 출력 제거, 인라인 처리
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

public:
    Yolo_loader() :
        env(ORT_LOGGING_LEVEL_WARNING, "HandDetect"),
        session(env, model_path, session_options),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        input_buffer(3 * input_height * input_widht) {

        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }

    void get_data(cv::Mat frame) {

        if (frame.empty()) {
            std::cerr << "ERROR: 입력 프레임이 비어있습니다!" << std::endl;
            return;
        }

        if (frame.rows == 0 || frame.cols == 0) {
            std::cerr << "ERROR: 프레임 크기가 0입니다: " << frame.size() << std::endl;
            return;
        }
        cv::Mat processed;
        cv::resize(frame, processed, cv::Size(input_widht, input_height));
        std::cout << processed.size() << std::endl;

        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
        processed.convertTo(processed, CV_32F, 1.0 / 255.0);

        input_buffer = reshapeToNCHW(processed);
    }

    Detection pred_pose() {
        try {
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
            auto results = session.Run(Ort::RunOptions{},
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
        catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime 에러: " << e.what() << std::endl;
            return Detection{};  // 기본 Detection 반환
        }
        catch (const std::exception& e) {
            std::cerr << "일반 에러: " << e.what() << std::endl;
            return Detection{};  // 기본 Detection 반환
        }
    }

    // pred_pose() 함수 내에서 호출할 때:
    // Detection return_value = this->SupressNonmax(results);

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


