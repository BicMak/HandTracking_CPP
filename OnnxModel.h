#ifndef BOX_VISUALIZER_H
#define BOX_VISUALIZER_H

#include <string.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

struct Onnx_Outputs{
    std::vector<float> landmarks;
    std::vector<float> hand_type;
    std::vector<float> hand_score;
};



class Onnx_loader{
private:
    const ORTCHAR_T* model_path = L"D:/cpp_project/Mediapipe_practice/models/hand_landmark_sparse_Nx3x224x224.onnx";
    Ort::SessionOptions session_options;
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
    std::vector<float> input_buffer;
    std::vector<int64_t> input_shape = { 1, 3, 224, 224 };

    // 성능 최적화: 디버그 출력 제거, 인라인 처리
    std::vector<float> reshapeToNCHW(const cv::Mat& image) {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        std::vector<float> result;
        result.reserve(3 * 224 * 224);

        for (int c = 0; c < 3; c++) {
            float* ptr = (float*)channels[c].data;
            result.insert(result.end(), ptr, ptr + 224 * 224);
        }
        return result;
    }

public:
    Onnx_loader() :
        env(ORT_LOGGING_LEVEL_WARNING, "HandLandmark"),
        session(env, model_path, session_options),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        input_buffer(3 * 224 * 224) {

        session_options.SetIntraOpNumThreads(1);
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
        cv::resize(frame, processed, cv::Size(224, 224));
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
        processed.convertTo(processed, CV_32F, 1.0 / 255.0);

        std::cout << "크기 (Size): " << processed.size() <<std::endl;
        std::cout << "자료형 (type): " << processed.type() << std::endl;
        std::cout << "채널 (channel): " << processed.channels() << std::endl;
        input_buffer = reshapeToNCHW(processed);
    }

    Onnx_Outputs pred_pose() {
        // 텐서 생성
        std::cout << "추론 함수 호출!!" << std::endl;
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_buffer.data(),
            input_buffer.size(),
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = { "input" };
        const char* output_names[] = { "xyz_x21", "hand_score", "lefthand_0_or_righthand_1" };

        auto results = session.Run(Ort::RunOptions{},
            input_names, &input_tensor, 1,
            output_names, 3);

        Onnx_Outputs output;

        // 결과 복사
        auto landmarks_size = results[0].GetTensorTypeAndShapeInfo().GetElementCount();
        auto score_size = results[1].GetTensorTypeAndShapeInfo().GetElementCount();
        auto type_size = results[2].GetTensorTypeAndShapeInfo().GetElementCount();

        float* landmarks_ptr = results[0].GetTensorMutableData<float>();
        float* score_ptr = results[1].GetTensorMutableData<float>();
        float* type_ptr = results[2].GetTensorMutableData<float>();

        output.landmarks.assign(landmarks_ptr, landmarks_ptr + landmarks_size);
        output.hand_score.assign(score_ptr, score_ptr + score_size);
        output.hand_type.assign(type_ptr, type_ptr + type_size);

        return output;
    }
};

#endif