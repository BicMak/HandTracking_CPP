#include <string.h>
#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>




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

struct Outputs{
	std::vector<float> landmarks;
	std::vector<float> hand_score;
	std::vector<float> hand_type;
};

class Onnx_loader {
private:
	const ORTCHAR_T* model_path = L"D:/cpp_project/Mediapipe_practice/models/hand_landmark_sparse_Nx3x224x224.onnx";
	Ort::SessionOptions session_options;
	Ort::Env env;
	Ort::Value input_tensor;
	cv::Mat input_data;
	Outputs result_data;
	Ort::Session session{ env, L"model.onnx", session_options };

	std::vector<float> reshapeToNCHW(const cv::Mat& image) {
		// 채널 분리 후 재배열
		std::vector<cv::Mat> channels;
		cv::split(image, channels);

		std::vector<float> result;
		result.reserve(1 * 3 * 224 * 224);

		// NCHW 순서: N=1, C=3, H=224, W=224
		for (int c = 0; c < 3; c++) {
			float* ptr = (float*)channels[c].data;
			result.insert(result.end(), ptr, ptr + 224 * 224);
		}

		return result;  // [1, 3, 224, 224] 형태
	}


public:
	Onnx_loader() {}

	void get_data(cv::Mat frame ) {
		std::vector<float> reshaped_image;

		//frame preprocessing
		input_data = frame;
		cv::resize(frame, input_data, cv::Size(224, 224));
		cv::cvtColor(input_data, input_data, cv::COLOR_BGR2RGB);
		input_data.convertTo(input_data, CV_32F, 1.0 / 255.0);

		// 메모리 정보
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		reshaped_image = reshapeToNCHW(input_data);

		// 텐서 생성
		std::vector<int64_t> input_shape = { 1, 3, 224, 224 };
		input_tensor = Ort::Value::CreateTensor<float>(
			memory_info,
			reshaped_image.data(),
			reshaped_image.size(),
			input_shape.data(),
			input_shape.size()
		);

	}

	Outputs pred_pose() {
		std::vector<const char*> input_names = { "input" };
		std::vector<const char*> output_names = { 
			"xyz_x21",                    // 손 랜드마크 좌표 [batch, 63]
			"hand_score",                 // 손 감지 점수 [batch, 1]  
			"lefthand_0_or_righthand_1"   // 왼손(0)/오른손(1) [batch, 1]
		};

		// 추론 실행
		auto result = session.Run(Ort::RunOptions{},
			input_names.data(), &input_tensor, 1,
			output_names.data(), 3);

		float* landmarks_ptr = result[0].GetTensorMutableData<float>();
		float* score_ptr = result[1].GetTensorMutableData<float>();
		float* type_ptr = result[2].GetTensorMutableData<float>();

		size_t landmarks_size = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
		size_t score_size = result[1].GetTensorTypeAndShapeInfo().GetElementCount();
		size_t type_size = result[2].GetTensorTypeAndShapeInfo().GetElementCount();
		
		result_data.landmarks.assign(landmarks_ptr, landmarks_ptr + landmarks_size);
		result_data.hand_score.assign(score_ptr, score_ptr + score_size);
		result_data.hand_type.assign(type_ptr, type_ptr + type_size);
		
		return result_data;
	}

};