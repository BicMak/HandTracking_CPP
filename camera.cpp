// 간단한 이미지 생성
cv::Mat img = cv::Mat::zeros(200, 200, CV_8UC3);
cv::circle(img, cv::Point(100, 100), 50, cv::Scalar(0, 255, 0), -1);

std::cout << "OpenCV 이미지 생성 성공! 크기: " << img.rows << "x" << img.cols << std::endl;

// 2. LibTorch 테스트
torch::Tensor tensor = torch::rand({ 2, 3 });
std::cout << "LibTorch 텐서 생성 성공!" << std::endl;
std::cout << "텐서:\n" << tensor << std::endl;

// 3. CUDA 확인
if (torch::cuda::is_available()) {
    std::cout << "CUDA 사용 가능!" << std::endl;
}
else {
    std::cout << "CPU만 사용 가능" << std::endl;
}

// 4. 이미지 표시 (선택사항)
cv::imshow("Test Image", img);
std::cout << "이미지 창이 열렸습니다. 아무 키나 누르세요." << std::endl;
cv::waitKey(0);
cv::destroyAllWindows();

std::cout << "테스트 완료!" << std::endl;
return 0;