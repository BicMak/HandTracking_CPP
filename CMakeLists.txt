cmake_minimum_required(VERSION 3.10)
# 지원되는 경우 MSVC 컴파일러에 대해 핫 다시 로드 사용하도록 설정합니다.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()
project("Mediapipe_practice")

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV 설정
set(OpenCV_DIR "D:/library/opencv/build")
find_package(OpenCV CONFIG REQUIRED)

# LibTorch 설정
set(Libtorch_DIR "D:/library/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${Libtorch_DIR})
find_package(Torch REQUIRED)

# ONNX Runtime 설정 (추가된 부분)
set(ONNXRUNTIME_ROOT_PATH "D:/library/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/onnxruntime.lib)

# 컴파일러 플래그 설정
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 실행 파일 생성

add_executable(Mediapipe_practice 
    "Mediapipe_practice.cpp" 
    
    "OnnxModel.h"
 "box_visualizer.h" "Mouse_event.h" "OnnxYolo.h" )

# 헤더 파일 경로 추가 (추가된 부분)
target_include_directories(Mediapipe_practice PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})

# 라이브러리 링크
target_link_libraries(Mediapipe_practice
    PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${ONNXRUNTIME_LIB}  # 추가된 부분
)

# MSVC용 설정
if(MSVC)
    target_compile_options(Mediapipe_practice PRIVATE /bigobj)
    
    # LibTorch DLL 복사
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET Mediapipe_practice
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:Mediapipe_practice>)
    
    # ONNX Runtime DLL 복사 (추가된 부분)
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT_PATH}/lib/*.dll")
    add_custom_command(TARGET Mediapipe_practice
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${ONNXRUNTIME_DLLS}
                       $<TARGET_FILE_DIR:Mediapipe_practice>)
endif()

# C++ 표준 설정 (백업)
set_property(TARGET Mediapipe_practice PROPERTY CXX_STANDARD 20)