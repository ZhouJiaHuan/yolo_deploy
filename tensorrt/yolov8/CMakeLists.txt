cmake_minimum_required(VERSION 3.9)
project(yolov8_trt)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda (modified to your cuda path)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt (modified to your tensorRT path)
include_directories(/opt/TensorRT-8.5.1.7/include)
link_directories(/opt/TensorRT-8.5.1.7/lib)

add_executable(yolov8_trt src/main.cpp src/yolov8_trt.cpp)
target_link_libraries(yolov8_trt nvinfer cudart ${OpenCV_LIBS})

