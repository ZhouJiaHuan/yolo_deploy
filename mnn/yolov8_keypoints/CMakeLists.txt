cmake_minimum_required(VERSION 3.9)
project(yolov8_keypoints_mnn)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# MNN include directory (default: /usr/local/include)
include_directories(
    ./include
    /usr/local/include
    )

# libMNN.so directory (default: /usr/local/lib)
link_directories(
    /usr/local/lib
    )

add_executable(yolov8_keypoints_mnn src/main.cpp src/yolov8_keypoints_mnn.cpp)
target_link_libraries(yolov8_keypoints_mnn MNN ${OpenCV_LIBS})
