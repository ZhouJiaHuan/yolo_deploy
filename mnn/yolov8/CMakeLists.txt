cmake_minimum_required(VERSION 3.9)
project(yolov8_mnn)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(
    ./include
    /usr/local/include
    )

link_directories(
    /usr/local/lib
    )

add_executable(yolov8_mnn src/main.cpp src/yolov8_mnn.cpp)
target_link_libraries(yolov8_mnn MNN ${OpenCV_LIBS})
