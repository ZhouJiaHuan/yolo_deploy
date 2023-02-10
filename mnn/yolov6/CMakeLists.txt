cmake_minimum_required(VERSION 3.9)
project(yolov6_mnn)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(
    ./include
    /usr/local/include
    )

link_directories(
    /usr/local/lib
    )

add_executable(yolov6_mnn src/main.cpp src/yolov6_mnn.cpp)
target_link_libraries(yolov6_mnn MNN ${OpenCV_LIBS})

