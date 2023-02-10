cmake_minimum_required(VERSION 3.9)
project(yolox_mnn)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(
    ./include
    /usr/local/include
    )

link_directories(
    /usr/local/lib
    )

add_executable(yolox_mnn src/main.cpp src/yolox_mnn.cpp)
target_link_libraries(yolox_mnn MNN ${OpenCV_LIBS})

