cmake_minimum_required(VERSION 3.9)
project(yolox_mnn)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# MNN header files: /usr/local/include/MNN
include_directories(
    ./include
    /usr/local/include
    )

# MNN shared library: /usr/local/lib/libMNN.so
link_directories(
    /usr/local/lib
    )

add_executable(yolox_mmdet_mnn src/main.cpp src/yolox_mmdet_mnn.cpp)
target_link_libraries(yolox_mmdet_mnn MNN ${OpenCV_LIBS})

