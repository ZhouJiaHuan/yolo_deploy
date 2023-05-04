#ifndef _YOLOV8_KEYPOINTS_TRT_HPP_
#define _YOLOV8_KEYPOINTS_TRT_HPP_


#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"


struct Object
{
    cv::Rect2f rect;
    int label;
    float prob;
    std::vector<std::array<float, 3>> kpt;
};


struct TRTKeypointsConfigs
{
    std::string trtPath;
    cv::Size inputSize;
    int kptNum;
    float boxScore = 0.25;
    float kptScore = 0.5;
    float nmsThr = 0.5;
    std::array<float, 3> means = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> norms = {1.0f, 1.0f, 1.0f};
};


class Yolov8KeypointsTRT
{
public:
    static void nmSortedboxes(const std::vector<Object>& objects,
        std::vector<int>& picked, float nmsThr=0.1);
    
    static void drawObject(cv::Mat& img, const std::vector<Object>& objects);
    Yolov8KeypointsTRT(const TRTKeypointsConfigs& cfg);
    ~Yolov8KeypointsTRT();
    void detect(cv::Mat& img, std::vector<Object>& objects);

private:
    void initTRT();
    cv::Mat preProcess(cv::Mat& src);
    void postProcess(const float* output, std::vector<Object>& objects,
        const cv::Size& imgSize);
    void blobFromImage(float* blob, cv::Mat& img);

private:
    TRTKeypointsConfigs trtCfg_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    float* blob_;
    float* output_;
    size_t outputSize_;
    int inputIdx_, outputIdx_;
    void* buffers_[2];
    cudaStream_t stream_;

    const std::string INPUT_BLOB_NAME = "images";
    const std::string OUTPUT_BLOB_NAME = "output0";
};

#endif