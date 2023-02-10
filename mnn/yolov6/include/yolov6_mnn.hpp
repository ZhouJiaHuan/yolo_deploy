#ifndef _YOLOV6_MNN_HPP_
#define _YOLOV6_MNN_HPP_


#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>


struct Object
{
    cv::Rect2f rect;
    int label;
    float prob;
};


struct MNNDetConfigs
{
    std::string mnnPath;
    cv::Size inputSize;
    int classNum;
    int threads = 2;    
    float scoreThr = 0.25;
    float nmsThr = 0.5;
    std::array<float, 3> means = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> norms = {1.0f, 1.0f, 1.0f};
};


class Yolov6MNN
{
public:
    static void nmSortedboxes(const std::vector<Object>& objects,
        std::vector<int>& picked, float nmsThr=0.1);
    
    static void drawObject(cv::Mat& img, const std::vector<Object>& objects,
        const std::vector<std::string>& labels);
    Yolov6MNN(const MNNDetConfigs& cfg);
    ~Yolov6MNN();
    void detect(cv::Mat& img, std::vector<Object>& objects);

private:
    cv::Mat preProcess(cv::Mat& src);
    void postProcess(MNN::Tensor const* output, std::vector<Object>& objects,
        const cv::Size& imgSize);

private:
    MNNDetConfigs mnnCfg_;
    std::shared_ptr<MNN::Interpreter> interpreter_;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* inputTensor_ = nullptr;
};  


#endif