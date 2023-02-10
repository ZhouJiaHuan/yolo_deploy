#ifndef _YOLOX_MNN_HPP_
#define _YOLOX_MNN_HPP_


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


struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};


struct MNNDetConfigs
{
    std::string mnnPath;
    cv::Size inputSize;
    int classNum;
    int threads = 1;    
    float scoreThr = 0.25;
    float nmsThr = 0.5;
    std::array<float, 3> means = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> norms = {1.0f, 1.0f, 1.0f};
};


class YoloxMNN
{
public:
    YoloxMNN(const MNNDetConfigs& cfg);
    ~YoloxMNN();
    void detect(cv::Mat& img, std::vector<Object>& objects);
    void drawObject(cv::Mat& img, const std::vector<Object>& objects,
        const std::vector<std::string> labels);

private:
    cv::Mat staticResize(cv::Mat& img);
    void generateYoloxProposals(MNN::Tensor const* output,
        std::vector<Object>& objects);
    void nmSortedboxes(const std::vector<Object>& objects, std::vector<int>& picked);
    void decodeOutput(MNN::Tensor const* output, std::vector<Object>& objects,
        const cv::Size& imgSize);

private:
    MNNDetConfigs mnnCfg_;
    std::shared_ptr<MNN::Interpreter> interpreter_;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* inputTensor_ = nullptr;

    std::vector<int> strides_ = {8, 16, 32};


};  


#endif