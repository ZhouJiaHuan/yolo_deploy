#ifndef _YOLOX_MMDET_MNN_HPP_
#define _YOLOX_MMDET_MNN_HPP_


#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <map>
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

/* Deploy of YOLOx trained with mmdetection 3.0
 * The ONNX model was converted with mmdeploy 1.1.0
 * and the MNN model was converted with MNNConvert (MNN >= 2.0.0).
 * NMS process was included in ONNX model (end to end).
 * 
 * Note:
 * Only the static model was supported during the conversion
 * from ONNX to MNN. The dynamic model may cause some problems
 */
class YoloxMMdetMNN
{
public:
    YoloxMMdetMNN(const MNNDetConfigs& cfg);
    ~YoloxMMdetMNN();
    void detect(cv::Mat& img, std::vector<Object>& objects);
    void drawObject(cv::Mat& img, const std::vector<Object>& objects,
        const std::vector<std::string> labels);

private:
    cv::Mat staticResize(cv::Mat& img);
    void decodeOutput(std::map<std::string, MNN::Tensor*> output,
        std::vector<Object>& objects, const cv::Size& imgSize);

private:
    MNNDetConfigs mnnCfg_;
    std::shared_ptr<MNN::Interpreter> interpreter_;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* inputTensor_ = nullptr;

    const std::string outputName1_ = "dets";
    const std::string outputName2_ = "labels";
};  


#endif