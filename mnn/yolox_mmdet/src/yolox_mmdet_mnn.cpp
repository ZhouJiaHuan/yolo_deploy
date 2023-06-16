#include "yolox_mmdet_mnn.hpp"


static const cv::Scalar CV_COLOR_BLUE(255, 0, 0);
static const cv::Scalar CV_COLOR_GREEN(0, 255, 0);
static const cv::Scalar CV_COLOR_RED(0, 0, 255);
static const cv::Scalar CV_COLOR_BLACK(0, 0 ,0);
static const cv::Scalar CV_COLOR_WHITE(255, 255, 255);


bool objectCompare(const Object& obj1, const Object& obj2)
{
    return obj1.prob > obj2.prob;
};


YoloxMMdetMNN::YoloxMMdetMNN(const MNNDetConfigs& mnnCfg): mnnCfg_(mnnCfg)
{
    const auto& mnnPath = mnnCfg_.mnnPath;
    if (mnnPath.rfind(".mnn") != mnnPath.size()-4)
    {
        std::cout << "invalid model format! expected '.mnn' model" << std::endl;
        exit(EXIT_FAILURE);
    }

    interpreter_ = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(mnnPath.c_str()));

    const float means[3] = {mnnCfg_.means[0], mnnCfg_.means[1], mnnCfg_.means[2]};
    const float norms[3] = {mnnCfg_.norms[0]/255.0f, mnnCfg_.norms[1]/255.0f, mnnCfg_.norms[2]/255.0f};
    pretreat_ = std::shared_ptr<MNN::CV::ImageProcess>(
        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, means, 3, norms, 3));

    MNN::ScheduleConfig config;
    config.numThread = mnnCfg.threads;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
    session_ = interpreter_->createSession(config);
    inputTensor_ = interpreter_->getSessionInput(session_, nullptr);
}


YoloxMMdetMNN::~YoloxMMdetMNN()
{
    interpreter_->releaseModel();
    interpreter_->releaseSession(session_);
}


cv::Mat YoloxMMdetMNN::staticResize(cv::Mat& img)
{
    // 1. resize with the minimal ratio of W and H
    const int& inputW = mnnCfg_.inputSize.width;
    const int& inputH = mnnCfg_.inputSize.height;
    float r = std::min(inputW/(img.cols*1.0), inputH/(img.rows*1.0));
    int unpadW = r * img.cols;
    int unpadH = r * img.rows;
    cv::Mat re(unpadH, unpadW, CV_8UC3);
    cv::resize(img, re, re.size());
    // 2. padding the bottom-right with value 114
    cv::Mat out(inputH, inputW, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}


void YoloxMMdetMNN::decodeOutput(std::map<std::string, MNN::Tensor*> output,
    std::vector<Object>& objects, const cv::Size& imgSize)
{
    const int& imgW = imgSize.width;
    const int& imgH = imgSize.height;
    const int& inputW = mnnCfg_.inputSize.width;
    const int& inputH = mnnCfg_.inputSize.height;
    float scale = std::min(inputW / (imgW*1.0), inputH / (imgH*1.0));
    auto dets = output[outputName1_];
    auto labels = output[outputName2_];
    int proposalNum = dets->shape()[1];
    for (int i=0; i< proposalNum; ++ i)
    {
        auto pos = dets->host<float>() + 5 * i;
        float x1 = pos[0];
        float y1 = pos[1];
        float x2 = pos[2];
        float y2 = pos[3];
        float scores = pos[4];
        int labelIdx = labels->host<int>()[i];

        if (scores >= mnnCfg_.scoreThr)
        {
            // rescale and clip
            x1 = std::max(std::min(x1/scale, (float)imgW-1), 0.0f);
            y1 = std::max(std::min(y1/scale, (float)imgH-1), 0.0f);
            x2 = std::max(std::min(x2/scale, (float)imgW-1), 0.0f);
            y2 = std::max(std::min(y2/scale, (float)imgH-1), 0.0f);
            Object obj;
            obj.rect = cv::Rect2f(x1, y1, x2-x1, y2-y1);
            obj.prob = scores;
            obj.label = labelIdx;
            objects.push_back(obj);
        } 
    }
}


void YoloxMMdetMNN::detect(cv::Mat& img, std::vector<Object>& objects)
{
    const int& inputW = mnnCfg_.inputSize.width;
    const int& inputH = mnnCfg_.inputSize.height;
    const int& classNum = mnnCfg_.classNum;

    cv::Mat prImg = staticResize(img);
    interpreter_->resizeTensor(inputTensor_, {1, 3, inputH, inputW});
    interpreter_->resizeSession(session_);

    pretreat_->convert(prImg.data, inputW, inputH, prImg.step[0], inputTensor_);
    interpreter_->runSession(session_);
    auto output = interpreter_->getSessionOutputAll(session_);
    decodeOutput(output, objects, img.size());
}


void YoloxMMdetMNN::drawObject(cv::Mat& img, const std::vector<Object>& objects,
        const std::vector<std::string> labels)
{
    for (const auto& obj: objects)
    {
        cv::rectangle(img, obj.rect, CV_COLOR_GREEN, 2);
        int baseLine = 0;
        int score = round(obj.prob * 100);
        std::string label = labels[obj.label];
        label += ": " + std::to_string(score) + "%";
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
            0.6, 1, &baseLine);
        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        if (y > img.rows) 
        {
            y = img.rows;
        }
        cv::Rect backRect(cv::Point(x, y),
            cv::Size(labelSize.width, labelSize.height + baseLine));
        cv::rectangle(img, backRect, CV_COLOR_GREEN, -1);
        cv::putText(img, label.c_str(), cv::Point(x, y + labelSize.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_COLOR_RED, 2);
    }
}