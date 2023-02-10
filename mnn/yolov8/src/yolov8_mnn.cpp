#include "yolov8_mnn.hpp"


static const cv::Scalar CV_COLOR_BLUE(255, 0, 0);
static const cv::Scalar CV_COLOR_GREEN(0, 255, 0);
static const cv::Scalar CV_COLOR_RED(0, 0, 255);
static const cv::Scalar CV_COLOR_BLACK(0, 0 ,0);
static const cv::Scalar CV_COLOR_WHITE(255, 255, 255);


bool objectCompare(const Object& obj1, const Object& obj2)
{
    return obj1.prob > obj2.prob;
};


void Yolov8MNN::nmSortedboxes(const std::vector<Object>& objects,
    std::vector<int>& picked, float nmsThr)
{
    picked.clear();
    const int n = objects.size();
    std::vector<float> areas(n);
    for (int i=0; i<n; ++i)
    {
        areas[i] = objects[i].rect.area();
    }

    // NMS with all bboxes (across classes)
    for (int i=0; i < n; ++i)
    {
        const Object& a = objects[i];
        
        int keep = 1;
        for (int j=0; j<(int)picked.size(); ++j)
        {
            const Object& b = objects[picked[j]];
            float iArea = (a.rect & b.rect).area();
            float uArea = areas[i] + areas[picked[j]] - iArea;
            if (iArea / uArea > nmsThr)
            {
                keep = 0;
                break;
            }
        }
        if (keep)
        {
            picked.push_back(i);
        }
    }
}


void Yolov8MNN::drawObject(cv::Mat& img, const std::vector<Object>& objects,
    const std::vector<std::string>& labels)
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


Yolov8MNN::Yolov8MNN(const MNNDetConfigs& mnnCfg): mnnCfg_(mnnCfg)
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
        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, means, 3, norms, 3));

    MNN::ScheduleConfig config;
    config.numThread = mnnCfg_.threads;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
    session_ = interpreter_->createSession(config);
    inputTensor_ = interpreter_->getSessionInput(session_, nullptr);
}


Yolov8MNN::~Yolov8MNN()
{
    interpreter_->releaseModel();
    interpreter_->releaseSession(session_);
}


cv::Mat Yolov8MNN::preProcess(cv::Mat& src)
{
    // 1. resize with the minimal ratio of W and H
    const int& inputW = mnnCfg_.inputSize.width;
    const int& inputH = mnnCfg_.inputSize.height;
    float r = std::min(inputW/(src.cols*1.0), inputH/(src.rows*1.0));
    int unpadW = r * src.cols;
    int unpadH = r * src.rows;
    cv::Mat re(unpadH, unpadW, CV_8UC3);
    cv::resize(src, re, re.size());
    // 2. padding the bottom-right with pad value
    cv::Mat dst(inputH, inputW, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(dst(cv::Rect(0, 0, re.cols, re.rows)));
    return dst;
}


void Yolov8MNN::postProcess(MNN::Tensor const* output, std::vector<Object>& objects,
    const cv::Size& imgSize)
{
    std::vector<Object> proposals;
    int numProposal = output->shape()[2];
    int classNum = output->shape()[1] - 4;  // [xc, yc, w, h, c1, c2,]
    auto basicPos = output->host<float>();
    for (int idx=0; idx<numProposal; ++idx)
    {
        const float* pos = basicPos + idx;
        float cx = pos[0*numProposal];
        float cy = pos[1*numProposal];
        float w = pos[2*numProposal];
        float h = pos[3*numProposal];
        int clsId = 0;
        for (int id=0; id < classNum; ++id)
        {
            if (pos[(4+id)*numProposal] > pos[(4+clsId)*numProposal])
            {
                clsId = id;
            }
        }

        float prob = pos[(4+clsId)*numProposal];
        if (prob > mnnCfg_.scoreThr)
        {
            Object obj;
            obj.label = clsId;
            obj.prob = prob;
            obj.rect = cv::Rect(cx-0.5*w, cy-0.5*h, w, h);
            
            proposals.push_back(obj);
        }
    }
    std::sort(proposals.begin(), proposals.end(), &objectCompare);

    std::vector<int> picked;
    nmSortedboxes(proposals, picked, mnnCfg_.nmsThr);
    
    int count = picked.size();
    objects.resize(count);

    const int& imgW = imgSize.width;
    const int& imgH = imgSize.height;
    const int& inputW = mnnCfg_.inputSize.width;
    const int& inputH = mnnCfg_.inputSize.height;
    float scale = std::min(inputW / (imgW*1.0), inputH / (imgH*1.0));
    for (int i=0; i<count; ++i)
    {
        // scale
        objects[i] = proposals[picked[i]];
        auto& rect = objects[i].rect;
        float x1 = (rect.x) / scale;
        float y1 = (rect.y) / scale;
        float x2 = (rect.x + rect.width) / scale;
        float y2 = (rect.y + rect.height) / scale;
        // clip
        x1 = std::max(std::min(x1, (float)imgW-1), 0.0f);
        y1 = std::max(std::min(y1, (float)imgH-1), 0.0f);
        x2 = std::max(std::min(x2, (float)imgW-1), 0.0f);
        y2 = std::max(std::min(y2, (float)imgH-1), 0.0f);

        rect.x = x1;
        rect.y = y1;
        rect.width = x2 - x1;
        rect.height = y2 - y1;
    }
}


void Yolov8MNN::detect(cv::Mat& img, std::vector<Object>& objects)
{
    const int& inputW = mnnCfg_.inputSize.width;
    const int& inputH = mnnCfg_.inputSize.height;

    cv::Mat prImg = preProcess(img);
    interpreter_->resizeTensor(inputTensor_, {1, 3, inputH, inputW});
    interpreter_->resizeSession(session_);

    pretreat_->convert(prImg.data, inputW, inputH, prImg.step[0], inputTensor_);
    interpreter_->runSession(session_);

    MNN::Tensor* outputTensor = interpreter_->getSessionOutput(session_, NULL);
    MNN::Tensor outputTensorHost(outputTensor, outputTensor->getDimensionType());
    outputTensor->copyToHostTensor(&outputTensorHost);

    auto base = outputTensorHost.host<float>();
    postProcess(&outputTensorHost, objects, img.size());
}
