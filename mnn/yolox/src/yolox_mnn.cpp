#include "yolox_mnn.hpp"


static const cv::Scalar CV_COLOR_BLUE(255, 0, 0);
static const cv::Scalar CV_COLOR_GREEN(0, 255, 0);
static const cv::Scalar CV_COLOR_RED(0, 0, 255);
static const cv::Scalar CV_COLOR_BLACK(0, 0 ,0);
static const cv::Scalar CV_COLOR_WHITE(255, 255, 255);


bool objectCompare(const Object& obj1, const Object& obj2)
{
    return obj1.prob > obj2.prob;
};


YoloxMNN::YoloxMNN(const MNNDetConfigs& mnnCfg): mnnCfg_(mnnCfg)
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


YoloxMNN::~YoloxMNN()
{
    interpreter_->releaseModel();
    interpreter_->releaseSession(session_);
}


cv::Mat YoloxMNN::staticResize(cv::Mat& img)
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


void YoloxMNN::generateYoloxProposals(MNN::Tensor const* output,
    std::vector<Object>& objects)
{
    const int& inputW = mnnCfg_.inputSize.width;
    const int& inputH = mnnCfg_.inputSize.height;
    const int& classNum = mnnCfg_.classNum;
    const float& scoreThr = mnnCfg_.scoreThr;
    
    // 1. generate grid with different strides
    std::vector<GridAndStride> gridStrides;
    for (auto s: strides_)
    {
        int gridW = inputW / s;
        int gridH = inputH / s;
        for (int g1=0; g1<gridH; ++g1)
        {
            for (int g0=0; g0<gridW; ++g0)
            {
                gridStrides.push_back((GridAndStride){g0, g1, s});
            }        
        }
    }

    // 2. parse object info (x, y, w, h, label, score) of all anchors
    const int numAnchors = gridStrides.size();
    for (int anchorIdx=0; anchorIdx<numAnchors; ++anchorIdx)
    {
        // 2.1 parse current grid and stride (anchor point)
        const int grid0 = gridStrides[anchorIdx].grid0;
        const int grid1 = gridStrides[anchorIdx].grid1;
        const int stride = gridStrides[anchorIdx].stride;
        // 2.2 parse current box info: [x_c, y_c, w, h, obj_prob, cls1_score, ...clsN_score]
        // object pos should be mapped to input image with grid and stride
        const int basicPos = anchorIdx * (classNum + 5);
        const float* pos = output->host<float>() + basicPos;
        float xCenter = (pos[0] + grid0) * stride;
        float yCenter = (pos[1] + grid1) * stride;
        float w = exp(pos[2]) * stride;
        float h = exp(pos[3]) * stride;
        float x0 = xCenter - w * 0.5f;
        float y0 = yCenter - h * 0.5f;        
        float boxObject = pos[4];
        // 2.3 compute final cls_score: obj_prob * cls_score
        // remove bad bbox with score threshold
        for (int classIdx=0; classIdx<classNum; ++classIdx)
        {
            float clsScore = pos[5+classIdx];
            float boxProb = boxObject * clsScore;
            if (boxProb > scoreThr)
            {
                Object obj;
                obj.rect = cv::Rect2f(x0, y0, w, h);
                obj.label = classIdx;
                obj.prob = boxProb;
                objects.push_back(obj);
            }
        }
    }
    // 3. sort the bbox with cls_score
    std::sort(objects.begin(), objects.end(), &objectCompare);
}


void YoloxMNN::nmSortedboxes(const std::vector<Object>& objects,
    std::vector<int>& picked)
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
            if (iArea / uArea > mnnCfg_.nmsThr)
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


void YoloxMNN::decodeOutput(MNN::Tensor const* output, std::vector<Object>& objects,
    const cv::Size& imgSize)
{
    std::vector<Object> proposals;
    generateYoloxProposals(output, proposals);
    
    std::vector<int> picked;
    nmSortedboxes(proposals, picked);
    
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


void YoloxMNN::detect(cv::Mat& img, std::vector<Object>& objects)
{
    const int& inputW = mnnCfg_.inputSize.width;
    const int& inputH = mnnCfg_.inputSize.height;
    const int& classNum = mnnCfg_.classNum;

    cv::Mat prImg = staticResize(img);
    interpreter_->resizeTensor(inputTensor_, {1, 3, inputH, inputW});
    interpreter_->resizeSession(session_);

    pretreat_->convert(prImg.data, inputW, inputH, prImg.step[0], inputTensor_);
    interpreter_->runSession(session_);
    MNN::Tensor* outputTensor = interpreter_->getSessionOutput(session_, NULL);
    auto outputShape = outputTensor->shape();

    MNN::Tensor outputTensorHost(outputTensor, outputTensor->getDimensionType());
    outputTensor->copyToHostTensor(&outputTensorHost);
    decodeOutput(&outputTensorHost, objects, img.size());
}


void YoloxMNN::drawObject(cv::Mat& img, const std::vector<Object>& objects,
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