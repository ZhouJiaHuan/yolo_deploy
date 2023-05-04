#include "yolov8_keypoints_trt.hpp"


static const cv::Scalar CV_COLOR_BLUE(255, 0, 0);
static const cv::Scalar CV_COLOR_GREEN(0, 255, 0);
static const cv::Scalar CV_COLOR_RED(0, 0, 255);
static const cv::Scalar CV_COLOR_BLACK(0, 0 ,0);
static const cv::Scalar CV_COLOR_WHITE(255, 255, 255);

static const std::vector<cv::Scalar> COLOR_LIST
{
    {128, 255, 0}, {255, 128, 50}, {128, 0, 255}, {255, 255, 0},
    {255, 102, 255}, {255, 51, 255}, {51, 153, 255}, {255, 153, 153},
    {255, 51, 51}, {153, 255, 153}, {51, 255, 51}, {0, 255, 0},
    {255, 0, 51}, {153, 0, 153}, {51, 0, 51}, {0, 0, 0},
    {0, 102, 255}, {0, 51, 255}, {0, 153, 255}, {0, 153, 153}
};

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

static Logger gLogger;


bool objectCompare(const Object& obj1, const Object& obj2)
{
    return obj1.prob > obj2.prob;
};


void Yolov8KeypointsTRT::nmSortedboxes(const std::vector<Object>& objects,
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


void Yolov8KeypointsTRT::drawObject(cv::Mat& img, const std::vector<Object>& objects)
{
    for (const auto& obj: objects)
    {
        cv::rectangle(img, obj.rect, CV_COLOR_GREEN, 2);
        int baseLine = 0;
        int score = obj.prob * 100;
        std::string label = std::to_string(score) + "%";
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
        
        for (int i=0; i<(int)obj.kpt.size(); ++i)
        {
            auto kpt = obj.kpt[i];
            auto color = COLOR_LIST[i%COLOR_LIST.size()];
            if (kpt[2] > 0)
            {
                int px = kpt[0];
                int py = kpt[1];
                cv::circle(img, cv::Point(kpt[0], kpt[1]), 2, color, -1);
            }
        }
    }
}


Yolov8KeypointsTRT::Yolov8KeypointsTRT(const TRTKeypointsConfigs& trtCfg): trtCfg_(trtCfg)
{    
    initTRT();
}


Yolov8KeypointsTRT::~Yolov8KeypointsTRT()
{
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
    delete output_;
    CHECK(cudaStreamDestroy(stream_));
    CHECK(cudaFree(buffers_[inputIdx_]));
    CHECK(cudaFree(buffers_[outputIdx_]));
}


void Yolov8KeypointsTRT::initTRT()
{    
    char* modelStream{nullptr};
    size_t size{0};
    std::ifstream file(trtCfg_.trtPath, std::ios::binary);
    assert(file.good());
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    modelStream = new char[size];
    assert(modelStream);
    file.read(modelStream, size);
    file.close();

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    assert(runtime_ != nullptr);
    engine_ = runtime_->deserializeCudaEngine(modelStream, size);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ !=nullptr);
    delete[] modelStream;

    auto outDims = engine_->getBindingDimensions(1);
    outputSize_ = 1;
    for (int i=0; i<outDims.nbDims; ++i)
    {
        outputSize_ *= outDims.d[i];
    }
    const int& inputW = trtCfg_.inputSize.width;
    const int& inputH = trtCfg_.inputSize.height;
    blob_ = new float[inputH*inputW*3];
    output_ = new float[outputSize_];
    inputIdx_ = engine_->getBindingIndex(INPUT_BLOB_NAME.c_str());
    outputIdx_ = engine_->getBindingIndex(OUTPUT_BLOB_NAME.c_str());
    assert(engine_->getBindingDataType(inputIdx_) == nvinfer1::DataType::kFLOAT);
    assert(engine_->getBindingDataType(outputIdx_) == nvinfer1::DataType::kFLOAT);

    CHECK(cudaMalloc(&buffers_[inputIdx_], 3*inputH*inputW*sizeof(float)));
    CHECK(cudaMalloc(&buffers_[outputIdx_], outputSize_*sizeof(float)));
    CHECK(cudaStreamCreate(&stream_));
}


cv::Mat Yolov8KeypointsTRT::preProcess(cv::Mat& src)
{
    // 1. resize with the minimal ratio of W and H
    const int& inputW = trtCfg_.inputSize.width;
    const int& inputH = trtCfg_.inputSize.height;
    float r = std::min(inputW/(src.cols*1.0), inputH/(src.rows*1.0));
    int unpadW = r * src.cols;
    int unpadH = r * src.rows;
    cv::Mat re(unpadH, unpadW, CV_8UC3);
    cv::resize(src, re, re.size());
    // 2. padding the bottom-right with pad value
    cv::Mat dst(inputH, inputW, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(dst(cv::Rect(0, 0, re.cols, re.rows)));
    // 3. BGR -> RGB
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    return dst;
}


void Yolov8KeypointsTRT::postProcess(const float* output, std::vector<Object>& objects,
    const cv::Size& imgSize)
{
    std::vector<Object> proposals;
    int numProposal = outputSize_ / (3*trtCfg_.kptNum + 5);
    std::cout << outputSize_ << std::endl;
    for (int idx=0; idx<numProposal; ++idx)
    {
        float cx = output[idx+0*numProposal];
        float cy = output[idx+1*numProposal];
        float w = output[idx+2*numProposal];
        float h = output[idx+3*numProposal];
        float prob = output[idx+4*numProposal];
        if (prob > trtCfg_.boxScore)
        {
            Object obj;
            obj.label = 0;
            obj.prob = prob;
            obj.rect = cv::Rect(cx-0.5*w, cy-0.5*h, w, h);
            for (int i=0; i<trtCfg_.kptNum; ++i)
            {
                float px = output[idx+(5+3*i+0)*numProposal];
                float py = output[idx+(5+3*i+1)*numProposal];
                float score = output[idx+(5+3*i+2)*numProposal];
                obj.kpt.push_back({px, py, score});
            }
            proposals.push_back(obj);
        }
    }
    std::sort(proposals.begin(), proposals.end(), &objectCompare);

    std::vector<int> picked;
    nmSortedboxes(proposals, picked, trtCfg_.nmsThr);
    
    int count = picked.size();
    objects.resize(count);

    const int& imgW = imgSize.width;
    const int& imgH = imgSize.height;
    const int& inputW = trtCfg_.inputSize.width;
    const int& inputH = trtCfg_.inputSize.height;
    float scale = std::min(inputW / (imgW*1.0), inputH / (imgH*1.0));
    for (int i=0; i<count; ++i)
    {
        // bbox
        objects[i] = proposals[picked[i]];
        auto& rect = objects[i].rect;
        float x1 = (rect.x) / scale;
        float y1 = (rect.y) / scale;
        float x2 = (rect.x + rect.width) / scale;
        float y2 = (rect.y + rect.height) / scale;
        x1 = std::max(std::min(x1, (float)imgW-1), 0.0f);
        y1 = std::max(std::min(y1, (float)imgH-1), 0.0f);
        x2 = std::max(std::min(x2, (float)imgW-1), 0.0f);
        y2 = std::max(std::min(y2, (float)imgH-1), 0.0f);
        rect.x = x1;
        rect.y = y1;
        rect.width = x2 - x1;
        rect.height = y2 - y1;

        // keypoints
        for (auto& kpt: objects[i].kpt)
        {
            if (kpt[2] > trtCfg_.kptScore)
            {
                kpt[0] /= scale;
                kpt[1] /= scale;
            }
            else
            {
                kpt[0] = -1;
                kpt[1] = -1;
            }
        }
    }
}


void Yolov8KeypointsTRT::blobFromImage(float* blob, cv::Mat& img)
{
    int H = img.rows;
    int W = img.cols;
    for (size_t c = 0; c < 3; c++)
    {
        const float& mean = trtCfg_.means[c];
        const float& norm = trtCfg_.norms[c];
        for (size_t h = 0; h < H; h++)
        {
            for (size_t w = 0; w < W; w++)
            {
                blob_[c * W * H + h * W + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]-mean) / norm);
            }
        }
    }
}


void Yolov8KeypointsTRT::detect(cv::Mat& img, std::vector<Object>& objects)
{
    const int& inputW = trtCfg_.inputSize.width;
    const int& inputH = trtCfg_.inputSize.height;

    // resize -> normalize -> blob
    cv::Mat prImg = preProcess(img);
    blobFromImage(blob_, prImg);

    // memory copy and inference
    CHECK(cudaMemcpyAsync(buffers_[inputIdx_], blob_, 3*inputH*inputW*sizeof(float),
                          cudaMemcpyHostToDevice, stream_));
    context_->enqueueV2(buffers_, stream_, nullptr);
    CHECK(cudaMemcpyAsync(output_, buffers_[outputIdx_], outputSize_*sizeof(float),
                          cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    // parse output -> nms -> rescale to src
    postProcess(output_, objects, img.size());
}
