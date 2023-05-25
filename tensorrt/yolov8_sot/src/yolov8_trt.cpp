#include "yolov8_trt.hpp"


static const cv::Scalar CV_COLOR_BLUE(255, 0, 0);
static const cv::Scalar CV_COLOR_GREEN(0, 255, 0);
static const cv::Scalar CV_COLOR_RED(0, 0, 255);
static const cv::Scalar CV_COLOR_BLACK(0, 0 ,0);
static const cv::Scalar CV_COLOR_WHITE(255, 255, 255);

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

#define DEVICE 0  // GPU id

static Logger gLogger;


bool objectCompare(const Object& obj1, const Object& obj2)
{
    return obj1.prob > obj2.prob;
};


void Yolov8TRT::nmSortedboxes(const std::vector<Object>& objects,
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


void Yolov8TRT::drawObject(cv::Mat& img, const std::vector<Object>& objects,
    const std::vector<std::string>& labels)
{
    for (const auto& obj: objects)
    {
        cv::rectangle(img, obj.rect, CV_COLOR_GREEN, 1);
        int baseLine = 0;
        int score = obj.prob * 100;
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
            cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_COLOR_RED, 1);
    }
}


Yolov8TRT::Yolov8TRT(const TRTDetConfigs& trtCfg): trtCfg_(trtCfg)
{    
    initTRT();
}


Yolov8TRT::~Yolov8TRT()
{
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
    delete output_;
    CHECK(cudaStreamDestroy(stream_));
    CHECK(cudaFree(buffers_[inputIdx_]));
    CHECK(cudaFree(buffers_[outputIdx_]));
}


void Yolov8TRT::initTRT()
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


cv::Mat Yolov8TRT::preProcess(cv::Mat& src)
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


void Yolov8TRT::postProcess(const float* output, std::vector<Object>& objects,
    const cv::Size& imgSize)
{
    int classNum = trtCfg_.classNum;
    std::vector<Object> proposals;
    int numProposal = outputSize_ / (classNum + 4);
    for (int idx=0; idx<numProposal; ++idx)
    {
        float cx = output[idx+0*numProposal];
        float cy = output[idx+1*numProposal];
        float w = output[idx+2*numProposal];
        float h = output[idx+3*numProposal];
        int clsId = 0;
        for (int id=0; id < classNum; ++id)
        {
            if (output[idx+(4+id)*numProposal] > output[idx+(4+clsId)*numProposal])
            {
                clsId = id;
            }
        }

        float prob = output[idx+(4+clsId)*numProposal];
        if (prob > trtCfg_.scoreThr)
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


void Yolov8TRT::blobFromImage(float* blob, cv::Mat& img)
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


void Yolov8TRT::detect(cv::Mat& img, std::vector<Object>& objects)
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
