#include "yolov8_mnn.hpp"

static const std::vector<std::string> COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush" 
};


void image_demo(int argc, char** argv)
{
    const std::string imgPath = argv[4];
    const int size = atoi(argv[5]);
    bool show = argc == 7? atoi(argv[6]): 0;

    std::vector<std::string> labels;
    std::string dataset = argv[3];
    if (dataset == "coco")
    {
        labels = COCO_NAMES;
    }
    else
    {
        std::cout << "unrecognized dataset: "  << argv[2] << std::endl;
        exit(EXIT_FAILURE);
    }

    MNNDetConfigs configs;
    configs.mnnPath = argv[2];
    configs.inputSize = cv::Size(size, size);
    configs.classNum = labels.size();
    configs.threads = 2;
    configs.scoreThr = 0.25;
    configs.nmsThr = 0.5;
    configs.means = {0.0f, 0.0f, 0.0f};
    configs.norms = {1.0f, 1.0f, 1.0f};

    Yolov8MNN yolo(configs);

    cv::Mat image = cv::imread(imgPath);
    std::vector<Object> objects;
    auto start = std::chrono::steady_clock::now();
    yolo.detect(image, objects);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    float timeMs = elapsed.count()*1000;
    std::cout << "detect " << (int)objects.size() << " objects in " <<  timeMs << " ms\n";
    for (const auto& obj: objects)
    {
        std::cout << labels[obj.label] << ": " << obj.prob << "\t";
        std::cout << "(" << obj.rect.tl() << ", " << obj.rect.br() << ")" << std::endl; 
    }
    if (show)
    {
        yolo.drawObject(image, objects, labels);
        cv::imshow("result", image);
        cv::waitKey(0);
    }
}


void stream_demo(int argc, char** argv)
{
    const int camId = atoi(argv[4]);
    const int size = atoi(argv[5]);
    bool show = argc == 7? atoi(argv[6]): 0;

    std::vector<std::string> labels;
    std::string dataset = argv[3];
    if (dataset == "coco")
    {
        labels = COCO_NAMES;
    }
    else
    {
        std::cout << "unrecognized dataset: "  << argv[2] << std::endl;
        exit(EXIT_FAILURE);
    }

    MNNDetConfigs configs;
    configs.mnnPath = argv[2];
    configs.inputSize = cv::Size(size, size);
    configs.classNum = labels.size();
    configs.threads = 2;
    configs.scoreThr = 0.25;
    configs.nmsThr = 0.5;
    configs.means = {0.0f, 0.0f, 0.0f};
    configs.norms = {1.0f, 1.0f, 1.0f};

    Yolov8MNN yolo(configs);

    cv::Mat image;
    cv::VideoCapture cap(camId);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::vector<Object> objects;
    while (true)
    {
        cap >> image;
        objects.clear();
        auto start = std::chrono::steady_clock::now();
        yolo.detect(image, objects);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        float timeMs = elapsed.count()*1000;
        std::cout << "detect " << (int)objects.size() << " objects in " <<  timeMs << " ms\n";

        if (show)
        {
            yolo.drawObject(image, objects, labels);
            cv::imshow("result", image);
            if (cv::waitKey(1) == 'q')
            {
                break;
            }
        }
    }
}


int main(int argc, char** argv)
{
    if (argc != 6 && argc != 7)
    {
        fprintf(stderr, "usage: %s [demo] [model_path] [dataset] [input] [input_size] [show] ..\n", argv[0]);
        return -1;
    }
    
    std::string demo = argv[1];
    if (demo == "image")
    {
        image_demo(argc, argv);
    }
    else if (demo == "stream")
    {
        stream_demo(argc, argv);
    }
    else
    {
        std::cout << "unrecognized demo type: "  << argv[1] << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}
