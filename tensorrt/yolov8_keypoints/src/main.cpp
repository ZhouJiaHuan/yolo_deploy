#include "yolov8_keypoints_trt.hpp"

#define CUDA_DEVICE 0

void image_demo(int argc, char** argv)
{
    const std::string imgPath = argv[3];
    const int size = atoi(argv[4]);
    bool show = argc == 7? atoi(argv[6]): 0;

    TRTKeypointsConfigs configs;
    configs.trtPath = argv[2];
    configs.inputSize = cv::Size(size, size);
    configs.kptNum = atoi(argv[5]);
    configs.boxScore = 0.25;
    configs.kptScore = 0.5;
    configs.nmsThr = 0.5;
    configs.means = {0.0f, 0.0f, 0.0f};
    configs.norms = {255.0f, 255.0f, 255.0f};

    Yolov8KeypointsTRT yolo(configs);

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
        std::cout << obj.prob << "\t";
        std::cout << "(" << obj.rect.tl() << ", " << obj.rect.br() << ")" << std::endl; 
    }
    if (show)
    {
        yolo.drawObject(image, objects);
        cv::imshow("result", image);
        cv::waitKey(0);
    }
}


void stream_demo(int argc, char** argv)
{
    const int camId = atoi(argv[3]);
    const int size = atoi(argv[4]);
    bool show = argc == 7? atoi(argv[6]): 0;

    TRTKeypointsConfigs configs;
    configs.trtPath = argv[2];
    configs.inputSize = cv::Size(size, size);
    configs.kptNum = atoi(argv[5]);
    configs.boxScore = 0.25;
    configs.kptScore = 0.5;
    configs.nmsThr = 0.5;
    configs.means = {0.0f, 0.0f, 0.0f};
    configs.norms = {255.0f, 255.0f, 255.0f};

    Yolov8KeypointsTRT yolo(configs);

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
            yolo.drawObject(image, objects);
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
        fprintf(stderr, "usage: %s [demo] [model_path] [input] [input_size] [kpt_num] [show] ..\n", argv[0]);
        return -1;
    }
    cudaSetDevice(CUDA_DEVICE);
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