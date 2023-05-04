#include "yolov8_keypoints_mnn.hpp"


void image_demo(int argc, char** argv)
{
    const std::string imgPath = argv[3];
    const int size = atoi(argv[4]);
    bool show = argc == 6? atoi(argv[5]): 0;

    MNNKeypointsConfigs configs;
    configs.mnnPath = argv[2];
    configs.inputSize = cv::Size(size, size);
    configs.threads = 2;
    configs.boxScore = 0.25;
    configs.kptScore = 0.5;
    configs.nmsThr = 0.5;
    configs.means = {0.0f, 0.0f, 0.0f};
    configs.norms = {1.0f, 1.0f, 1.0f};

    Yolov8KeypointsMNN yolo(configs);

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
    bool show = argc == 6? atoi(argv[5]): 0;

    std::vector<std::string> labels;
    std::string dataset = argv[3];

    MNNKeypointsConfigs configs;
    configs.mnnPath = argv[2];
    configs.inputSize = cv::Size(size, size);
    configs.threads = 2;
    configs.boxScore = 0.25;
    configs.kptScore = 0.5;
    configs.nmsThr = 0.5;
    configs.means = {0.0f, 0.0f, 0.0f};
    configs.norms = {1.0f, 1.0f, 1.0f};

    Yolov8KeypointsMNN yolo(configs);

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
    if (argc != 5 && argc != 6)
    {
        fprintf(stderr, "usage: %s [demo] [model_path] [input] [input_size] [show] ..\n", argv[0]);
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
