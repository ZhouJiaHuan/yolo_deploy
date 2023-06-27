#include <ctime>
#include "yolov8_mnn.hpp"
#include "tracker.hpp"

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

#define CV_FONT cv::FONT_HERSHEY_COMPLEX_SMALL

std::vector<Object> objects;
Tracker tracker(0.033, 300, 0.5, Tracker::KF_BBOX_XYWH);
cv::Mat image;

static void onMouse(int event, int x, int y, int, void*)
{
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
        std::cout << "select target with position: " << x << ' ' << y << std::endl;
        tracker.selectTarget(cv::Point(x, y), objects);
        break;
    case cv::EVENT_LBUTTONUP:
        if (tracker.withTarget())
        {
            std::cout << "target class id: " << tracker.getLabel() << std::endl;
            tracker.draw(image);
        }
        else
        {
            std::cout << "target select failed!" << std::endl;
        }
        break;
    default:
        break;
    }
}

cv::VideoCapture initCap(std::string camId)
{
    cv::VideoCapture cap;
    if (camId.size() == 1)
    {
        cap = cv::VideoCapture(atoi(camId.c_str()));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
    }
    else
    {
        cap = cv::VideoCapture(camId);
    }
    return cap;
}


void track_demo(int argc, char** argv)
{
    const int size = atoi(argv[5]);
    std::vector<std::string> labels;
    std::string dataset = argv[3];
    bool save = argc == 7? atoi(argv[6]): 0;
    if (dataset == "coco")
    {
        labels = COCO_NAMES;
    }
    else
    {
        std::cerr << "unrecognized dataset: "  << argv[2] << std::endl;
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

    cv::VideoCapture cap = initCap(argv[4]);
    if (!cap.isOpened())
    {
        std::cerr << "open camera / video failed!" << std::endl;
        return;
    }

    cv::VideoWriter writer;
    if (save)
    {
        cap >> image;  // clear cache
        std::time_t stamp = std::time(0);
        std::string saveName = "result_" + std::to_string(stamp) + ".mp4";
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size saveSize = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                                 (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open(saveName, cv::VideoWriter::fourcc('m','p','4','v'), fps, saveSize);
        std::cout << "save video to " << saveName << std::endl;
    }

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("result", onMouse);

    while (true)
    {
        cap >> image;
        if (!image.data)
        {
            break;
        }
        objects.clear();
        auto start = std::chrono::steady_clock::now();
        yolo.detect(image, objects);
        if (tracker.withTarget())
        {
            tracker.step(objects);
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        float timeMs = elapsed.count()*1000;
        // std::cout << "detect " << (int)objects.size() << " objects in " <<  timeMs << " ms\n";
        yolo.drawObject(image, objects, labels);
        tracker.draw(image);
        std::string speedStr = "speed: " + std::to_string(int(1000/timeMs)) + " FPS"; 
        cv::putText(image, speedStr, cv::Point(5, 20), CV_FONT, 1, cv::Scalar(0, 0, 255));
        if (save)
        {
            writer.write(image);
        }
        cv::imshow("result", image);
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    cap.release();
    writer.release();
}


int main(int argc, char** argv)
{
    if (argc != 6 && argc != 7)
    {
        fprintf(stderr, "usage: %s [demo] [model_path] [dataset] [input] [input_size] [save] ..\n", argv[0]);
        return -1;
    }
    std::string demo = argv[1];
    if (demo == "track")
    {
        track_demo(argc, argv);
    }
    else
    {
        std::cout << "unrecognized demo type: "  << argv[1] << std::endl;
        exit(EXIT_FAILURE);
    }
    return 0;
}