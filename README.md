<img src="./assets/track_result.gif" alt="track_result" style="zoom: 150%;" />

## Introduction

This repository implemented the inference of yolo detectors with **ONNXRuntime**, **MNN** and **TensorRT** framework for fast deployment: 

|       model       |                    Github                     | ONNX | MNN  | TensorRT |
| :---------------: | :-------------------------------------------: | :--: | :--: | :------: |
|       YOLOx       | https://github.com/Megvii-BaseDetection/YOLOX |  N   |  Y   |    N     |
| YOLOx (mmdet 3.0) |   https://github.com/open-mmlab/mmdetection   |  Y   |  Y   |    N     |
|      YOLOv6       |       https://github.com/meituan/YOLOv6       |  N   |  Y   |    Y     |
|      YOLOv8       |  https://github.com/ultralytics/ultralytics   |  Y   |  Y   |    Y     |
| YOLOv7-keypoints  |    https://github.com/derronqi/yolov7-face    |  Y   |  N   |    N     |
| YOLOv8-keypoints  |  https://github.com/ultralytics/ultralytics   |  Y   |  Y   |    Y     |
|    YOLOv8-sot     |  https://github.com/ultralytics/ultralytics   |  N   |  N   |    Y     |

## Requirements

- MNN >= 2.0.0
- ONNXRuntime >=1.8.1
- TensorRT >= 8.5

## Demo

```shell
## MNN inference (Yolov8)
cd mnn/yolov8/
mkdir build && cd build
cmake ..
make
./yolov8_mnn [demo] [model_path] [dataset] [input] [input_size] [show]


## ONNXRuntime inference
python yolov8_onnx_inference.py DEMO MODEL INPUT [--score_thr SCORE_THR] [--nms_thr NMS_THR] [--input_size INPUT_SIZE] [--classes_txt CLASSES_TXT] [--show]


## TensorRT inference
# you may need to modify the TensorRT path in CMakeLists.txt according to your install path
# include_directories(/opt/TensorRT-8.5.1.7/include)
# link_directories(/opt/TensorRT-8.5.1.7/lib)
cd tensorrt/yolov8/
mkdir build && cd build
cmake ..
make
./yolov8_trt [demo] [model_path] [dataset] [input] [input_size] [show]
```

Yolov8-n inference with image using MNN:

```shell
/yolov8_mnn image /path/to/yolov8n_coco_640_fp16.mnn coco ../../../assets/bus.jpg 640

# output
detect 6 objects in 51.0658 ms
person: 0.873672    ([669.938, 376.312], [808.312, 877.5])
person: 0.868409    ([47.25, 398.25], [243, 901.125])
bus: 0.862305    ([20.25, 229.5], [796.5, 754.312])
person: 0.819331    ([221.062, 405], [344.25, 855.562])
stop sign: 0.343525    ([0, 253.125], [32.0625, 322.312])
person: 0.3012    ([0, 550.125], [65.8125, 872.438])
```

Yolov8-n inference with camera using MNN:

```shell
./yolov8_mnn stream /path/to/yolov8n_coco_640_fp16.mnn coco 0 640 1
```

Yolov8-n inference with image using ONNXRuntime

```shell
python yolov8_onnx_inference.py image /path/to/yolov8n_coco_640.onnx ../assets/bus.jpg --input_size 640

# output
input info:  NodeArg(name='images', type='tensor(float)', shape=[1, 3, 640, 640])
output info:  NodeArg(name='output0', type='tensor(float)', shape=[1, 84, 8400])
person: 0.875    [670, 376, 809, 878]
person: 0.869    [48, 399, 244, 902]
bus: 0.863    [21, 229, 798, 754]
person: 0.82    [221, 405, 344, 857]
stop sign: 0.346    [0, 254, 32, 325]
person: 0.301    [0, 551, 67, 873]
```

Yolov8-n inference with camera using ONNXRuntime

```shell
python yolov8_onnx_inference.py stream /path/to/yolov8n_coco_640.onnx 0 --input_size 640 --show
```

Yolov8-n inference with image using TensorRT

```shell
./yolov8_trt image /path/to/yolov8n_coco_640_fp16.trt coco ../../../assets/bus.jpg 640

# output
detect 6 objects in 386.102 ms
person: 0.875488        ([669.938, 376.312], [808.312, 877.5])
person: 0.868164        ([48.9375, 398.25], [243, 901.125])
bus: 0.862305   ([20.25, 229.5], [796.5, 754.312])
person: 0.820312        ([221.062, 405], [344.25, 855.562])
stop sign: 0.348633     ([0, 253.125], [32.0625, 322.312])
person: 0.301514        ([0, 550.125], [65.8125, 872.438])
```

Yolov8-n inference with camera using TensorRT

```shell
./yolov8_trt stream /path/to/yolov8n_coco_640_fp16.trt coco 0 640 1
```

Yolov8-n SOT with camera using TensorRT

```bash
./yolov8_sot track /path/to/yolov8n_coco_640_fp16.trt coco 0 640
# select your tracking target by clicking the detection bounding box with mouse left button
```

Yolov8-n SOT with video file using TensorRT

```bash
./yolov8_sot track /path/to/yolov8n_coco_640_fp16.trt coco /path/to/video.mp4 640
# select your tracking target by clicking the detection bounding box with mouse left button
```

