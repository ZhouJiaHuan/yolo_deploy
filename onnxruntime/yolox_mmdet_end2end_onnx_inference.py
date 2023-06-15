'''
yolox onnxruntime inference
the model was trained with mmdetection3.0 and converted with mmdeploy
github:
https://github.com/open-mmlab/mmdetection (3.0.0)
https://github.com/open-mmlab/mmdeploy (1.1.0)

'''

import os
import cv2
import argparse
import onnxruntime
import numpy as np
from typing import List, Tuple, Dict

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

    
class YoloxInferenceEnd2end(object):
    ''' yolox onnxruntime end2end inference from mmdetection 3.0
    '''

    def __init__(self,
                 onnx_path: str,
                 input_size: Tuple[int],
                 class_names: Tuple[str],
                 score_thr=0.25) -> None:
        assert onnx_path.endswith('.onnx'), f"invalid onnx model: {onnx_path}"
        assert os.path.exists(onnx_path), f"model not found: {onnx_path}"
        self.sess = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print("input info: ", self.sess.get_inputs()[0])
        print("output info: ", self.sess.get_outputs()[0])
        self.input_size = input_size
        self.class_names = class_names
        self.score_thr = score_thr
        np.random.seed(0)
        self.color_list = np.random.randint(0, 255, (len(class_names), 3)).tolist()

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        ''' preprocess image for model inference
        '''
        input_w, input_h = self.input_size
        if len(img.shape) == 3:
            padded_img = np.ones((input_w, input_h, 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114
        r = min(input_w / img.shape[0], input_h / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        # (H, W, C) BGR -> (C, H, W) RGB
        padded_img = padded_img.transpose((2, 0, 1))[::-1, ]
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    def _postprocess(self, output: List[np.ndarray], ratio) -> Dict:
        result = {'boxes': [], 'idxes': [], 'scores': []}
        pred = output[0][0]
        class_idx = output[1][0]
        mask = pred[:, -1] > self.score_thr
        pred = pred[mask]
        class_idx = class_idx[mask]

        if pred.shape[0] == 0:
            return result
        
        boxes = pred[:, :4] / ratio
        scores = pred[:, 4]
        result['boxes'] = boxes.tolist()
        result['idxes'] = class_idx.tolist()
        result['scores'] = scores.tolist()
        return result
    
    def detect(self, img: np.ndarray) -> Dict:
        img, ratio = self._preprocess(img)
        ort_input = {self.sess.get_inputs()[0].name: img[None, :]}
        output = self.sess.run(None, ort_input)
        result = self._postprocess(output, ratio)
        return result
    
    def draw_result(self, img: np.ndarray, result: Dict) -> np.ndarray:
        boxes = result['boxes']
        idxes = result['idxes']
        scores = result['scores']

        for box, idx, score in zip(boxes, idxes, scores):
            x1, y1, x2, y2 = list(map(int, box))
            cls_name = self.class_names[idx]
            color = self.color_list[idx]
            label = "{}: {:.0f}%".format(cls_name, score*100)
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y1), (x1+label_size[0], y1+label_size[1]+baseline),
                color, -1)
            cv2.putText(img, label, (x1, y1+label_size[1]), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
        return img

def parse_args():
    parser = argparse.ArgumentParser("yolov8 onnx inference demo")
    parser.add_argument("demo", type=str,
        help="demo type, image or stream")
    parser.add_argument("model", type=str,
        help="onnx model path")
    parser.add_argument("input", type=str, default="0",
        help="camera id | video path | image path")
    parser.add_argument("--score_thr", type=float, default=0.25,
        help="score threshold")
    parser.add_argument("--input_size", type=int, default=640,
        help="input size")
    parser.add_argument("--classes_txt", type=str, default='',
        help="class names txt file.")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()

def demo_image(args):
    input_size = (args.input_size, args.input_size)
    class_names = COCO_CLASSES
    if os.path.exists(args.classes_txt):
        with open(args.classes_txt, 'r') as f:
            class_names = tuple([name.strip() for name in f.readlines()])

    yolo = YoloxInferenceEnd2end(args.model, input_size, class_names, args.score_thr)
    img = cv2.imread(args.input)
    if img is None:
        print(f"read image failed: {args.input}")
        return
    result = yolo.detect(img)
    img = yolo.draw_result(img, result)
    boxes = result['boxes']
    idxes = result['idxes']
    scores = result['scores']
    for box, idx, score in zip(boxes, idxes, scores):
        box = list(map(int, box))
        print(f"{class_names[idx]}: {score:.3}\t{box}")
        
    if (args.show):
        cv2.imshow("detect", img)
        cv2.waitKey(0)

def demo_stream(args):
    input_size = (args.input_size, args.input_size)
    class_names = COCO_CLASSES
    if os.path.exists(args.classes_txt):
        with open(args.classes_txt, 'r') as f:
            class_names = tuple([name.strip() for name in f.readlines()])

    yolo = YoloxInferenceEnd2end(args.model, input_size, class_names, args.score_thr)

    if (len(args.input)) == 1:
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = yolo.detect(frame)
        frame = yolo.draw_result(frame, result)
        if args.show:
            cv2.imshow("detect", frame)
            if cv2.waitKey(1) == ord('q'):  # Esc
                break
        else:
            num = len(result['boxes'])
            print(f"detect {num} objects")

def main():
    args = parse_args()
    if args.demo == 'image':
        return demo_image(args)
    elif args.demo == 'stream':
        return demo_stream(args)

if __name__ == "__main__":
    main()
