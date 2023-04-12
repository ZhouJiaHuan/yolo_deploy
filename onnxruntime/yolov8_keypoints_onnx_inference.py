'''
yolov8-pose onnxruntime inference
github: https://github.com/ultralytics/ultralytics
'''

import os
import numpy as np
import cv2
import argparse
import json
import onnxruntime
from tqdm import tqdm
from typing import List, Tuple, Dict

COLOR_LIST = list([[128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
                   [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
                   [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0],
                   [255, 0, 51], [153, 0, 153], [51, 0, 51], [0, 0, 0],
                   [0, 102, 255], [0, 51, 255], [0, 153, 255], [0, 153, 153]])

def xywh2xyxy(box: np.ndarray) -> np.ndarray:
    box_xyxy = box.copy()
    box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
    box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
    box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
    box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
    return box_xyxy

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    '''
    box and boxes are format as [x1, y1, x2, y2]
    '''
    # inter area
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, xmax-xmin) * np.maximum(0, ymax-ymin)

    # union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return inter_area / union_area

def nms_process(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    sorted_idx = np.argsort(scores)[::-1]
    keep_idx = []
    while sorted_idx.size > 0:
        idx = sorted_idx[0]
        keep_idx.append(idx)
        ious = compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])
        rest_idx = np.where(ious < iou_thr)[0]
        sorted_idx = sorted_idx[rest_idx+1]
    return keep_idx

class Yolov8KeypointsInference(object):
    ''' yolov8-keypoints onnxruntime inference
    '''

    def __init__(self,
                 onnx_path: str,
                 input_size: Tuple[int],
                 box_score=0.25,
                 kpt_score=0.5,
                 nms_thr=0.2
                 ) -> None:
        assert onnx_path.endswith('.onnx'), f"invalid onnx model: {onnx_path}"
        assert os.path.exists(onnx_path), f"model not found: {onnx_path}"
        self.sess = onnxruntime.InferenceSession(onnx_path)
        print("input info: ", self.sess.get_inputs()[0])
        print("output info: ", self.sess.get_outputs()[0])
        self.input_size = input_size
        self.box_score = box_score
        self.kpt_score = kpt_score
        self.nms_thr = nms_thr

    def _preprocess(self, img: np.ndarray):
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
        predict = output[0].squeeze(0).T
        predict = predict[predict[:, 4] > self.box_score, :]
        scores = predict[:, 4]
        boxes = predict[:, 0:4] / ratio
        boxes = xywh2xyxy(boxes)
        kpts = predict[:, 5:]
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3*j+2] < self.kpt_score:
                    kpts[i, 3*j: 3*(j+1)] = [-1, -1, -1]
                else:
                    kpts[i, 3*j] /= ratio
                    kpts[i, 3*j+1] /= ratio 
        idxes = nms_process(boxes, scores, self.nms_thr)
        result = {'boxes': boxes[idxes,: ].astype(int).tolist(),
                  'kpts': kpts[idxes,: ].astype(float).tolist(),
                  'scores': scores[idxes].tolist()}
        return result

    def detect(self, img: np.ndarray) -> Dict:
        img, ratio = self._preprocess(img)
        ort_input = {self.sess.get_inputs()[0].name: img[None, :]/255}
        output = self.sess.run(None, ort_input)
        result = self._postprocess(output, ratio)
        return result
    
    def draw_result(self, img: np.ndarray, result: Dict, with_label=False) -> np.ndarray:
        boxes, kpts, scores = result['boxes'], result['kpts'], result['scores']
        for box, kpt, score in zip(boxes, kpts, scores):
            x1, y1, x2, y2 = box
            label_str = "{:.0f}%".format(score*100)
            label_size, baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if with_label:
                cv2.rectangle(img, (x1, y1), (x1+label_size[0], y1+label_size[1]+baseline),
                    (0, 0, 255), -1)
                cv2.putText(img, label_str, (x1, y1+label_size[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
            for idx in range(len(kpt) // 3):
                x, y, score = kpt[3*idx: 3*(idx+1)]
                if score > 0:
                    cv2.circle(img, (int(x), int(y)), 2, COLOR_LIST[idx], -1)
        return img

def parse_args():
    parser = argparse.ArgumentParser("yolov8 keypoints onnx inference demo")
    parser.add_argument("demo", type=str,
        help="demo type, image | stream")
    parser.add_argument("model", type=str,
        help="onnx model path")
    parser.add_argument("input", type=str, default="0",
        help="camera id | video path | image path | image dir")
    parser.add_argument("--box_score", type=float, default=0.25,
        help="box score threshold")
    parser.add_argument("--kpt_score", type=float, default=0.5,
        help="keypoint score threshold")
    parser.add_argument("--nms_thr", type=float, default=0.5,
        help="nms threshold")
    parser.add_argument("--input_size", type=int, default=640,
        help="input size")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()

def demo_image(args):
    input_size = (args.input_size, args.input_size)
    yolo = Yolov8KeypointsInference(args.model, input_size, args.box_score, args.kpt_score, args.nms_thr)
    img = cv2.imread(args.input)
    if img is None:
        print(f"read image failed: {args.input}")
        return
    result = yolo.detect(img)
    img = yolo.draw_result(img, result)
    boxes = result['boxes']
    kpts = result['kpts']
    scores = result['scores']
    for box, kpt, score in zip(boxes, kpts, scores):
        kpt = list(map(lambda k: round(k, 3), kpt))
        print(f"{score:.3}\t{box}\t{kpt}")
      
    if args.show:
        cv2.imshow("detect", img)
        cv2.waitKey(0)

def demo_image_dir(args):
    input_size = (args.input_size, args.input_size)
    yolo = Yolov8KeypointsInference(args.model, input_size, args.box_score, args.kpt_score, args.nms_thr)

    assert os.path.isdir(args.input), f"invalid image directory: {args.input}"
    img_list = [os.path.basename(img_name) for img_name in os.listdir(args.input) 
                if img_name.split('.')[-1] in ('jpg', 'png')]
    save_dir = 'result_' + os.path.abspath(args.input).replace('/', '_')
    os.makedirs(save_dir, exist_ok=True)
    for img_name in tqdm(img_list):
        img_path = os.path.join(args.input, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f'read image failed: {img_path}')
            continue
        result = yolo.detect(img)
        img_result = yolo.draw_result(img, result)
        cv2.imwrite(os.path.join(save_dir, img_name), img_result)
        with open(os.path.join(save_dir, img_name.split('.')[0]+'.json'), 'w') as f:
            result['img_name'] = img_name
            result['img_size'] = img.shape[:2][::-1]
            json.dump(result, f)
        if args.show:
            cv2.imshow('detect', img_result)
            cv2.waitKey(100)

def demo_stream(args):
    input_size = (args.input_size, args.input_size)
    yolo = Yolov8KeypointsInference(args.model, input_size, args.box_score, args.kpt_score, args.nms_thr)

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
        if os.path.isfile(args.input):
            return demo_image(args)
        elif os.path.isdir(args.input):
            return demo_image_dir(args)
        else:
            print(f"invalid input: {args.input}")
            return
    elif args.demo == 'stream':
        return demo_stream(args)

if __name__ == "__main__":
    main()