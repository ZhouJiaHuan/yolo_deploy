'''
yolov7-face onnxruntime inference
github: https://github.com/derronqi/yolov7-face
onnx models should be exported with nms, refer to official github for more details
'''

import os
import numpy as np
import cv2
import argparse
import onnxruntime
from typing import Tuple, Dict

COLOR_LIST = list([[128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
                   [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
                   [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0]])


class Yolov7KeypointsInference(object):
    ''' yolov7-face onnxruntime inference
    '''

    def __init__(self,
                 onnx_path: str,
                 input_size: Tuple[int],
                 box_score=0.25,
                 kpt_score=1.5
                 ) -> None:
        assert onnx_path.endswith('.onnx'), f"invalid onnx model: {onnx_path}"
        assert os.path.exists(onnx_path), f"model not found: {onnx_path}"
        self.sess = onnxruntime.InferenceSession(onnx_path)
        print("input info: ", self.sess.get_inputs()[0])
        print("output info: ", self.sess.get_outputs()[0])
        self.input_size = input_size
        self.box_score = box_score
        self.kpt_score = kpt_score

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
    
    def _postprocess(self, output: np.ndarray, ratio) -> Dict:
        boxes = output[0][:, 0:4] / ratio
        scores, labels = output[0][:, 4], output[0][:, 5]
        idxes = scores > self.box_score
        kpts = output[0][:, 6:][idxes, :].tolist()
        for kpt in kpts:
            for idx in range(len(kpt) // 3):
                if kpt[3*idx+2] < self.kpt_score:
                    kpt[3*idx: 3*(idx+1)] = [-1, -1, -1]
                else:
                    kpt[3*idx] /= ratio
                    kpt[3*idx+1] /= ratio 
        result = {'boxes': boxes[idxes,: ].astype(int).tolist(),
                  'kpts': kpts,
                  'scores': scores[idxes].tolist(),
                  'labels': labels[idxes].astype(int).tolist()}
        return result

    def detect(self, img: np.ndarray) -> Dict:
        img, ratio = self._preprocess(img)
        ort_input = {self.sess.get_inputs()[0].name: img[None, :]/255}
        output = self.sess.run(None, ort_input)
        result = self._postprocess(output, ratio)
        return result
    
    def draw_result(self, img: np.ndarray, result: Dict, with_label=False) -> np.ndarray:
        boxes, kpts = result['boxes'], result['kpts']
        scores, labels = result['scores'], result['labels']
        for box, kpt, score, label in zip(boxes, kpts, scores, labels):
            x1, y1, x2, y2 = box
            label_str = "{}: {:.0f}%".format(label, score*100)
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
    parser = argparse.ArgumentParser("yolov7 keypoints onnx inference demo")
    parser.add_argument("demo", type=str,
        help="demo type, image or stream")
    parser.add_argument("model", type=str,
        help="onnx model path")
    parser.add_argument("input", type=str, default="0",
        help="camera id | video path | image path")
    parser.add_argument("--box_score", type=float, default=0.25,
        help="box score threshold")
    parser.add_argument("--kpt_score", type=float, default=0.5,
        help="keypoint score threshold")
    parser.add_argument("--input_size", type=int, default=640,
        help="input size")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def demo_image(args):
    input_size = (args.input_size, args.input_size)

    yolo = Yolov7KeypointsInference(args.model, input_size, args.box_score, args.kpt_score)
    img = cv2.imread(args.input)
    if img is None:
        print(f"read image failed: {args.input}")
        return
    result = yolo.detect(img)
    img = yolo.draw_result(img, result)
    boxes = result['boxes']
    kpts = result['kpts']
    labels = result['labels']
    scores = result['scores']
    for box, kpt, label, score in zip(boxes, kpts, labels, scores):
        kpt = list(map(lambda k: round(k, 3), kpt))
        print(f"{int(label)}: {score:.3}\t{box}\t{kpt}")
      
    if (args.show):
        cv2.imshow("detect", img)
        cv2.waitKey(0)


def demo_stream(args):
    input_size = (args.input_size, args.input_size)
    yolo = Yolov7KeypointsInference(args.model, input_size, args.box_score, args.kpt_score)

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