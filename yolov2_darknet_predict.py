# coding=utf-8

import argparse
import os
import sys
import time

import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
from PIL import Image, ImageDraw, ImageFont

# 非パッケージライブラリの呼び出し用にパス追加
path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)


class CocoPredictor:

    def __init__(self, class_file):
        from yolov2 import YOLOv2, YOLOv2Predictor
        # hyper parameters
        weight_file = "configure/yolov2_darknet.model"
        self.n_classes = 80
        self.n_boxes = 5
        self.detection_thresh = 0.4
        self.iou_thresh = 0.4

        try:
            with open(class_file, 'r') as f:
                self.labels = f.read().strip().split()
        except FileNotFoundError as e:
            raise e
        anchors = [[0.738768, 0.874946], [2.42204, 2.65704], [4.30971, 7.04493], [10.246, 4.59428],
                   [12.6868, 11.8741]]

        # load model
        print("loading coco model...")
        yolov2 = YOLOv2(n_classes=self.n_classes, n_boxes=self.n_boxes)
        serializers.load_hdf5(weight_file, yolov2)  # load saved model
        model = YOLOv2Predictor(yolov2)
        model.init_anchor(anchors)
        model.predictor.train = False
        model.predictor.finetune = False
        self.model = model

    def __call__(self, orig_img):
        from lib.utils import Box, nms, reshape_to_yolo_size
        orig_input_width, orig_input_height = orig_img.size
        img = reshape_to_yolo_size(orig_img)
        input_width, input_height = img.size
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # forward
        x_data = img[np.newaxis, :, :, :]
        x = Variable(x_data)
        x, y, w, h, conf, prob = self.model.predict(x)

        # parse results
        _, _, _, grid_h, grid_w = x.shape
        x = F.reshape(x, (self.n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (self.n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (self.n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (self.n_boxes, grid_h, grid_w)).data
        conf = F.reshape(conf, (self.n_boxes, grid_h, grid_w)).data
        prob = F.transpose(
            F.reshape(prob, (self.n_boxes, self.n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
        detected_indices = (conf * prob).max(axis=0) > self.detection_thresh

        prob = prob.transpose(1, 2, 3, 0)
        results = [{
            "class_id":
            prob[detected_indices][i].argmax(),
            "label":
            self.labels[prob[detected_indices][i].argmax()],
            "probs":
            prob[detected_indices][i],
            "conf":
            conf[detected_indices][i],
            "objectness":
            conf[detected_indices][i] * prob[detected_indices][i].max(),
            "box":
            Box(x[detected_indices][i] * orig_input_width,
                y[detected_indices][i] * orig_input_height,
                w[detected_indices][i] * orig_input_width,
                h[detected_indices][i] * orig_input_height).crop_region(
                    orig_input_height, orig_input_width)
        } for i in range(detected_indices.sum())]

        # nms
        nms_results = nms(results, self.iou_thresh)
        return nms_results


if __name__ == "__main__":
    # argument parse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('path', help="Root image directory path")
    parser.add_argument(
        '--name', '-n', default='configure/names.txt', help="Class name definition file")
    args = parser.parse_args()
    image_file = args.path

    start = time.time()
    # read image
    print("loading image...")
    orig_img = Image.open(image_file)

    predictor = CocoPredictor(args.name)
    print("after load: {}".format(time.time() - start))
    nms_results = predictor(orig_img)

    # draw result
    draw = ImageDraw.Draw(orig_img)
    with open("results/yolov2_result_%s.txt" % path.split("\\")[-1], "w") as f:
        for result in nms_results:
            left, top = result["box"].int_left_top()
            right, bottom = result["box"].int_right_bottom()
            draw.rectangle(
                (result["box"].int_left_top(), result["box"].int_right_bottom()),
                outline=(0, 255, 0))
            text = '%s(%2d%%)' % (result["label"], result["probs"].max() * result["conf"] * 100)
            font = ImageFont.truetype("C:\\Windows\\Fonts\\msgothic.ttc", 24)
            draw.text((left, bottom - 24), text, font=font)
            # print(text)
            f.write(text + "\n")

    print("after predict: {}".format(time.time() - start))
    print("save results to yolov2_result_%s.jpg" % path.split("\\")[-1])
    orig_img.save("results/yolov2_result_%s.jpg" % path.split("\\")[-1])
