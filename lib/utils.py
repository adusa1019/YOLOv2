from chainer import Variable
import chainer.functions as F
import numpy as np


# x, y, w, hの4パラメータを保持するだけのクラス
class Box():

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def int_left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return round(self.x - half_width), round(self.y - half_height)

    def left_top(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return self.x - half_width, self.y - half_height

    def int_right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return round(self.x + half_width), round(self.y + half_height)

    def right_bottom(self):
        half_width = self.w / 2
        half_height = self.h / 2
        return [self.x + half_width, self.y + half_height]

    def crop_region(self, h, w):
        left, top = self.left_top()
        right, bottom = self.right_bottom()
        left = max(0, left)
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        self.w = right - left
        self.h = bottom - top
        self.x = (right + left) / 2
        self.y = (bottom + top) / 2
        return self


# 2本の線の情報を受取り、被ってる線分の長さを返す。あくまで線分
def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# chainerのVariable用のoverlap
def multi_overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = F.maximum(x1 - len1_half, x2 - len2_half)
    right = F.minimum(x1 + len1_half, x2 + len2_half)

    return right - left


# 2つのboxを受け取り、被ってる面積を返す(intersection of 2 boxes)
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# chainer用
def multi_box_intersection(a, b):
    w = multi_overlap(a.x, a.w, b.x, b.w)
    h = multi_overlap(a.y, a.h, b.y, b.h)
    zeros = Variable(np.zeros(w.shape, dtype=w.data.dtype))
    zeros.to_gpu()

    w = F.maximum(w, zeros)
    h = F.maximum(h, zeros)

    area = w * h
    return area


# 2つのboxを受け取り、合計面積を返す。(union of 2 boxes)
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# chianer用
def multi_box_union(a, b):
    i = multi_box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# compute iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


# chainer用
def multi_box_iou(a, b):
    return multi_box_intersection(a, b) / multi_box_union(a, b)


# non maximum suppression
def nms(predicted_results, iou_thresh):
    predicted_results = sorted(predicted_results, key=lambda x: x["conf"], reverse=True)
    nms_results = [
        result for i, result in enumerate(predicted_results)
        if all((box_iou(result["box"], r["box"]) < iou_thresh for r in predicted_results[:i]))
    ]
    return nms_results


# reshape to yolo size
def reshape_to_yolo_size(img):
    GRID_SIZE = 32
    width, height = img.size
    MIN_PIXEL = 320
    MAX_PIXEL = 416

    min_edge = min(width, height)
    if min_edge < MIN_PIXEL:
        width *= MIN_PIXEL / min_edge
        height *= MIN_PIXEL / min_edge
    max_edge = max(width, height)
    if max_edge > MAX_PIXEL:
        width *= MAX_PIXEL / max_edge
        height *= MAX_PIXEL / max_edge

    width = round(width / GRID_SIZE) * GRID_SIZE
    height = round(height / GRID_SIZE) * GRID_SIZE
    return img.resize((width, height))
