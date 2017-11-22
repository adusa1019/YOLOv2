# coding=utf-8

import argparse
import os
import sys

import numpy as np
from chainer import serializers

# 非パッケージライブラリの呼び出し用にパス追加
path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(path)
from yolov2 import YOLOv2

parser = argparse.ArgumentParser(description="darknet weight file to chainer weight file")
parser.add_argument('file', help="Weight file path")
args = parser.parse_args()

print("loading", args.file)
with open(args.file, "rb") as f:
    dat = np.fromfile(f, dtype=np.float32)[5:]  # skip header(5xint)

# load model
print("loading initial model...")
n_classes = 80
n_boxes = 5
last_out = (n_classes + 5) * n_boxes

yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)

layers = [
    [3, 32, 3],
    [32, 64, 3],
    [64, 128, 3],
    [128, 64, 1],
    [64, 128, 3],
    [128, 256, 3],
    [256, 128, 1],
    [128, 256, 3],
    [256, 512, 3],
    [512, 256, 1],
    [256, 512, 3],
    [512, 256, 1],
    [256, 512, 3],
    [512, 1024, 3],
    [1024, 512, 1],
    [512, 1024, 3],
    [1024, 512, 1],
    [512, 1024, 3],
    [1024, 1024, 3],
    [1024, 1024, 3],
    [3072, 1024, 3],
]

offset = 0
for i, l in enumerate(layers):
    in_ch = l[0]
    out_ch = l[1]
    ksize = l[2]

    # load bias(Bias.bはout_chと同じサイズ)
    txt = "yolov2.bias%d.b.data = dat[%d:%d]" % (i + 1, offset, offset + out_ch)
    offset += out_ch
    exec(txt)

    # load bn(BatchNormalization.gammaはout_chと同じサイズ)
    txt = "yolov2.bn%d.gamma.data = dat[%d:%d]" % (i + 1, offset, offset + out_ch)
    offset += out_ch
    exec(txt)

    # (BatchNormalization.avg_meanはout_chと同じサイズ)
    txt = "yolov2.bn%d.avg_mean = dat[%d:%d]" % (i + 1, offset, offset + out_ch)
    offset += out_ch
    exec(txt)

    # (BatchNormalization.avg_varはout_chと同じサイズ)
    txt = "yolov2.bn%d.avg_var = dat[%d:%d]" % (i + 1, offset, offset + out_ch)
    offset += out_ch
    exec(txt)

    # load convolution weight
    # (Convolution2D.Wは、outch * in_ch * フィルタサイズ
    # これを(out_ch, in_ch, 3, 3)にreshapeする)
    txt = "yolov2.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % (
        i + 1, offset, offset + (out_ch * in_ch * ksize * ksize), out_ch, in_ch, ksize, ksize)
    offset += (out_ch * in_ch * ksize * ksize)
    exec(txt)
    print(i + 1, offset)

# load last convolution weight(BiasとConvolution2Dのみロードする)
in_ch = 1024
out_ch = last_out
ksize = 1

txt = "yolov2.bias%d.b.data = dat[%d:%d]" % (i + 2, offset, offset + out_ch)
offset += out_ch
exec(txt)

txt = "yolov2.conv%d.W.data = dat[%d:%d].reshape(%d, %d, %d, %d)" % (
    i + 2, offset, offset + (out_ch * in_ch * ksize * ksize), out_ch, in_ch, ksize, ksize)
offset += out_ch * in_ch * ksize * ksize
exec(txt)
print(i + 2, offset)

print("save weights file to yolov2_darknet.model")
serializers.save_hdf5("yolov2_darknet.model", yolov2)
