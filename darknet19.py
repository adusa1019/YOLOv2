#!/usr/bin/env python
# coding=utf-8

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from lib.functions import sum_of_squared_error


class Darknet19(chainer.Chain):
    """
    Darknet19
    - It takes (224, 224, 3) or (448, 448, 4) sized image as input
    """

    def __init__(self):
        super(Darknet19, self).__init__()
        with self.init_scope():
            # common layers for both pretrained layers and yolov2
            self.conv1 = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(32, use_beta=False)
            self.bias1 = L.Bias(shape=(32,))
            self.conv2 = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(64, use_beta=False)
            self.bias2 = L.Bias(shape=(64,))
            self.conv3 = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True)
            self.bn3 = L.BatchNormalization(128, use_beta=False)
            self.bias3 = L.Bias(shape=(128,))
            self.conv4 = L.Convolution2D(128, 64, ksize=1, stride=1, pad=0, nobias=True)
            self.bn4 = L.BatchNormalization(64, use_beta=False)
            self.bias4 = L.Bias(shape=(64,))
            self.conv5 = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True)
            self.bn5 = L.BatchNormalization(128, use_beta=False)
            self.bias5 = L.Bias(shape=(128,))
            self.conv6 = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True)
            self.bn6 = L.BatchNormalization(256, use_beta=False)
            self.bias6 = L.Bias(shape=(256,))
            self.conv7 = L.Convolution2D(256, 128, ksize=1, stride=1, pad=0, nobias=True)
            self.bn7 = L.BatchNormalization(128, use_beta=False)
            self.bias7 = L.Bias(shape=(128,))
            self.conv8 = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True)
            self.bn8 = L.BatchNormalization(256, use_beta=False)
            self.bias8 = L.Bias(shape=(256,))
            self.conv9 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True)
            self.bn9 = L.BatchNormalization(512, use_beta=False)
            self.bias9 = L.Bias(shape=(512,))
            self.conv10 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True)
            self.bn10 = L.BatchNormalization(256, use_beta=False)
            self.bias10 = L.Bias(shape=(256,))
            self.conv11 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True)
            self.bn11 = L.BatchNormalization(512, use_beta=False)
            self.bias11 = L.Bias(shape=(512,))
            self.conv12 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True)
            self.bn12 = L.BatchNormalization(256, use_beta=False)
            self.bias12 = L.Bias(shape=(256,))
            self.conv13 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True)
            self.bn13 = L.BatchNormalization(512, use_beta=False)
            self.bias13 = L.Bias(shape=(512,))
            self.conv14 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True)
            self.bn14 = L.BatchNormalization(1024, use_beta=False)
            self.bias14 = L.Bias(shape=(1024,))
            self.conv15 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True)
            self.bn15 = L.BatchNormalization(512, use_beta=False)
            self.bias15 = L.Bias(shape=(512,))
            self.conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True)
            self.bn16 = L.BatchNormalization(1024, use_beta=False)
            self.bias16 = L.Bias(shape=(1024,))
            self.conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True)
            self.bn17 = L.BatchNormalization(512, use_beta=False)
            self.bias17 = L.Bias(shape=(512,))
            self.conv18 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True)
            self.bn18 = L.BatchNormalization(1024, use_beta=False)
            self.bias18 = L.Bias(shape=(1024,))
            # new layer
            self.conv19 = L.Convolution2D(1024, 10, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        batch_size = x.shape[0]

        # common layer
        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x))), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h))), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h))), slope=0.1)
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h))), slope=0.1)
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h))), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h))), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h))), slope=0.1)
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h))), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias9(self.bn9(self.conv9(h))), slope=0.1)
        h = F.leaky_relu(self.bias10(self.bn10(self.conv10(h))), slope=0.1)
        h = F.leaky_relu(self.bias11(self.bn11(self.conv11(h))), slope=0.1)
        h = F.leaky_relu(self.bias12(self.bn12(self.conv12(h))), slope=0.1)
        h = F.leaky_relu(self.bias13(self.bn13(self.conv13(h))), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h))), slope=0.1)
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h))), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h))), slope=0.1)
        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h))), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h))), slope=0.1)
        # new layer
        h = self.conv19(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)
        # reshape
        y = F.reshape(h, (batch_size, -1))
        return y


class Darknet19Predictor(chainer.Chain):

    def __init__(self, predictor):
        super(Darknet19Predictor, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        if t.ndim == 2:  # use squared error when label is one hot label
            print("bad")
            y = F.softmax(y)
            # loss = F.mean_squared_error(y, t)
            loss = sum_of_squared_error(y, t)
            accuracy = F.accuracy(y, t.data.argmax(axis=1).astype(np.int32))
        else:  # use softmax cross entropy when label is normal label
            # print("good")
            loss = F.softmax_cross_entropy(y, t)
            accuracy = F.accuracy(y, t)
        chainer.report({
            'loss': loss,
            'accuracy': accuracy,
        }, self)
        return loss

    def predict(self, x):
        y = self.predictor(x)
        return F.softmax(y)
