import numpy as np
import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F

from lib.utils import Box, box_iou, multi_box_iou
from lib.functions import reorg


class YOLOv2(chainer.Chain):
    """
    YOLOv2 network definition
    - It takes (416, 416, 3) sized image as input
    """

    def __init__(self, n_classes, n_boxes):
        super(YOLOv2, self).__init__()
        with self.init_scope():
            # common layers for both pretrained layers and yolov2 #
            self.conv1 = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(32, use_beta=False, eps=2e-5)
            self.bias1 = L.Bias(shape=(32,))
            self.conv2 = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(64, use_beta=False, eps=2e-5)
            self.bias2 = L.Bias(shape=(64,))
            self.conv3 = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True)
            self.bn3 = L.BatchNormalization(128, use_beta=False, eps=2e-5)
            self.bias3 = L.Bias(shape=(128,))
            self.conv4 = L.Convolution2D(128, 64, ksize=1, stride=1, pad=0, nobias=True)
            self.bn4 = L.BatchNormalization(64, use_beta=False, eps=2e-5)
            self.bias4 = L.Bias(shape=(64,))
            self.conv5 = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True)
            self.bn5 = L.BatchNormalization(128, use_beta=False, eps=2e-5)
            self.bias5 = L.Bias(shape=(128,))
            self.conv6 = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True)
            self.bn6 = L.BatchNormalization(256, use_beta=False, eps=2e-5)
            self.bias6 = L.Bias(shape=(256,))
            self.conv7 = L.Convolution2D(256, 128, ksize=1, stride=1, pad=0, nobias=True)
            self.bn7 = L.BatchNormalization(128, use_beta=False, eps=2e-5)
            self.bias7 = L.Bias(shape=(128,))
            self.conv8 = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True)
            self.bn8 = L.BatchNormalization(256, use_beta=False, eps=2e-5)
            self.bias8 = L.Bias(shape=(256,))
            self.conv9 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True)
            self.bn9 = L.BatchNormalization(512, use_beta=False, eps=2e-5)
            self.bias9 = L.Bias(shape=(512,))
            self.conv10 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True)
            self.bn10 = L.BatchNormalization(256, use_beta=False, eps=2e-5)
            self.bias10 = L.Bias(shape=(256,))
            self.conv11 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True)
            self.bn11 = L.BatchNormalization(512, use_beta=False, eps=2e-5)
            self.bias11 = L.Bias(shape=(512,))
            self.conv12 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True)
            self.bn12 = L.BatchNormalization(256, use_beta=False, eps=2e-5)
            self.bias12 = L.Bias(shape=(256,))
            self.conv13 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True)
            self.bn13 = L.BatchNormalization(512, use_beta=False, eps=2e-5)
            self.bias13 = L.Bias(shape=(512,))
            self.conv14 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True)
            self.bn14 = L.BatchNormalization(1024, use_beta=False, eps=2e-5)
            self.bias14 = L.Bias(shape=(1024,))
            self.conv15 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True)
            self.bn15 = L.BatchNormalization(512, use_beta=False, eps=2e-5)
            self.bias15 = L.Bias(shape=(512,))
            self.conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True)
            self.bn16 = L.BatchNormalization(1024, use_beta=False, eps=2e-5)
            self.bias16 = L.Bias(shape=(1024,))
            self.conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True)
            self.bn17 = L.BatchNormalization(512, use_beta=False, eps=2e-5)
            self.bias17 = L.Bias(shape=(512,))
            self.conv18 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True)
            self.bn18 = L.BatchNormalization(1024, use_beta=False, eps=2e-5)
            self.bias18 = L.Bias(shape=(1024,))
            # new layer
            self.conv19 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, nobias=True)
            self.bn19 = L.BatchNormalization(1024, use_beta=False)
            self.bias19 = L.Bias(shape=(1024,))
            self.conv20 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, nobias=True)
            self.bn20 = L.BatchNormalization(1024, use_beta=False)
            self.bias20 = L.Bias(shape=(1024,))
            self.conv21 = L.Convolution2D(3072, 1024, ksize=3, stride=1, pad=1, nobias=True)
            self.bn21 = L.BatchNormalization(1024, use_beta=False)
            self.bias21 = L.Bias(shape=(1024,))
            self.conv22 = L.Convolution2D(
                1024, n_boxes * (5 + n_classes), ksize=1, stride=1, pad=0, nobias=True)
            self.bias22 = L.Bias(shape=(n_boxes * (5 + n_classes),))
        self.n_boxes = n_boxes
        self.n_classes = n_classes

    def __call__(self, x):
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
        high_resolution_feature = reorg.reorg(h)  # 高解像度特徴量をreorgでサイズ落として保存しておく
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h))), slope=0.1)
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h))), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h))), slope=0.1)
        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h))), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h))), slope=0.1)

        # new layer
        h = F.leaky_relu(self.bias19(self.bn19(self.conv19(h))), slope=0.1)
        h = F.leaky_relu(self.bias20(self.bn20(self.conv20(h))), slope=0.1)
        h = F.concat((high_resolution_feature, h), axis=1)  # output concatnation
        h = F.leaky_relu(self.bias21(self.bn21(self.conv21(h))), slope=0.1)
        h = self.bias22(self.conv22(h))

        return h


class YOLOv2Predictor(chainer.Chain):

    def __init__(self, predictor):
        super(YOLOv2Predictor, self).__init__(predictor=predictor)
        self.anchors = [[5.375, 5.03125], [5.40625, 4.6875], [2.96875, 2.53125],
                        [2.59375, 2.78125], [1.9375, 3.25]]
        self.thresh = 0.6
        self.seen = 0
        self.unstable_seen = 5000

    def init_anchor(self, anchors):
        self.anchors = anchors

    def predict(self, input_x, detection_thresh):
        batch_size, input_channel, input_h, input_w = input_x.shape
        output = self.predictor(Variable(input_x))
        *_, grid_h, grid_w = output.shape
        x, y, w, h, conf, prob = F.split_axis(
            F.reshape(output, (batch_size, self.predictor.n_boxes, self.predictor.n_classes + 5,
                               grid_h, grid_w)), (1, 2, 3, 4, 5),
            axis=2)

        # Variable->numpyへの変換
        x = F.sigmoid(x).data  # xのactivation
        y = F.sigmoid(y).data  # yのactivation
        w = w.data
        h = h.data
        conf = F.sigmoid(conf).data  # confのactivation
        prob = F.softmax(prob, axis=2).data  # probablitiyのacitivation

        # x, y, w, hを絶対座標へ変換
        x_shift = np.broadcast_to(np.arange(grid_w, dtype=np.float32), x.shape)
        y_shift = np.broadcast_to(np.arange(grid_h, dtype=np.float32)[:, np.newaxis], y.shape)
        w_anchor = np.broadcast_to(
            np.array(self.anchors, dtype=np.float32)[:, 0][:, np.newaxis, np.newaxis, np.newaxis],
            w.shape)
        h_anchor = np.broadcast_to(
            np.array(self.anchors, dtype=np.float32)[:, 1][:, np.newaxis, np.newaxis, np.newaxis],
            h.shape)
        x = (x + x_shift) / grid_w
        y = (y + y_shift) / grid_h
        w = np.exp(w) * w_anchor / grid_w
        h = np.exp(h) * h_anchor / grid_h

        # 物体があるかの判定用配列を計算
        conf = np.squeeze(conf)
        prob = np.squeeze(np.transpose(prob, (2, 0, 1, 3, 4)))
        detected_indices = (conf * prob).max(axis=0) > detection_thresh

        # (batch_size,:1より大きいときだけ存在) n_box, grid_h, grid_w に整形
        x = np.squeeze(x)[detected_indices]
        y = np.squeeze(y)[detected_indices]
        w = np.squeeze(w)[detected_indices]
        h = np.squeeze(h)[detected_indices]
        conf = conf[detected_indices]
        # batch_size == 1 のときだけ次元が一つ落ちるので動的に変更後の形を決定
        prob = prob.transpose((*tuple((range(1, len(prob.shape)))), 0))[detected_indices]
        return x, y, w, h, conf, prob
