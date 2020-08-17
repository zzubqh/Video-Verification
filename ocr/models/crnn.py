#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torch ocr model
@author: chineseocr
"""
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
from collections import OrderedDict
from torch.autograd import Variable


def img_normalization(img, img_h=32):
    scale = img.size[1] * 1.0 / img_h
    w = img.size[0] / scale
    w = int(w)
    img = img.resize((w, img_h), Image.BILINEAR)
    img = (np.array(img) / 255.0 - 0.5) / 0.5
    return img


def converter_str2labe(res, alphabet):
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
    return ''.join(raw)


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, img_h, nc, nclass, nh, leakyRelu=False, alphabet=None):
        super(CRNN, self).__init__()
        assert img_h % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.alphabet = alphabet
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = self.rnn(conv)
        T, b, h = output.size()
        output = output.view(T, b, -1)
        return output

    def load_weights(self, path):
        trainWeights = torch.load(path, map_location=lambda storage, loc: storage)
        modelWeights = OrderedDict()
        for k, v in trainWeights.items():
            name = k.replace('module.', '')  # remove `module.`
            modelWeights[name] = v
        self.load_state_dict(modelWeights)
        if torch.cuda.is_available():
            self.cuda()
        self.eval()

    def predict(self, image):
        """
        :param image: PIL.Image格式并转换成灰度图，
        :return:
        """
        image = img_normalization(image, 32)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
        else:
            image = image.cpu()

        image = image.view(1, 1, *image.size())
        image = Variable(image)
        if image.size()[-1] < 8:
            return ''
        preds = self(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        raw = converter_str2labe(preds, self.alphabet)
        return raw
