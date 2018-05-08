#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import OrderedDict

import cv2
from skimage import io
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim
import os
from skimage import io
import pandas as pd
import numpy as np
from skimage import img_as_float
from tqdm import tqdm


class Net(nn.Module):

    def __init__(self, num_of_classes):
        super(Net, self).__init__()
        # input image channel, output channels, kernel size square convolution
        # kernel
        # input size = 28
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # input size = 14
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # input size = 16
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.vp = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_of_classes)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.bn1(self.vp(self.conv1(x))))
        x = F.relu(self.bn2(self.vp(self.conv3(self.conv2(x)))))
        x = self.drop2D(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


net = Net(num_of_classes=33)
net = net.double()


class PropagationBase(object):

    def __init__(self, model, cuda=False):
        self.model = model
        self.model.eval()
        if cuda:
            self.model.cuda()
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()
        self._set_hook_func()
        self.image = None

    def _set_hook_func(self):
        raise NotImplementedError

    def _encode_one_hot(self, idx):
        one_hot = torch.DoubleTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    def forward(self, image):
        self.image = image
        self.preds = self.model.forward(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.data.sort(0, True)
        return self.prob, self.idx

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


class GradCAM(PropagationBase):

    def _set_hook_func(self):

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        # return grads / l2_norm.data[0]
        return grads / l2_norm.data[0]

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        self.map_size = grads.size()[2:]
        return nn.AvgPool2d(self.map_size)(grads)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = torch.DoubleTensor(self.map_size).zero_()
        for fmap, weight in zip(fmaps[0], weights[0]):
            res = fmap * weight.data.expand_as(fmap)
            gcam += fmap * weight.data.expand_as(fmap)
        gcam = F.relu(Variable(gcam))

        gcam = gcam.data.cpu().numpy()
        gcam -= gcam.min()
        if gcam.max() != 0:
            gcam /= gcam.max()
        gcam = cv2.resize(gcam, (self.image.size(3), self.image.size(2)))

        return gcam

    def save(self, filename, gcam, raw_image):
        gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
        # gcam = gcam.astype(np.float) + raw_image.astype(np.float)
        if gcam.max() != 0:
            gcam = gcam / gcam.max() * 255.0
        cv2.imwrite(filename, np.uint8(gcam))


class BackPropagation(PropagationBase):
    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def generate(self):
        output = self.image.grad.data[0].numpy()[0]
        return output

    def save(self, filename, data):
        abs_max = np.maximum(-1 * data.min(), data.max())
        data = data / abs_max * 127.0 + 127.0
        cv2.imwrite(filename, np.uint8(data))


class GuidedBackPropagation(BackPropagation):

    def _set_hook_func(self):

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_in[0]

            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return torch.clamp(grad_in[0], min=0.0)

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
