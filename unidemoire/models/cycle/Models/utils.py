import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

import torch.nn as nn
from torchvision.utils import save_image
from math import log10, exp, sqrt, cos, pi
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

## DCT Transform
class DCT(nn.Module):
    def __init__(self):
        super(DCT, self).__init__()

        conv_shape = (1, 1, 64, 64)
        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0/8)
        r2 = sqrt(2.0/8)
        for i in range(8):
            _u = 2*i+1
            for j in range(8):
                _v = 2*j+1
                index = i*8+j
                for u in range(8):
                    for v in range(8):
                        index2 = u*8+v
                        t = cos(_u*u*pi/16)*cos(_v*v*pi/16)
                        t = t*r1 if u==0 else t*r2
                        t = t*r1 if v==0 else t*r2
                        kernel[0,0,index2,index] = t

        self.kernel = torch.tensor(kernel, requires_grad = False, dtype=torch.float32)

    def forward(self, inputs):

        device = inputs.device
        kernel = self.kernel.to(device)
        k = kernel.permute(3, 1, 2, 0)
        k = torch.reshape(k, (64, 1, 8, 8))

        b, c, h, w = inputs.size()
        scale_r = h // 8
        new_inputs = torch.reshape(inputs, (b, c * scale_r * scale_r, 8, 8))

        outputs = torch.zeros_like(new_inputs)

        num_of_p = c * scale_r * scale_r

        for i in range(num_of_p):
            patch = new_inputs[:, i, :, :]
            patch = patch.unsqueeze(dim=1)
            patch = patch.to(device).float()

            new_patch = F.conv2d(patch, k, stride = 8)
            new_patch = torch.reshape(new_patch, (b, 1, 8, 8)).squeeze(dim = 1)

            outputs[:, i, :, :] = new_patch

        outputs = torch.reshape(outputs, (b, c, h, w))

        return outputs

class Local_DCT(nn.Module):
    def __init__(self):
        super(Local_DCT, self).__init__()

        conv_shape = (1, 1, 64, 64)
        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0 / 8)
        r2 = sqrt(2.0 / 8)
        for i in range(8):
            _u = 2 * i + 1
            for j in range(8):
                _v = 2 * j + 1
                index = i * 8 + j
                for u in range(8):
                    for v in range(8):
                        index2 = u * 8 + v
                        t = cos(_u * u * pi / 16) * cos(_v * v * pi / 16)
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[0, 0, index2, index] = t

        self.kernel = torch.tensor(kernel, requires_grad=False, dtype=torch.float32)

    def forward(self, inputs):

        device = inputs.device
        kernel = self.kernel.to(device)
        k = kernel.permute(3, 1, 2, 0)
        k = torch.reshape(k, (64, 1, 8, 8))

        b, c, h, w = inputs.size()
        scale_r = h // 8
        new_inputs = torch.reshape(inputs, (b, c * scale_r * scale_r, 8, 8))

        outputs = torch.zeros_like(new_inputs)

        num_of_p = c * scale_r * scale_r

        for i in range(num_of_p):
            patch = new_inputs[:, i, :, :]
            patch = patch.unsqueeze(dim=1)
            patch = patch.to(device).float()

            new_patch = F.conv2d(patch, k, stride=8)
            new_patch = torch.reshape(new_patch, (b, 1, 8, 8)).squeeze(dim=1)

            outputs[:, i, :, :] = new_patch

        outputs = torch.reshape(outputs, (b, c, h, w))

        return outputs

class Inverse_DCT(nn.Module):
    def __init__(self):
        super(Inverse_DCT, self).__init__()

        conv_shape = (1, 1, 64, 64)
        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0/8)
        r2 = sqrt(2.0/8)
        for i in range(8):
            _u = 2*i+1
            for j in range(8):
                _v = 2*j+1
                index = i*8+j
                for u in range(8):
                    for v in range(8):
                        index2 = u*8+v
                        t = cos(_u*u*pi/16)*cos(_v*v*pi/16)
                        t = t*r1 if u==0 else t*r2
                        t = t*r1 if v==0 else t*r2
                        kernel[0,0,index2,index] = t

        self.kernel = torch.tensor(kernel, requires_grad = False, dtype=torch.float32)

        self.kernel = self.kernel.permute(0, 1, 3, 2)

    def forward(self, inputs):

        device = inputs.device
        kernel = self.kernel.to(device)
        k = kernel.permute(3, 1, 2, 0)
        k = torch.reshape(k, (64, 1, 8, 8))

        b, c, h, w = inputs.size()
        scale_r = h // 8
        new_inputs = torch.reshape(inputs, (b, c * scale_r * scale_r, 8, 8))

        outputs = torch.zeros_like(new_inputs)

        num_of_p = c * scale_r * scale_r

        for i in range(num_of_p):
            patch = new_inputs[:, i, :, :]
            patch = patch.unsqueeze(dim=1)
            patch = patch.to(device).float()

            new_patch = F.conv2d(patch, k, stride = 8)
            new_patch = torch.reshape(new_patch, (b, 1, 8, 8)).squeeze(dim = 1)

            outputs[:, i, :, :] = new_patch

        outputs = torch.reshape(outputs, (b, c, h, w))

        return outputs.clamp(min = 0, max = 1)

## block wise mapping
def block_wise_mapping(net, input, input_size, pad):
    b, c, _, _ = input.size()
    window = create_window(pad, b, c, pad // 2)

    pad_in = padarray(input, pad)

    pad_out = torch.zeros_like(pad_in)
    pnorm = torch.zeros_like(pad_in)

    device = input.device

    i = 0
    j = 0

    stride = pad // 2

    _,_, height, width = pad_in.size()

    while(i < height - input_size + 1):
        while(j < width - input_size + 1):
            patch = pad_in[:,:,i : i + input_size, j : j + input_size]
            patch = patch.to(device).float()

            pout = net(patch)

            if i < height - input_size and j < width - input_size:
                pout = pout[:,:,0 : 0 + pad, 0 : 0 + pad]

                mask = window.to(device)
                p_after = pout * mask

                pad_out[:,:,i : i + pad, j : j + pad] = pad_out[:,:,i : i + pad, j : j + pad] + p_after
                pnorm[:,:,i : i + pad, j : j + pad] = pnorm[:,:,i : i + pad, j : j + pad] + mask
            else:
                pad_out[:, :, i : i + input_size, j : j + input_size] = pad_out[:, :, i : i + input_size, j : j + input_size] + pout
                pnorm[:, :, i : i + input_size, j : j + input_size] = pnorm[:, :, i : i + input_size, j : j + input_size] + 1.0

            j = j + stride

        i = i + stride
        j = 0

    output = pad_out[:,:,0 : 1024, 0 : 1024] / pnorm[:,:,0 : 1024, 0 : 1024]

    return output

def create_window(window_size, batch, channel, sigma):
	_1D_window = gaussian(window_size, sigma).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = _2D_window.expand(batch, channel, window_size, window_size)
	return window

def padarray(input, size_pad):
    b,c,h,w = input.size()
    device = input.device

    new_h = h + size_pad
    new_w = w + size_pad
    output = torch.zeros((b, c, new_h, new_w)).to(device)

    output[:,:,0 : h, 0 : w] = input[:,:,:,:]
    # output[:,:,h : new_h, w : new_w] = 0.0

    return output

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
	return gauss / gauss.sum()