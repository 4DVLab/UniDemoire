from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

import math
import pickle

import numpy as np

from torch.autograd import Function

from torch.autograd import Variable

from scipy import signal

from math import exp, sqrt

class Chroma_Loss(nn.Module):
	def __init__(self):
		super(Chroma_Loss, self).__init__()
		self.L1 = nn.L1Loss()

	def RGB2UV(self, x):
		device = x.device
		b, c, h, w = x.size()

		E = x

		R = E[:, 0, :, :]
		G = E[:, 1, :, :]
		B = E[:, 2, :, :]

		X = 0.4887 * R + 0.3107 * G + 0.2006 * B
		Y = 0.1762 * R + 0.8130 * G + 0.0108 * B
		Z = 0.0001 * R + 0.0102 * G + 0.9898 * B

		A = X + 15 * Y + 3 * Z
		u = torch.zeros((b, h, w)).cuda(device)
		u[A != 0] = 4 * X[A != 0] / A[A != 0]

		v = torch.zeros((b, h, w)).cuda(device)
		v[A != 0] = 9 * Y[A != 0] / A[A != 0]

		u = u * 410 / 255
		v = v * 410 / 255

		return u, v

	def forward(self, out, gt):
		out_u, out_v = self.RGB2UV(out)
		gt_u, gt_v = self.RGB2UV(gt)

		diff_u = self.L1(out_u, gt_u)
		diff_v = self.L1(out_v, gt_v)

		loss = 1 / 2 * (diff_u + diff_v)

		return loss

class Sobel_Grads(nn.Module):
    def __init__(self):
        super(Sobel_Grads, self).__init__()

        c = 3

        gx = np.array([[1., 2. , 1.], [0., 0., 0.], [-1., -2. , -1.]], dtype='float32')
        self.gx = Variable(torch.from_numpy(gx).expand(c, 1, 3, 3).contiguous(), requires_grad = False)

        gy = np.array([[1., 0. , -1.], [2., 0., -2.], [1., 0. , -1.]], dtype='float32')
        self.gy = Variable(torch.from_numpy(gy).expand(c, 1, 3, 3).contiguous(), requires_grad=False)

    def forward(self, inputs):
        device = inputs.device

        self.gx = self.gx.to(device)
        self.gy = self.gy.to(device)

        c = 3
        Gx = F.conv2d(inputs, self.gx, padding = 1, groups = c)
        Gy = F.conv2d(inputs, self.gy, padding = 1, groups = c)

        magnitude = torch.sqrt(Gx * Gx + Gy * Gy)

        return magnitude

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
	return gauss / gauss.sum()


def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
	mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
	mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
	def __init__(self, window_size=11, size_average=True):
		super(SSIM, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)

			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)

			self.window = window
			self.channel = channel

		return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
	(_, channel, _, _) = img1.size()
	window = create_window(window_size, channel)

	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)

	return _ssim(img1, img2, window, window_size, channel, size_average)

class L2_SSIM_Sobel(nn.Module):
	def __init__(self):
		super(L2_SSIM_Sobel, self).__init__()
		self.L2 = nn.MSELoss()
		self.ssim = SSIM()
		self.sobel = Sobel_Grads()

	def forward(self, out, gt):
		loss_L2 = self.L2(out, gt)
		loss_ssim = self.ssim(out, gt)

		s_out = self.sobel(out)
		s_gt = self.sobel(gt)
		loss_sobel = self.L2(s_out, s_gt)

		loss = loss_L2 - 0.1 * loss_ssim + 0.1 * loss_sobel

		return loss

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error)
        return loss

class L1_Charbonnier_SSIM(nn.Module):
	def __init__(self):
		super(L1_Charbonnier_SSIM, self).__init__()
		self.L1C = L1_Charbonnier_loss()
		self.ssim = SSIM()

	def forward(self, out, gt):
		loss_L1 = self.L1C(out, gt)
		loss_ssim = self.ssim(out, gt)

		loss = loss_L1 - 0.1 * loss_ssim

		return loss

class WaveletTransform(nn.Module):
	def __init__(self, scale=1, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True):
		super(WaveletTransform, self).__init__()

		self.scale = scale
		self.dec = dec
		self.transpose = transpose

		ks = int(math.pow(2, self.scale))
		nc = 3 * ks * ks

		if dec:
			self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3,
								  bias=False)
		else:
			self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0,
										   groups=3, bias=False)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				f = open(params_path, 'rb')
				dct = pickle.load(f)
				f.close()
				m.weight.data = torch.from_numpy(dct['rec%d' % ks])
				m.weight.requires_grad = False

	def forward(self, x):
		if self.dec:
			output = self.conv(x)
			if self.transpose:
				osz = output.size()
				# print(osz)
				output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
		else:
			if self.transpose:
				xx = x
				xsz = xx.size()
				xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
			output = self.conv(xx)
		return output

class L1_Charbonnier_Wavelet(nn.Module):
	def __init__(self):
		super(L1_Charbonnier_Wavelet, self).__init__()
		self.L1C = L1_Charbonnier_loss()
		self.wavelet = WaveletTransform(dec = True)

	def forward(self, out, gt):
		loss_L1 = self.L1C(out, gt)

		wavelet_out = self.wavelet(out)
		wavelet_gt = self.wavelet(gt)
		loss_wavelet = self.L1C(wavelet_out, wavelet_gt)

		loss = loss_L1 + loss_wavelet

		return loss

class L1_DCT(nn.Module):
	def __init__(self):
		super(L1_DCT, self).__init__()
		self.L1 = nn.L1Loss()
		self.L2 = nn.MSELoss()

	def forward(self, out, gt):
		loss_L1 = self.L1(out, gt)

		dct_out = dct(out)
		dct_gt = dct(gt)
		loss_dct = self.L2(dct_out, dct_gt)

		loss = loss_L1 + loss_dct

		return loss

def Gradient(I):
	G_vertical = I[:, :, :, :-1] - I[:, :, :, 1:]
	G_horizontal = I[:, :, :-1, :] - I[:, :, 1:, :]

	return G_vertical, G_horizontal

class L1_DCT_Grad(nn.Module):
	def __init__(self):
		super(L1_DCT_Grad, self).__init__()
		self.L1 = nn.L1Loss()
		self.L2 = nn.MSELoss()

	def forward(self, out, gt):
		loss_L1 = self.L1(out, gt)

		dct_out = dct(out)
		dct_gt = dct(gt)
		loss_dct = self.L2(dct_out, dct_gt)

		gradx_out, grady_out = Gradient(out)
		gradx_gt, grady_gt = Gradient(gt)
		loss_grad = self.L1(gradx_out, gradx_gt) + self.L1(grady_out, grady_gt)

		loss = loss_L1 + loss_dct + loss_grad

		return loss

class L1_UV(nn.Module):
	def __init__(self):
		self.L1 = nn.L1Loss()
		self.LUV = Chroma_Loss()

	def forward(self, out, gt):
		loss_L1 = self.L1(out, gt)

		loss_UV = self.LUV(out, gt)

		loss = loss_L1 + 0.1 * loss_UV

		return loss

class Advance_Sobel_Grads(nn.Module):
	def __init__(self):
		super(Advance_Sobel_Grads, self).__init__()

		c = 3
		gx = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype='float32')
		self.gx = Variable(torch.from_numpy(gx).expand(c, 1, 3, 3).contiguous(), requires_grad=False)

		gy = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], dtype='float32')
		self.gy = Variable(torch.from_numpy(gy).expand(c, 1, 3, 3).contiguous(), requires_grad=False)

		gu = np.array([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]], dtype='float32')
		self.gu = Variable(torch.from_numpy(gu).expand(c, 1, 3, 3).contiguous(), requires_grad=False)

		gv = np.array([[2., 1., 0.], [1., 0., -1.], [0., -1., -2.]], dtype='float32')
		self.gv = Variable(torch.from_numpy(gv).expand(c, 1, 3, 3).contiguous(), requires_grad=False)

	def forward(self, inputs):
		device = inputs.device

		self.gx = self.gx.to(device)
		self.gy = self.gy.to(device)
		self.gu = self.gu.to(device)
		self.gv = self.gv.to(device)

		c = 3
		Gx = F.conv2d(inputs, self.gx, padding=1, groups=c)
		Gy = F.conv2d(inputs, self.gy, padding=1, groups=c)
		Gu = F.conv2d(inputs, self.gu, padding=1, groups=c)
		Gv = F.conv2d(inputs, self.gv, padding=1, groups=c)

		outputs = torch.cat((Gx, Gy, Gu, Gv), dim = 1)

		return outputs

class L1_ASL(nn.Module):
	def __init__(self):
		super(L1_ASL, self).__init__()
		self.L1 = nn.L1Loss()
		self.ASL = Advance_Sobel_Grads()

	def forward(self, out, gt):
		loss_L1 = self.L1(out, gt)

		sobel_out = self.ASL(out)
		sobel_gt = self.ASL(gt)
		loss_ASL = self.L1(sobel_out, sobel_gt)

		loss = loss_L1 + 0.25 * loss_ASL

		return loss

eps = 0.000001

def polarFFT(input):
	x_f = torch.rfft(input, 2, onesided=False)
	x_f_r = x_f[:, :, :, :, 0].contiguous()
	x_f_i = x_f[:, :, :, :, 1].contiguous()

	x_f_r[x_f_r == 0] = eps
	x_mag = torch.sqrt(x_f_r * x_f_r + x_f_i * x_f_i)
	x_pha = torch.atan2(x_f_i, x_f_r)

	return x_mag, x_pha

class FFT_loss(nn.Module):
	def __init__(self):
		super(FFT_loss, self).__init__()
		self.criterion = nn.MSELoss()

	def forward(self, output, gt):
		out_FFT = polarFFT(output)[0]
		gt_FFT = polarFFT(gt)[0]

		loss = self.criterion(out_FFT, gt_FFT)

		return loss

class FFT_plus_L1_ASL(nn.Module):
	def __init__(self):
		super(FFT_plus_L1_ASL, self).__init__()
		self.FFT = FFT_loss()
		self.L1ASL = L1_ASL()

	def forward(self, output, gt):
		loss_0 = self.FFT(output, gt)
		loss_1 = self.L1ASL(output, gt)

		loss =  0.001 * loss_0 + loss_1

		return loss
