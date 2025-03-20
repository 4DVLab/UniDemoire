import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from .common import *
from torchvision import models as tv
from torch.nn.parameter import Parameter

from unidemoire.models.mbcnn.LossNet import L1_LOSS, L1_Advanced_Sobel_Loss
import scipy.stats as st


class PoissonGradientLoss(nn.Module):
    def __init__(self):
        super(PoissonGradientLoss, self).__init__()
        
        ## Define the Laplacian kernel
        self.laplace_kernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        laplace_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.laplace_kernel.weight = nn.Parameter(laplace_filter)

        for param in self.laplace_kernel.parameters():
            param.requires_grad = False
            
        self.L2_Loss = torch.nn.MSELoss()   
        self.L1_Loss = torch.nn.L1Loss() 
    
    def norm(self, tensor):
        # [0, 1] → [-1, 1]
        return (tensor + 1.0) / 2.0
 
    def forward(self, fusion_result, natural, moire_pattern, device='cuda'):
        self.laplace_kernel.to(device)
        fusion_result = self.norm(fusion_result)
        natural       = self.norm(natural)
        moire_pattern = self.norm(moire_pattern)
        
        b, c, h, w = fusion_result.shape
        # x_channel_list = [x[:,i,:,:] for i in range(c)]
        fusion_result_channel_list = [fusion_result[:,i,:,:].view(b, 1, h, w) for i in range(c)]
        natural_channel_list       = [natural[:,i,:,:].view(b, 1, h, w)       for i in range(c)]
        moire_pattern_channel_list = [moire_pattern[:,i,:,:].view(b, 1, h, w) for i in range(c)]

        loss = 0.0
        for i in range(c):
            # ΔI_result, ΔI_n, ΔI_mp
            delta_fusion_result = self.laplace_kernel(fusion_result_channel_list[i])
            delta_natural       = self.laplace_kernel(natural_channel_list[i])
            delta_moire_pattern = self.laplace_kernel(moire_pattern_channel_list[i])
            # L = Σ_c |ΔI_result - (ΔI_n+ΔI_mp)|^2
            # loss += self.L2_Loss(delta_fusion_result, (delta_natural + delta_moire_pattern))
            loss += self.L1_Loss(delta_fusion_result, (delta_natural + delta_moire_pattern))
        return loss 


class PMTNet_Loss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(PMTNet_Loss, self).__init__()
        self.L_c = CharbonnierLoss()
        self.ASL = L1_Advanced_Sobel_Loss_Noreduce(device=device)
        self.C_r = ColorLoss()
        self.blur_rgb = Blur(3)
    
    def forward(self, output1, output2, output3, clear1, cur_lr):
        clear2 = F.interpolate(clear1, scale_factor=0.5, mode='bilinear', align_corners=False)
        clear3 = F.interpolate(clear1, scale_factor=0.25, mode='bilinear', align_corners=False)

        lambda_1 = 1
        lambda_2 = 0.3        
        if cur_lr > 1e-5:
            lambda_3 = 0.18
            eta = 1
        else:
            lambda_3 = 0
            eta = 2            
        
        L_s1 = lambda_1*self.L_c(output1, clear1) + lambda_2*self.ASL(output1, clear1) + lambda_3*self.C_r(self.blur_rgb(output1), self.blur_rgb(clear1))
        L_s2 = lambda_1*self.L_c(output2, clear2) + lambda_2*self.ASL(output2, clear2) + lambda_3*self.C_r(self.blur_rgb(output2), self.blur_rgb(clear2))
        L_s3 = lambda_1*self.L_c(output3, clear3) + lambda_2*self.ASL(output3, clear3) + lambda_3*self.C_r(self.blur_rgb(output3), self.blur_rgb(clear3))

        loss = eta*L_s1 + L_s2 + L_s3
        loss = loss.sum() / clear1.shape[0]
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss

class L1_Advanced_Sobel_Loss(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.conv_op_x = nn.Conv2d(3,3, 3, bias=False)
        self.conv_op_y = nn.Conv2d(3,3, 3, bias=False)

        sobel_kernel_x = np.array([[[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]]], dtype='float32')
        sobel_kernel_y = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    # def forward(self, edge_outputs, image_target):
    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = torch.abs(edge_Y_xoutputs) + torch.abs(edge_Y_youtputs)

        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)

        diff = torch.add(edge_Youtputs, -edge_Y)
        error = torch.abs(diff)
        loss = torch.sum(error) / outputs.size(0)
        return loss

class L1_Advanced_Sobel_Loss_Noreduce(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super().__init__()
        self.device = device
        self.conv_op_x = nn.Conv2d(3,3, 3, bias=False)
        self.conv_op_y = nn.Conv2d(3,3, 3, bias=False)

        sobel_kernel_x = np.array([[[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]],
                                   [[2, 1, 0], [1, 0, -1], [0,-1, -2]]], dtype='float32')
        sobel_kernel_y = np.array([[[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
                                   [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 3, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 3, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x).to(device)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y).to(device)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    # def forward(self, edge_outputs, image_target):
    def forward(self, outputs, image_target):
        edge_Y_xoutputs = self.conv_op_x(outputs)
        edge_Y_youtputs = self.conv_op_y(outputs)
        edge_Youtputs = torch.abs(edge_Y_xoutputs) + torch.abs(edge_Y_youtputs)

        edge_Y_x = self.conv_op_x(image_target)
        edge_Y_y = self.conv_op_y(image_target)
        edge_Y = torch.abs(edge_Y_x) + torch.abs(edge_Y_y)

        diff = torch.add(edge_Youtputs, -edge_Y)
        error = torch.abs(diff)
        loss = torch.sum(torch.sum(torch.sum(error, dim=-1), dim=-1), dim=-1)
        loss = loss / image_target.shape[1] / image_target.shape[2] / image_target.shape[3]
        # loss = torch.sum(error) / outputs.size(0)
        return loss


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel * 0.053
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter


class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        self.nc = nc
        self.kernlen = 30
        kernel = gauss_kernel(kernlen=self.kernlen, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))
        x = F.conv2d(x, self.weight, stride=1, padding=self.kernlen//2, groups=self.nc)
        return x


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x1, x2):
        return torch.sum(torch.pow((x1 - x2), 2)).div(2 * x1.size()[0])


class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(CosineSimilarityLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim, eps)
        
    def forward(self, x, y):
        x_vector = x.view(1,-1).contiguous()
        y_vector = y.view(1,-1).contiguous()
        loss = 1.0 - self.cos(x_vector, y_vector)
        return loss

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        features = input.view(a * b, c * d).contiguous()  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def forward(self, input, target):
        G_target = self.gram_matrix(target).detach()
        G_input  = self.gram_matrix(input).detach()
        loss = F.mse_loss(G_input, G_target)
        return loss


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class MBCNN_Loss(torch.nn.Module):
    def __init__(self):
        super(MBCNN_Loss, self).__init__()
        self.criterion_l1 = L1_LOSS()
        self.criterion_advanced_sobel_l1 = L1_Advanced_Sobel_Loss()
    
    def forward(self, output3, output2, output1, clear1):
        clear2 = F.interpolate(clear1, scale_factor=0.5, mode='bilinear', align_corners=False)
        clear3 = F.interpolate(clear1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        Loss_l1 = self.criterion_l1(output1, clear1)
        Loss_advanced_sobel_l1 = self.criterion_advanced_sobel_l1(output1, clear1)
        Loss_l12 = self.criterion_l1(output2, clear2)
        Loss_advanced_sobel_l12 = self.criterion_advanced_sobel_l1(output2, clear2)
        Loss_l13 = self.criterion_l1(output3, clear3)
        Loss_advanced_sobel_l13 = self.criterion_advanced_sobel_l1(output3, clear3)

        Loss1 = Loss_l1 + (0.25) * Loss_advanced_sobel_l1
        Loss2 = Loss_l12 + (0.25) * Loss_advanced_sobel_l12
        Loss3 = Loss_l13 + (0.25) * Loss_advanced_sobel_l13

        loss = Loss1 + Loss2 + Loss3
        return loss

class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p
    def forward(self, out1, out2, out3, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam*F.l1_loss(out1, gt1)
        loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam*F.l1_loss(out2, gt2)
        loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam*F.l1_loss(out3, gt3)
        
        return loss1+loss2+loss3            

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    
class PerceptualLoss(torch.nn.Module):
    def __init__(self, feature_layers=[], style_layers=[0, 1, 2, 3], resize=True):
        super(PerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks    = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.resize = resize

    def forward(self, input, target, device, feature_layers=[], style_layers=[]):
        # device = input.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            block = block.to(device)
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                (bs, ch, h, w) = x.size()
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1) / (ch * h * w)
                gram_y = act_y @ act_y.permute(0, 2, 1) / (ch * h * w)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class TVLoss(nn.Module):
    def __init__(self, weight: float=1) -> None:
        """Total Variation Loss

        Args:
            weight (float): weight of TV loss
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
        tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)


class ColorHistogramMatchingLoss(nn.Module):
    def __init__(self, d_hist=64, hist_boundary=[-3, 3], fall_off=0.02, eps=1e-6, device='cuda'):
        super(ColorHistogramMatchingLoss, self).__init__()
        self.d_hist = d_hist
        self.hist_boundary = hist_boundary
        self.fall_off = fall_off
        self.eps = eps
        self.device = device
    
    def forward(self, x, y, device='cuda'):
        self.device = device
        x_hist = self.get_histogram_feature(x)
        y_hist = self.get_histogram_feature(y)

        y_sqred = torch.sqrt(y_hist)
        x_sqred = torch.sqrt(x_hist)
        
        diff = y_sqred - x_sqred
        h = torch.sum(torch.square(diff), dim=(1,2,3))

        h_norm = torch.sqrt(h)
        h_norm = h_norm * (torch.sqrt(torch.ones((y.shape[0]))/2)).to(self.device)
        
        # Used mean reduction, other option is sum reduction
        h_norm = torch.mean(h_norm)
        
        return h_norm

    def get_hist_c(self, ur, u, vr, v, i_y):
        rdiffu = torch.abs(ur - u)
        rdiffv = torch.abs(vr - v)

        # So far for k Eq. 3, 
        # Inner absolute values for each pixel log-chrominance value with each bin of H(u,v,c)
        # k has two parts multiplied thus implemented as k = k_u x k_v
        rdiffu = 1 / (1 + torch.square(rdiffu/self.fall_off))
        rk_v = 1 / (1 + torch.square(rdiffv/self.fall_off))
        i_y = torch.unsqueeze(i_y, dim=2)

        rdiffu = rdiffu*i_y
        rdiffu = rdiffu.transpose(1, 2)
        rdiffu = torch.bmm(rdiffu, rk_v)  # Compute intensity weighted impact of chrominance values

        return rdiffu

    
    def get_histogram_feature(self, img):
        img = img + self.eps
        # reshape such that img_flat = (batchsize, 3, W*H)
        img_flat = torch.reshape(img, (img.shape[0], img.shape[1], -1))  

        # Pixel intesities I_y at Eq. 2
        i_y = torch.sqrt(torch.square(img_flat[:, 0]) + torch.square(img_flat[:, 1]) + torch.square(img_flat[:, 2]))
        
        # log_R(img), log_G(img), log_B(img)
        log_r = torch.log(img_flat[:, 0])
        log_g = torch.log(img_flat[:, 1])
        log_b = torch.log(img_flat[:, 2])
        
        # u,v parameters for each channel
        # each channel normalization values with respect to other two channels
        ur = log_r - log_g
        vr = log_r - log_b
        ug = -ur
        vg = -ur + vr
        ub = -vr
        vb = -vr + ur        

        u = torch.linspace(self.hist_boundary[0], self.hist_boundary[1], self.d_hist).to(self.device)
        u = torch.unsqueeze(u, dim=0)   # Make (h,) to (1, h) so that 
                                        # for each element in ur there will 
                                        # be difference with each u value.
        v = torch.linspace(self.hist_boundary[0], self.hist_boundary[1], self.d_hist).to(self.device)
        v = torch.unsqueeze(v, dim=0)

        ur = torch.unsqueeze(ur, dim=2) # Make each element an array it 
                                        # Difference will be [N*N, 1] - [1,h] = [N*N, h]
                                        # See broadcasting for further
        ug = torch.unsqueeze(ug, dim=2) 
        ub = torch.unsqueeze(ub, dim=2)
        vr = torch.unsqueeze(vr, dim=2) 
        vg = torch.unsqueeze(vg, dim=2) 
        vb = torch.unsqueeze(vb, dim=2) 
        
        hist_r = self.get_hist_c(ur, u, vr, v, i_y)
        hist_g = self.get_hist_c(ug, u, vg, v, i_y)
        hist_b = self.get_hist_c(ub, u, vb, v, i_y)

        # For each channel of H(u,v,c) = H(u,v,R), H(u,v,G), H(u,v,B), k values are computed above
        histogram = torch.stack([hist_r, hist_g, hist_b], dim=1)
        
        # Normalize histogram such that sum of H(u,v,c) of an image is 1
        sum_of_uvc = torch.sum(torch.sum(torch.sum(histogram, dim=3), dim=2), dim=1)
        sum_of_uvc = torch.reshape(sum_of_uvc, (-1, 1, 1, 1))
        histogram = histogram / sum_of_uvc
        return histogram