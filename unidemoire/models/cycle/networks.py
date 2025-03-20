import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from .Models.models import GM2_UNet5_256, GM2_UNet5_128, GM2_UNet5_64, TMB, Discriminator
from .Models.Loss_func_demoire import L1_ASL

class CycleModel(nn.Module):
    def __init__(self,  device='cuda:0'):
        super(CycleModel, self).__init__()
        
        # Initialize generator and discriminator
        self.Net_Demoire_256 = GM2_UNet5_256(6, 3)
        self.Net_Demoire_128 = GM2_UNet5_128(6, 3)
        self.Net_Demoire_64 = GM2_UNet5_64(3, 3)
        self.Net_Demoire_TMB = TMB(256, 1)

        # 256 size
        self.G_Artifact_256_2 = GM2_UNet5_256(6, 3)
        self.D_Moire_256 = Discriminator(6, 256, 256)
        self.D_Clear_256 = Discriminator(6, 256, 256)

        # 128 size
        self.G_Artifact_128_2 = GM2_UNet5_128(6, 3)
        self.D_Moire_128 = Discriminator(6, 128, 128)
        self.D_Clear_128 = Discriminator(6, 128, 128)

        # 64 size
        self.G_Artifact_64_1 = TMB(256, 1)
        self.G_Artifact_64_2 = GM2_UNet5_64(6, 3)
        self.D_Moire_64 = Discriminator(6, 64, 64)
        self.D_Clear_64 = Discriminator(6, 64, 64)

        self.is_device_set = False

        if device != 'cpu':
            self.set_device(device)
        
        self.init_models_weights()

        self.downx2 = nn.UpsamplingNearest2d(scale_factor = 0.5)
        self.upx2   = nn.UpsamplingNearest2d(scale_factor = 2)

        self.Tensor = torch.cuda.FloatTensor

        # LOSS FUNCTIONS
        # self.criterion_GAN = torch.nn.MSELoss()
        # self.criterion_cycle = torch.nn.L1Loss()
        # self.criterion_MSE = torch.nn.MSELoss()
        # self.criterion_content = L1_ASL()
        # self.Loss = L1_ASL()


    def set_device(self, device):
        self.device = device
        self.Net_Demoire_256 = self.Net_Demoire_256.to(self.device)
        self.Net_Demoire_128 = self.Net_Demoire_128.to(self.device)
        self.Net_Demoire_64 = self.Net_Demoire_64.to(self.device)
        self.Net_Demoire_TMB = self.Net_Demoire_TMB.to(self.device)

        # 256 size
        self.G_Artifact_256_2 = self.G_Artifact_256_2.to(self.device)
        self.D_Moire_256 = self.D_Moire_256.to(self.device)
        self.D_Clear_256 = self.D_Clear_256.to(self.device)

        # 128 size
        self.G_Artifact_128_2 = self.G_Artifact_128_2.to(self.device)
        self.D_Moire_128 = self.D_Moire_128.to(self.device)
        self.D_Clear_128 = self.D_Clear_128.to(self.device)

        # 64 size
        self.G_Artifact_64_1 = self.G_Artifact_64_1.to(self.device)
        self.G_Artifact_64_2 = self.G_Artifact_64_2.to(self.device)
        self.D_Moire_64 = self.D_Moire_64.to(self.device)
        self.D_Clear_64 = self.D_Clear_64.to(self.device)
        
        self.is_device_set = True


    def init_models_weights(self):
        # Initialize weights
        self.Net_Demoire_256.apply(self.weights_init)
        self.Net_Demoire_128.apply(self.weights_init)
        self.Net_Demoire_64.apply(self.weights_init)
        self.Net_Demoire_TMB.apply(self.weights_init)

        # 256 size
        self.G_Artifact_256_2.apply(self.weights_init)
        self.D_Moire_256.apply(self.weights_init)
        self.D_Clear_256.apply(self.weights_init)

        # 128 size
        self.G_Artifact_128_2.apply(self.weights_init)
        self.D_Moire_128.apply(self.weights_init)
        self.D_Clear_128.apply(self.weights_init)

        # 64 size
        self.G_Artifact_64_1.apply(self.weights_init)
        self.G_Artifact_64_2.apply(self.weights_init)
        self.D_Moire_64.apply(self.weights_init)
        self.D_Clear_64.apply(self.weights_init)

    # Custom weights initialization called on network
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, MOIRE, CLEAR, device, res):
        # load data
        MOIRE_256 = MOIRE
        MOIRE_128 = self.downx2(MOIRE_256)
        MOIRE_64  = self.downx2(MOIRE_128)

        CLEAR_256 = CLEAR
        CLEAR_128 = self.downx2(CLEAR_256)
        CLEAR_64  = self.downx2(CLEAR_128)
        
        if res == 64:
            return self.forward_64(MOIRE_64, CLEAR_64, historgram, device)


    def forward_64(self, MOIRE_64, CLEAR_64, historgram, device):
        historgram = historgram.float()

        # valid_256 = Variable(Tensor(MOIRE_256.size(0), 1, 1, 1).fill_(1.0).to(device), requires_grad=False)
        # fake_256  = Variable(Tensor(MOIRE_256.size(0), 1, 1, 1).fill_(0.0).to(device), requires_grad=False)

        # valid_128 = Variable(Tensor(MOIRE_128.size(0), 1, 1, 1).fill_(1.0).to(device), requires_grad=False)
        # fake_128  = Variable(Tensor(MOIRE_128.size(0), 1, 1, 1).fill_(0.0).to(device), requires_grad=False)

        valid_64 = Variable(self.Tensor(MOIRE_64.size(0), 1, 1, 1).fill_(1.0).to(device), requires_grad=False)
        # fake_64  = Variable(Tensor(MOIRE_64.size(0), 1, 1, 1).fill_(0.0).to(device), requires_grad=False)  


        # ------------------
        #  Model Forward in 64 size
        # ------------------    
        self.G_Artifact_64_1.train()
        self.G_Artifact_64_2.train()
        self.Net_Demoire_64.eval()
        self.Net_Demoire_TMB.eval()

        # GENERATOR-1
        learned_Moire_64 = self.G_Artifact_64_1(historgram)
        pseudo_Moire_64 = CLEAR_64 * learned_Moire_64 

        pseudo_Moire_filter_64 = pseudo_Moire_64 - kornia.filters.median_blur(pseudo_Moire_64, (3, 3)) 
        pseudo_Moire_cat_64 = torch.cat((pseudo_Moire_64, pseudo_Moire_filter_64), dim=1)   

        # GENERATOR-2
        z_64_2 = Variable(self.Tensor(np.random.uniform(-1, 1, size=CLEAR_64.shape).astype(np.float32)).to(cuda))
        deep_real_Clean_Noise_64 = torch.cat((pseudo_Moire_64.detach(), z_64_2), dim=1)    
        learned_Moire_64_2 = self.G_Artifact_64_2(deep_real_Clean_Noise_64)                      
        deep_pseudo_Moire_64 = pseudo_Moire_64.detach() + learned_Moire_64_2   

        deep_pseudo_Moire_filter_64 = deep_pseudo_Moire_64 - kornia.filters.median_blur(deep_pseudo_Moire_64, (3, 3))
        deep_pseudo_Moire_cat_64 = torch.cat((deep_pseudo_Moire_64, deep_pseudo_Moire_filter_64), dim=1)


        # G2-CYCLE LOSS
        # Real Moire(256 size) -> DemoireNet -> Demoire image -> Downsampling for Multi-GAN
        Demoire_64 = self.Net_Demoire_64(MOIRE_64).detach()

        # DEMOIRE HISTOGRAM
        Demoire_1, Demoire_2, Demoire_3, Demoire_4 = torch.chunk(Demoire_64, 4, dim=0)

        Demoire_1 = torch.squeeze(Demoire_1)
        Demoire_2 = torch.squeeze(Demoire_2)
        Demoire_3 = torch.squeeze(Demoire_3)
        Demoire_4 = torch.squeeze(Demoire_4)

        trans = torchvision.transforms.ToPILImage()

        Demoire_1 = trans(Demoire_1)
        Demoire_2 = trans(Demoire_2)
        Demoire_3 = trans(Demoire_3)
        Demoire_4 = trans(Demoire_4)

        Demoire_GRAY_1 = transforms.Grayscale(num_output_channels=1)(Demoire_1)
        Demoire_GRAY_2 = transforms.Grayscale(num_output_channels=1)(Demoire_2)
        Demoire_GRAY_3 = transforms.Grayscale(num_output_channels=1)(Demoire_3)
        Demoire_GRAY_4 = transforms.Grayscale(num_output_channels=1)(Demoire_4)

        Demoire_gray_tensor_1 = torch.as_tensor(Demoire_GRAY_1.histogram())
        Demoire_gray_tensor_2 = torch.as_tensor(Demoire_GRAY_2.histogram())
        Demoire_gray_tensor_3 = torch.as_tensor(Demoire_GRAY_3.histogram())
        Demoire_gray_tensor_4 = torch.as_tensor(Demoire_GRAY_4.histogram())

        Demoire_histogram = torch.stack((Demoire_gray_tensor_1, Demoire_gray_tensor_2, Demoire_gray_tensor_3, Demoire_gray_tensor_4), dim=0)

        Demoire_histogram = Demoire_histogram / 64
        Demoire_histogram = Demoire_histogram.to(device)

        Demoire_bright_alpha = self.Net_Demoire_TMB(Demoire_histogram).detach()
        Demoire_bright_alpha = Demoire_bright_alpha + (1e-3)

        final_Demoire_64 = Demoire_64 / Demoire_bright_alpha

        ######################################
        # final_Demoire_64 histogram
        ######################################

        # DEMOIRE HISTOGRAM
        D_1, D_2, D_3, D_4 = torch.chunk(final_Demoire_64, 4, dim=0)

        D_1 = torch.squeeze(D_1)
        D_2 = torch.squeeze(D_2)
        D_3 = torch.squeeze(D_3)
        D_4 = torch.squeeze(D_4)

        trans = torchvision.transforms.ToPILImage()

        D_1 = trans(D_1)
        D_2 = trans(D_2)
        D_3 = trans(D_3)
        D_4 = trans(D_4)

        D_GRAY_1 = transforms.Grayscale(num_output_channels=1)(D_1)
        D_GRAY_2 = transforms.Grayscale(num_output_channels=1)(D_2)
        D_GRAY_3 = transforms.Grayscale(num_output_channels=1)(D_3)
        D_GRAY_4 = transforms.Grayscale(num_output_channels=1)(D_4)

        D_gray_tensor_1 = torch.as_tensor(D_GRAY_1.histogram())
        D_gray_tensor_2 = torch.as_tensor(D_GRAY_2.histogram())
        D_gray_tensor_3 = torch.as_tensor(D_GRAY_3.histogram())
        D_gray_tensor_4 = torch.as_tensor(D_GRAY_4.histogram())

        final_Demoire_64_histogram = torch.stack(
            (D_gray_tensor_1, D_gray_tensor_2, D_gray_tensor_3, D_gray_tensor_4), dim=0)

        final_Demoire_64_histogram = final_Demoire_64_histogram / 64
        final_Demoire_64_histogram = final_Demoire_64_histogram.to(device)

        learned_Moire_2 = self.G_Artifact_64_1(final_Demoire_64_histogram).detach()

        Reconv_Moire_64_1 = final_Demoire_64 * learned_Moire_2

        deep_Demoire_Noise_64 = torch.cat((Reconv_Moire_64_1, z_64_2), dim=1)
        learned_Moire_cycle_64_2 = self.G_Artifact_64_2(deep_Demoire_Noise_64)
        Reconv_Moire_64_2 = Reconv_Moire_64_1.detach() + learned_Moire_cycle_64_2

        return pseudo_Moire_cat_64, valid_64, deep_pseudo_Moire_cat_64, Reconv_Moire_64_2, MOIRE_64