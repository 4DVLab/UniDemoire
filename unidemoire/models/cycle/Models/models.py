import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
import numpy as np
from .utils import *

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torchvision.utils import save_image, make_grid

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#      DEMOIRE-NET
##############################
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU()
    )

class ResidualBlock3(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock3, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.block(x) + x
        return out

class PS_Upsample(nn.Module):
    def __init__(self, in_size):
        super(PS_Upsample, self).__init__()

        out_size = in_size * 2

        self.up = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.PixelShuffle(2)
        )

    def forward(self, inputs):
        outputs = self.up(inputs)

        return outputs

class GM2_UNet5_256(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GM2_UNet5_256, self).__init__()

        # 256->128
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU())
        # 128->64
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU())
        # 64->32
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU())
        # 32->16
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU())
        # 16->8
        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU())

        self.RB_1 = ResidualBlock3(64)
        self.RB_2 = ResidualBlock3(128)
        self.RB_3 = ResidualBlock3(256)
        self.RB_4 = ResidualBlock3(512)
        self.RB_5 = ResidualBlock3(512)

        self.stride_down1 = nn.Conv2d(64, 64, 2, padding=0, stride=2)
        self.stride_down2 = nn.Conv2d(128, 128, 2, padding=0, stride=2)
        self.stride_down3 = nn.Conv2d(256, 256, 2, padding=0, stride=2)
        self.stride_down4 = nn.Conv2d(512, 512, 2, padding=0, stride=2)
        self.stride_down5 = nn.Conv2d(512, 512, 2, padding=0, stride=2)
        self.GAP_down6 = nn.AdaptiveAvgPool2d(1)

        self.PS_Upsample5 = PS_Upsample(512)
        self.PS_Upsample4 = PS_Upsample(512)
        self.PS_Upsample3 = PS_Upsample(512)
        self.PS_Upsample2 = PS_Upsample(256)
        self.PS_Upsample1 = PS_Upsample(128)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up6 = double_conv(512 + 512, 512)
        self.dconv_up5 = double_conv(512 + 256, 512)
        self.dconv_up4 = double_conv(512 + 256, 512)
        self.dconv_up3 = double_conv(256 + 256, 256)
        self.dconv_up2 = double_conv(128 + 128, 128)
        self.dconv_up1 = double_conv(64 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, input):

        # conv1 (3,256,256) -> (64,256,256)
        conv_1 = self.conv_1(input)
        # RB1 (64,256,256) -> (64,256,256)
        RB_1 = self.RB_1(conv_1)
        # DOWN1 (64,256,256) -> (64,128,128)
        Down_1 = self.stride_down1(RB_1)


        # conv2 (64,128,128) -> (128,128,128)
        conv_2 = self.conv_2(Down_1)
        # RB2 (128,128,128) -> (128,128,128)
        RB_2 = self.RB_2(conv_2)
        # DOWN2 (128,128,128) -> (128,64,64)
        Down_2 = self.stride_down2(RB_2)


        # conv3 (128,64,64) -> (256,64,64)
        conv_3 = self.conv_3(Down_2)
        # RB3 (64,256,256) -> (64,256,256)
        RB_3 = self.RB_3(conv_3)
        # DOWN3 (256,64,64) -> (256,32,32)
        Down_3 = self.stride_down3(RB_3)


        # conv4 (256,32,32) -> (512,32,32)
        conv_4 = self.conv_4(Down_3)
        # RB4 (512,32,32) -> (512,32,32)
        RB_4 = self.RB_4(conv_4)
        # DOWN4 (512,32,32) -> (512,16,16)
        Down_4 = self.stride_down4(RB_4)


        # RB5 (512,16,16) -> (512,16,16)
        RB_5 = self.RB_5(Down_4)
        # DOWN5 (512,16,16) -> (512,8,8)
        Down_5 = self.stride_down5(RB_5)
        b, c, w, h = Down_5.size()

        # DOWN6 (512,8,8) -> (512,1,1)
        Down_6 = self.GAP_down6(Down_5)

        # Concat (512,1,1)*8 -> (512,8,8)
        CAT_Down_6 = Down_6.repeat(1,1,w,h)


        # Concat (512+512,8,8) -> (1024,8,8)
        x = torch.cat([Down_5, CAT_Down_6], dim=1)
        # Dconv (512+512,8,8) -> (512,8,8)
        x = self.dconv_up6(x)


        # Up (512, 8, 8)-> (256, 16, 16)
        x = self.PS_Upsample5(x)
        # Concat (512+256,16,16)
        x = torch.cat([x, RB_5], dim=1)
        # Dconv (512+256,16,16) -> (512,16,16)
        x = self.dconv_up5(x)


        # Up (512, 16, 16)-> (256, 32, 32)
        x = self.PS_Upsample4(x)
        # Concat (512+256,32,32)
        x = torch.cat([x, RB_4], dim=1)
        # Dconv (512+256,32,32) -> (512,32,32)
        x = self.dconv_up4(x)


        # Up (512, 32, 32)-> (256, 64, 64)
        x = self.PS_Upsample3(x)
        # Concat (256+256,64,64)
        x = torch.cat([x, RB_3], dim=1)
        # Dconv (256+256,64,64) -> (256,64,64)
        x = self.dconv_up3(x)


        # Up (256, 64, 64)-> (128, 128, 128)
        x = self.PS_Upsample2(x)
        # Concat (128+128,128,128)
        x = torch.cat([x, RB_2], dim=1)
        # Dconv (128+128,128,128) -> (128,128,128)
        x = self.dconv_up2(x)


        # Up (128, 128, 128)-> (64, 256, 256)
        x = self.PS_Upsample1(x)
        # Concat (64+64,256,256)
        x = torch.cat([x, RB_1], dim=1)
        # Dconv (64+64,256,256) -> (64,256,256)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

class GM2_UNet5_128(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GM2_UNet5_128, self).__init__()

        # 256->128
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU())
        # 128->64
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU())
        # 64->32
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU())
        # 32->16
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU())

        self.RB_1 = ResidualBlock3(64)
        self.RB_2 = ResidualBlock3(128)
        self.RB_3 = ResidualBlock3(256)
        self.RB_4 = ResidualBlock3(512)

        self.stride_down1 = nn.Conv2d(64, 64, 2, padding=0, stride=2)
        self.stride_down2 = nn.Conv2d(128, 128, 2, padding=0, stride=2)
        self.stride_down3 = nn.Conv2d(256, 256, 2, padding=0, stride=2)
        self.stride_down4 = nn.Conv2d(512, 512, 2, padding=0, stride=2)

        self.GAP_down6 = nn.AdaptiveAvgPool2d(1)

        self.PS_Upsample4 = PS_Upsample(512)
        self.PS_Upsample3 = PS_Upsample(512)
        self.PS_Upsample2 = PS_Upsample(256)
        self.PS_Upsample1 = PS_Upsample(128)

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.dconv_up5 = double_conv(512 + 512, 512)
        self.dconv_up4 = double_conv(512 + 256, 512)
        self.dconv_up3 = double_conv(256 + 256, 256)
        self.dconv_up2 = double_conv(128 + 128, 128)
        self.dconv_up1 = double_conv(64 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, input):

        # conv1 (3,128,128) -> (64,128,128)
        conv_1 = self.conv_1(input)
        # RB1 (64,128,128) -> (64,128,128)
        RB_1 = self.RB_1(conv_1)
        # DOWN1 (64,128,128) -> (64,64,64)
        Down_1 = self.stride_down1(RB_1)


        # conv2 (64,64,64) -> (128,64,64)
        conv_2 = self.conv_2(Down_1)
        # RB2 (128,64,64) -> (128,64,64)
        RB_2 = self.RB_2(conv_2)
        # DOWN2 (128,64,64) -> (128,32,32)
        Down_2 = self.stride_down2(RB_2)


        # conv3 (128,32,32) -> (256,32,32)
        conv_3 = self.conv_3(Down_2)
        # RB3 (256,32,32) -> (256,32,32)
        RB_3 = self.RB_3(conv_3)
        # DOWN3 (256,32,32) -> (256,16,16)
        Down_3 = self.stride_down3(RB_3)


        # conv4 (256,16,16) -> (512,16,16)
        conv_4 = self.conv_4(Down_3)
        # RB4 (512,16,16) -> (512,16,16)
        RB_4 = self.RB_4(conv_4)
        # DOWN4 (512,16,16) -> (512,8,8)
        Down_4 = self.stride_down4(RB_4)

        b, c, w, h = Down_4.size()

        # DOWN6 (512,8,8) -> (512,1,1)
        Down_5 = self.GAP_down6(Down_4)

        # Concat (512,1,1)*8 -> (512,8,8)
        CAT_Down_5 = Down_5.repeat(1,1,w,h)


        # Concat (512+512,8,8) -> (1024,8,8)
        x = torch.cat([Down_4, CAT_Down_5], dim=1)
        # Dconv (512+512,8,8) -> (512,8,8)
        x = self.dconv_up5(x)


        # Up (512, 8, 8)-> (256, 16, 16)
        x = self.PS_Upsample4(x)
        # Concat (256+512,16,16)
        x = torch.cat([x, RB_4], dim=1)
        # Dconv (256+512,16,16) -> (512,16,16)
        x = self.dconv_up4(x)


        # Up (512, 16, 16)-> (256, 32, 32)
        x = self.PS_Upsample3(x)
        # Concat (256+256,32,32)
        x = torch.cat([x, RB_3], dim=1)
        # Dconv (256+256,32,32) -> (256,32,32)
        x = self.dconv_up3(x)


        # Up (256, 32, 32)-> (128, 64, 64)
        x = self.PS_Upsample2(x)
        # Concat (128+128,64,64)
        x = torch.cat([x, RB_2], dim=1)
        # Dconv (128+128,64,64) -> (128,64,64)
        x = self.dconv_up2(x)


        # Up (128, 64, 64)-> (64, 128, 128)
        x = self.PS_Upsample1(x)
        # Concat (64+64,128,128)
        x = torch.cat([x, RB_1], dim=1)
        # Dconv (64+64,128,128) -> (64,128,128)
        x = self.dconv_up1(x)

        # conv_last (64,128,128) -> (3,128,128)
        out = self.conv_last(x)

        return out

class GM2_UNet5_64(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GM2_UNet5_64, self).__init__()

        # 256->128
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU())
        # 128->64
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU())
        # 64->32
        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU())

        self.RB_1 = ResidualBlock3(64)
        self.RB_2 = ResidualBlock3(128)
        self.RB_3 = ResidualBlock3(256)

        self.stride_down1 = nn.Conv2d(64, 64, 2, padding=0, stride=2)
        self.stride_down2 = nn.Conv2d(128, 128, 2, padding=0, stride=2)
        self.stride_down3 = nn.Conv2d(256, 256, 2, padding=0, stride=2)

        self.GAP_down6 = nn.AdaptiveAvgPool2d(1)

        self.PS_Upsample3 = PS_Upsample(256)
        self.PS_Upsample2 = PS_Upsample(256)
        self.PS_Upsample1 = PS_Upsample(128)

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(256 + 256, 256)
        self.dconv_up3 = double_conv(256 + 128, 256)
        self.dconv_up2 = double_conv(128 + 128, 128)
        self.dconv_up1 = double_conv(64 + 64, 64)

        self.conv_last = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, padding=0, stride=1)
        )

    def forward(self, input):

        # conv1 (3,64,64) -> (64,64,64)
        conv_1 = self.conv_1(input)
        # RB1 (64,64,64) -> (64,64,64)
        RB_1 = self.RB_1(conv_1)
        # DOWN1 (64,64,64) -> (64,32,32)
        Down_1 = self.stride_down1(RB_1)


        # conv2 (64,32,32) -> (128,32,32)
        conv_2 = self.conv_2(Down_1)
        # RB2 (128,32,32) -> (128,32,32)
        RB_2 = self.RB_2(conv_2)
        # DOWN2 (128,32,32) -> (128,16,16)
        Down_2 = self.stride_down2(RB_2)


        # conv3 (128,16,16) -> (256,16,16)
        conv_3 = self.conv_3(Down_2)
        # RB3 (256,16,16) -> (256,16,16)
        RB_3 = self.RB_3(conv_3)
        # DOWN3 (256,16,16) -> (256,8,8)
        Down_3 = self.stride_down3(RB_3)

        b, c, w, h = Down_3.size()

        # DOWN6 (256,8,8) -> (256,1,1)
        Down_4 = self.GAP_down6(Down_3)

        # Concat (256,1,1)*8 -> (256,8,8)
        CAT_Down_4 = Down_4.repeat(1,1,w,h)


        # Concat (256+256,8,8) -> (256,8,8)
        x = torch.cat([Down_3, CAT_Down_4], dim=1)
        # Dconv (256+256,8,8) -> (256,8,8)
        x = self.dconv_up4(x)

        # Up (256, 8, 8)-> (128, 16, 16)
        x = self.PS_Upsample3(x)
        # Concat (256+256,16,16)
        x = torch.cat([x, RB_3], dim=1)
        # Dconv (256+256,16,16) -> (256,16,16)
        x = self.dconv_up3(x)


        # Up (256, 16, 16)-> (128, 32, 32)
        x = self.PS_Upsample2(x)
        # Concat (128+128,32,32)
        x = torch.cat([x, RB_2], dim=1)
        # Dconv (128+128,32,32) -> (128,32,32)
        x = self.dconv_up2(x)


        # Up (128, 32, 32)-> (64, 64, 64)
        x = self.PS_Upsample1(x)
        # Concat (64+64,128,128)
        x = torch.cat([x, RB_1], dim=1)
        # Dconv (64+64,128,128) -> (64,128,128)
        x = self.dconv_up1(x)

        # conv_last (64,128,128) -> (3,128,128)
        out = self.conv_last(x)

        return out

##############################
#      G1- Color historgram
##############################
class TMB(nn.Module):
    def __init__(self, in_size, out_size):
        super(TMB, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):

        outputs = self.mlp(inputs)
        #print(outputs)
        b, c = outputs.size()
        outputs_2 = torch.reshape(outputs, (b, c, 1, 1))
        #print(outputs_2)
        outputs = outputs_2

        return outputs

##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape_, height_, width_):
        super(Discriminator, self).__init__()

        channels = input_shape_
        height = height_
        width = width_


        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            #if normalize:
                #layers.append(nn.InstanceNorm2d(out_filters))
                #layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 512),
            *discriminator_block(512, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

