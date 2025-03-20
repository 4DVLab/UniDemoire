import os
import sys

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import torchvision.transforms as transforms

from PIL import Image

from unidemoire.models.shooting.mosaicing_demosaicing_v2 import *
from unidemoire.models.shooting.image_transformer import ImageTransformer

def adjust_contrast_and_brightness(input_img, beta = 30):    
    beta = beta / 255.0     #* 亮度强度
    input_img = torch.clamp(input_img + beta, 0, 1)
    
    return input_img

def simulate_LCD_display(input_img, device):
    """ Simulate the display of raw images on LCD screen
    Input:
        original images (tensor): batch x channel x height x width 
    Output:
        LCD images (tensor): batch x channel x (height x scale_factor)  x (width x scale_factor) 
    """
    b, c, h, w = input_img.shape
    
    simulate_imgs = torch.zeros((b, c, h * 3, w * 3), dtype=torch.float32, device=device)
    red   = input_img[:, 0, :, :].repeat_interleave(3, dim=1)
    green = input_img[:, 1, :, :].repeat_interleave(3, dim=1)
    blue  = input_img[:, 2, :, :].repeat_interleave(3, dim=1)

    simulate_imgs[:, 0, :, 0::3] = red
    simulate_imgs[:, 1, :, 1::3] = green
    simulate_imgs[:, 2, :, 2::3] = blue

    return simulate_imgs


def demosaic_and_denoise(input_img, device):
    """ Apply demosaicing to the images
    Input:
        images (tensor): batch x (height x scale_factor) x (width x scale_factor)
    Output:
        demosaicing images (tensor): batch x channel x (height x scale_factor) x (width x scale_factor)
    """
    input_img = input_img.double()
    demosaicing_imgs = demosaicing_CFA_Bayer_bilinear(input_img)
    demosaicing_imgs = demosaicing_imgs.permute(0, 3, 1, 2)
    return demosaicing_imgs

def simulate_CFA(input_img, device):
    """ Simulate the raw reading of the camera sensor using bayer CFA
    Input:
        images (tensor): batch x channel x (height x scale_factor) x (width x scale_factor)
    Output:
        mosaicing images (tensor): batch x (height x scale_factor) x (width x scale_factor)
    """
    input_img = input_img.permute(0, 2, 3, 1)
    mosaicing_imgs = mosaicing_CFA_Bayer(input_img)
    return mosaicing_imgs

def random_rotation_3(org_images, lcd_images, device):
    """ Simulate the 3D rotatation during the shooting
    Input:
        images (tensor): batch x channel x height x width
    Rotate angle:
        theta (int): (-20, 20)
        phi (int): (-20, 20)
        gamma (int): (-20, 20)
    Output:
        rotated original images (tensor): batch x channel x height x width
        rotated LCD images (tensor): batch x channel x (height x scale_factor) x (width x scale_factor)
    """
    rotate_images     = torch.zeros_like(org_images).to(device)         # (bs, c, h, w)
    rotate_lcd_images = torch.zeros_like(lcd_images).to(device)         # (bs, c, 3h, 3w)

    for n, img in enumerate(org_images):

        Trans_org = ImageTransformer(img)
        Trans_lcd = ImageTransformer(lcd_images[n])

        theta, phi, gamma, rotate_img     = Trans_org.Perspective(random_f=True)
        _,       _,     _, rotate_lcd_img = Trans_lcd.Perspective(random_f=False, theta=theta, phi=phi, gamma=gamma)

        rotate_img     = rotate_img.squeeze(0)  
        rotate_lcd_img = rotate_lcd_img.squeeze(0)
        
        rotate_images[n, :]     = rotate_img
        rotate_lcd_images[n, :] = rotate_lcd_img

    return rotate_images, rotate_lcd_images


def Shooting(org_imgs, device):
    batch_size, channel, img_h, img_w = org_imgs.shape   
    alpha = random.randint(1,4)
    crop_ratio = 0.7

    noise = torch.randn([batch_size, img_h * alpha * 3, img_w * alpha * 3]).to(device)      
    noise = noise / 256.0

    resize_before_lcd = F.interpolate(org_imgs, scale_factor=alpha, mode="bilinear", align_corners=True)
    lcd_images = simulate_LCD_display(resize_before_lcd, device)    
    rotate_images, rotate_lcd_images = random_rotation_3(org_imgs, lcd_images, device)
    
    cfa_img = simulate_CFA(rotate_lcd_images, device)
    cfa_img_noise = cfa_img + noise                            
    # cfa_img_noise = cfa_img_noise.double()
    demosaic_img = demosaic_and_denoise(cfa_img_noise, device)  
    brighter_img = adjust_contrast_and_brightness(demosaic_img, beta=20)    

    at_images = F.interpolate(brighter_img, [img_h, img_w], mode='bilinear', align_corners=True)    
    at_images = torch.clamp(at_images, min=0, max=1)

    crop_edges = transforms.Compose([
        transforms.CenterCrop((int(img_h*crop_ratio), int(img_w*crop_ratio))),
        transforms.Resize((img_h, img_w)),

    ])
    rotate_images = crop_edges(rotate_images)
    at_images     = crop_edges(at_images)

    return at_images, rotate_images



trans = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor()
])
