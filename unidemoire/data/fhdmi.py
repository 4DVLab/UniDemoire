import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from PIL import ImageFile
import os

from .utils import default_loader, crop_loader, resize_loader, resize_then_crop_loader


class fhdmi_datasets(data.Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.loader = args["LOADER"]
        self.image_list = sorted([file for file in os.listdir(self.args["dataset_path"] + '/target') if file.endswith('.png')])

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in_gt = self.image_list[index]
        number = image_in_gt[4:9]
        image_in = 'src_' + number + '.png'
        if self.mode == 'train':
            path_tar = self.args["dataset_path"] + '/target/' + image_in_gt
            path_src = self.args["dataset_path"] + '/source/' + image_in
            if self.loader == 'crop':
                x = random.randint(0, 1920 - self.args["CROP_SIZE"])
                y = random.randint(0, 1080 - self.args["CROP_SIZE"])
                labels, moire_imgs = crop_loader(self.args["CROP_SIZE"], x, y, [path_tar, path_src])

            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args["RESIZE_SIZE"], [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]

            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            path_tar = self.args["dataset_path"] + '/target/' + image_in_gt
            path_src = self.args["dataset_path"] + '/source/' + image_in
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args["RESIZE_SIZE"], [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            else:
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number
        data['mode'] = self.mode
        return data

    def __len__(self):
        return len(self.image_list)