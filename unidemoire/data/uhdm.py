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


class uhdm_datasets(data.Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.loader = args["LOADER"]
        self.image_list = self._list_image_files_recursively(data_dir=self.args["dataset_path"])

    def _list_image_files_recursively(self, data_dir):
        file_list = []
        for home, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('gt.jpg'):
                    file_list.append(os.path.join(home, filename))
        file_list.sort()
        return file_list

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        path_tar = self.image_list[index]
        number = os.path.split(path_tar)[-1][0:4]
        path_src = os.path.split(path_tar)[0] + '/' + os.path.split(path_tar)[-1][0:4] + '_moire.jpg'
        if self.mode == 'train':
            if self.loader == 'crop':
                if os.path.split(path_tar)[0][-5:-3] == 'mi':
                    w = 4624
                    h = 3472
                else:
                    w = 4032
                    h = 3024
                x = random.randint(0, w - self.args["CROP_SIZE"])
                y = random.randint(0, h - self.args["CROP_SIZE"])
                labels, moire_imgs = crop_loader(self.args["CROP_SIZE"], x, y, [path_tar, path_src])

            elif self.loader == 'resize_then_crop':
                labels, moire_imgs = resize_then_crop_loader(self.args["CROP_SIZE"], self.args["RESIZE_SIZE"], [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]

            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args["RESIZE_SIZE"], [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]

            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args["RESIZE_SIZE"], [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            elif self.loader == 'resize_then_crop':
                labels, moire_imgs = resize_loader(self.args["RESIZE_SIZE"], [path_tar, path_src])
                data['origin_label'] = default_loader([path_tar])[0]
            else:
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        data['in_img'] = moire_imgs
        data['label']  = labels
        data['number'] = number
        
        data['mode'] = self.mode
        
        return data

    def __len__(self):
        # return 10       # debug
        return len(self.image_list)