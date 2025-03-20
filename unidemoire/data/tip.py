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

class tip_datasets(data.Dataset):

    def __init__(self, args, mode='train'):
        
        data_path = args['dataset_path']
        image_list = sorted([file for file in os.listdir(data_path + '/source') if file.endswith('.png')])
        self.image_list = image_list
        self.args = args
        self.mode = mode
        t_list = [transforms.ToTensor()]
        self.composed_transform = transforms.Compose(t_list)

    def default_loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in = self.image_list[index]
        image_in_gt = image_in[:-10] + 'target.png'
        number = image_in_gt[:-11]

        if self.mode == 'train':
            labels = self.default_loader(self.args['dataset_path'] + '/target/' + image_in_gt)
            moire_imgs = self.default_loader(self.args['dataset_path'] + '/source/' + image_in)

            w, h = labels.size
            i = random.randint(-6, 6)
            j = random.randint(-6, 6)
            labels = labels.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            moire_imgs = moire_imgs.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))

            labels = labels.resize((256, 256), Image.BILINEAR)
            moire_imgs = moire_imgs.resize((256, 256), Image.BILINEAR)

        elif self.mode == 'test':
            labels = self.default_loader(self.args['dataset_path'] + '/target/' + image_in_gt)
            moire_imgs = self.default_loader(self.args['dataset_path'] + '/source/' + image_in)

            w, h = labels.size
            labels = labels.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            moire_imgs = moire_imgs.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))

            labels = labels.resize((256, 256), Image.BILINEAR)
            moire_imgs = moire_imgs.resize((256, 256), Image.BILINEAR)


        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        moire_imgs = self.composed_transform(moire_imgs)
        labels = self.composed_transform(labels)

        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number
        data['mode'] = self.mode
        return data

    def __len__(self):
        return len(self.image_list)