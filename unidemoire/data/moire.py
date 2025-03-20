import os
import random

from PIL import Image, ImageFilter, ImageEnhance

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


class MoirePattern(Dataset):
    def __init__(self, dataset_path, resolution):
        self.resolution = resolution
        self.pil_to_tensor = transforms.ToTensor()
        self.dataset_path = dataset_path
        self.moire_layer_path = get_paths_from_images(self.dataset_path) 

    def __len__(self):
        return len(self.moire_layer_path)

    def calculate_sharpness(self, image):
        image_gray = image.convert('L')
        image_laplace = image_gray.filter(ImageFilter.FIND_EDGES)
        sharpness = np.std(np.array(image_laplace))
        return sharpness

    def calculate_colorfulness(self, image):
        image_lab = image.convert('LAB')
        l, a, b = image_lab.split()
        std_a = np.std(np.array(a))
        std_b = np.std(np.array(b))
        colorfulness = np.sqrt(std_a ** 2 + std_b ** 2)
        return colorfulness

    def calculate_image_quality(self, image):
        sharpness    = self.calculate_sharpness(image)
        colorfulness = self.calculate_colorfulness(image)
        return sharpness, colorfulness

    def __getitem__(self, index):
        while(True):
            ## TODO: try different index moire patterns
            for i in range(3):
                ## TODO: [Multi crop] + [Sharpness & Colorfulness selection]
                img_moire_layer = Image.open(self.moire_layer_path[index])
                self.transform_init() 
                img_moire_layer = self.transform(img_moire_layer)
                sharpness, colorfulness = self.calculate_image_quality(img_moire_layer)
                if sharpness < 15 or colorfulness < 2.0:
                    continue
                else:                    
                    img_moire_layer = ImageEnhance.Contrast(img_moire_layer).enhance(2.0)
                    img_moire_layer = self.pil_to_tensor(img_moire_layer)
                    return { "image": img_moire_layer }
            index = random.randint(0, len(self.moire_layer_path) - 1)
            
    def transform_init(self):
        w = h = self.resolution
        base_transforms = [transforms.RandomHorizontalFlip(p=0.5),]

        q = random.randint(0, 2)
        r = random.randint(0, 1)
        if r == 0:    # 4K crop into (w, h)
            extra_transforms = [transforms.RandomCrop(size=(h, w))]
        elif q == 0:  # 4K to 2K, then crop into (w, h)
            extra_transforms = [transforms.Resize(size=(1440, 2560)), transforms.RandomCrop(size=(h, w))]
        elif q == 1:  # 4K to 1080P, then crop into (w, h)
            extra_transforms = [transforms.Resize(size=(1080, 1920)), transforms.RandomCrop(size=(h, w))]
        elif q == 2:  # 4K resize into (w, h)
            extra_transforms = [transforms.Resize(size=(h, w))]

        tran_transform = transforms.Compose(extra_transforms + base_transforms)
        # test_transform = transforms.Compose([transforms.Resize((h, w))] + base_transforms)
        self.transform = tran_transform