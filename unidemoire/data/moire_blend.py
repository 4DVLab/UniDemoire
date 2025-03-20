import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from PIL import ImageFile
import os

from .utils import *

class moire_blending_datasets(data.Dataset):
    def __init__(self, args, mode='train', paired=True):
        self.args   = args
        self.mode   = mode
        self.loader = args["loader"]
        self.paired = paired
        self.natural_dataset_name = self.args["natural_dataset_name"]

        if self.args["natural_dataset_name"] == "UHDM":
            self.args["dataset_path"] = self.args["uhdm_dataset_path"]
        elif self.args["natural_dataset_name"] == "FHDMi":
            self.args["dataset_path"] = self.args["fhdmi_dataset_path"]
        elif self.args["natural_dataset_name"] == "TIP":
            self.args["dataset_path"] = self.args["tip_dataset_path"]
        elif self.args["natural_dataset_name"] == "MHRNID":
            self.args["dataset_path"] = self.args["mhrnid_dataset_path"]

        #* add_clean_image_only
        if 'add_clean_percent' in self.args:
            self.add_clean_percent = self.args['add_clean_percent']
        else:
            self.add_clean_percent = 1.0
        
        self.natural_list, self.moire_pattern_list = get_natural_image_list_and_moire_pattern_list(self.args, self.mode, self.add_clean_percent)
        if 'unpaired_real_moire_dataset' in self.args:
            self.unpaired_real_moire_list = get_unpaired_moire_images(self.args)        
        self.ori_len = len(self.natural_list)
        
        #* 1 to N modify
        if '1_to_N' in self.args:
            self.N = self.args['1_to_N']
            self.natural_list = self.natural_list * self.N
        
        t_list = [transforms.ToTensor()]
        self.moire_pattern_transform = transforms.Compose(t_list)
        
        if self.loader == 'crop':
            self.size = self.args['crop_size']
        elif self.loader == 'resize':
            self.size = self.args['resize_size']
    
    def __len__(self):
        return len(self.natural_list)
    
    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True        
        path_tar = self.natural_list[index]
        return self.getitem(index, path_tar)

    def getitem(self, index, path_tar):
        data = {}
        if self.mode == 'train':
            moire_pattern = self.getitem_moire_pattern_dataset()
            data['moire_pattern'] = moire_pattern

        if self.natural_dataset_name == "UHDM":
            moire_imgs, labels, number = self.getitem_uhdm_dataset(path_tar)
        elif self.natural_dataset_name == "FHDMi":
            moire_imgs, labels, number = self.getitem_fhdmi_dataset(path_tar)
        elif self.natural_dataset_name == "TIP":
            moire_imgs, labels, number = self.getitem_tip_dataset(path_tar)
        elif self.natural_dataset_name == "AIM":
            moire_imgs, labels, number = self.getitem_aim_dataset(path_tar)
        elif self.natural_dataset_name == "MHRNID":
            moire_imgs, labels, number = self.getitem_mhrnid_dataset(path_tar)
            
        elif self.natural_dataset_name == "UHDM and FHDMi":
            if index % (self.ori_len) < 4500:        #! UHDM training datasets size
                moire_imgs, labels, number = self.getitem_uhdm_dataset(path_tar)
            else:
                moire_imgs, labels, number = self.getitem_fhdmi_dataset(path_tar)
            
        data['natural'] = labels
        data['real_moire'] = moire_imgs
        data['number'] = number
        data['mode'] = self.mode
        
        return data

    def default_toTensor(self, img):
        t_list = [transforms.ToTensor()]
        composed_transform = transforms.Compose(t_list)
        return composed_transform(img)

    def default_loader(self, path):
        return Image.open(path).convert('RGB')

    def capture_moire_pattern_transform(self, img):
        w = h = self.size   
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
        test_transform = transforms.Compose(    
            [
                transforms.Resize((h, w)), 
                transforms.ToTensor(),
            ]
        )
        if self.mode == "train":
            transform = tran_transform
        else:
            transform = test_transform
        return transform(img)

    def sample_moire_pattern_transform(self, img):
        w = h = self.size   
        transform = transforms.Compose(
            [
                transforms.Resize((h, w)), 
                transforms.ToTensor(),
            ]
        )
        return transform(img)        

    def moirespace_moire_pattern_transform(self, img):
        w = h = self.size   
        transform = transforms.Compose(
            [
                transforms.RandomCrop((h, w)), 
                transforms.ToTensor(),
            ]
        )
        return transform(img) 

    def getitem_moire_pattern_dataset(self):
        index = random.randint(0, len(self.moire_pattern_list) - 1)
        path_moire    = self.moire_pattern_list[index]
        moire_pattern = Image.open(path_moire).convert('RGB')
        
        ##! Using sampled moire patterns here...    
        moire_pattern = self.sample_moire_pattern_transform(moire_pattern)

        ##* If you want to use the moire pattern from the 4K dataset directly...
        # moire_pattern = self.capture_moire_pattern_transform(moire_pattern)
        
        ##* Or if you want to use the patterns from MoireSpace...
        # moire_pattern = self.moirespace_moire_pattern_transform(moire_pattern)
        
        return moire_pattern
    
    def getitem_uhdm_dataset(self, path_tar):
        number = os.path.split(path_tar)[-1].split("_")[0]

        if self.mode == 'train':
            if self.paired == True:
                path_src = os.path.split(path_tar)[0] + '/' + number + '_moire.jpg'
            else:
                while(1):
                    random_num = random.randinit(0, len(self.natural_list) - 1)
                    random_num_str = "{:04d}".format(random_num)
                    if random_num_str != number:
                        path_src = os.path.split(path_tar)[0] + '/' + random_num_str + '_moire.jpg'
                        break   

            if self.loader == 'crop':
                if os.path.split(path_tar)[0][-5:-3] == 'mi':
                    w = 4624
                    h = 3472
                else:
                    w = 4032
                    h = 3024
                x = random.randint(0, w - self.args["crop_size"])
                y = random.randint(0, h - self.args["crop_size"])
                labels, moire_imgs = crop_loader(self.args["crop_size"], x, y, [path_tar, path_src])
            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args["resize_size"], [path_tar, path_src])
            elif self.loader == 'default': 
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            path_src = os.path.split(path_tar)[0] + '/' + number + '_moire.jpg' 
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args["resize_size"], [path_tar, path_src])
            elif self.loader == 'resize_then_crop':
                labels, moire_imgs = resize_then_crop_loader(self.args["resize_size"], [path_tar, path_src])
            else:   
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        return moire_imgs, labels, number
    
    def getitem_fhdmi_dataset(self, image_in_gt):
        number = image_in_gt[4:9]
        if self.mode == 'train':
            if self.paired == True:
                image_in = 'src_' + number + '.png'     
            else:
                while(1):
                    random_num = random.randinit(0, len(self.natural_list) - 1)
                    random_num_str = "{:05d}".format(random_num)
                    if random_num_str != number:
                        image_in = 'src_' + random_num_str + '.png'
                        break   
            path_tar = self.args["fhdmi_dataset_path"] + '/target/' + image_in_gt
            path_src = self.args["fhdmi_dataset_path"] + '/source/' + image_in
            if self.loader == 'crop':
                x = random.randint(0, 1920 - self.args["crop_size"])
                y = random.randint(0, 1080 - self.args["crop_size"])
                labels, moire_imgs = crop_loader(self.args["crop_size"], x, y, [path_tar, path_src])

            elif self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args["resize_size"], [path_tar, path_src])

            elif self.loader == 'default':
                labels, moire_imgs = default_loader([path_tar, path_src])

        elif self.mode == 'test':
            image_in = 'src_' + number + '.png'     
            path_tar = self.args["fhdmi_dataset_path"] + '/target/' + image_in_gt
            path_src = self.args["fhdmi_dataset_path"] + '/source/' + image_in
            if self.loader == 'resize':
                labels, moire_imgs = resize_loader(self.args["resize_size"], [path_tar, path_src])
            else:
                labels, moire_imgs = default_loader([path_tar, path_src])

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError
        
        return moire_imgs, labels, number
    
    def getitem_tip_dataset(self, image_in):
        image_in_gt = image_in[:-10] + 'target.png'
        number = image_in_gt[:-11]

        if self.mode == 'train':
            if self.paired != True:
                while(1):
                    random_index = random.randinit(0, len(self.natural_list) - 1)
                    image_in = self.natural_list[random_index]
                    random_num_str = image_in[:-11]
                    if random_num_str != number:
                        break   
            gt_path = self.args["tip_dataset_path"] + '/target/' + image_in_gt
            in_path = self.args["tip_dataset_path"] + '/source/' + image_in

            labels = self.default_loader(gt_path)
            moire_imgs = self.default_loader(in_path)

            w, h = labels.size
            i = random.randint(-6, 6)
            j = random.randint(-6, 6)
            labels = labels.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))
            moire_imgs = moire_imgs.crop((int(w / 6) + i, int(h / 6) + j, int(w * 5 / 6) + i, int(h * 5 / 6) + j))

            labels = labels.resize((256, 256), Image.BILINEAR)
            moire_imgs = moire_imgs.resize((256, 256), Image.BILINEAR)

        elif self.mode == 'test':
            labels = self.default_loader(self.args["tip_dataset_path"] + '/target/' + image_in_gt)
            moire_imgs = self.default_loader(self.args["tip_dataset_path"] + '/source/' + image_in)

            w, h = labels.size
            labels = labels.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            moire_imgs = moire_imgs.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))

            labels = labels.resize((256, 256), Image.BILINEAR)
            moire_imgs = moire_imgs.resize((256, 256), Image.BILINEAR)

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError

        moire_imgs = self.default_toTensor(moire_imgs)       
        labels = self.default_toTensor(labels)

        return moire_imgs, labels, number

    def getitem_mhrnid_dataset(self, path_tar):
        mhrnid_image = Image.open(path_tar).convert('RGB')
        w = h = self.size
        mhrnid_image_w, mhrnid_image_h = mhrnid_image.size

        crop_transform   = transforms.Compose([transforms.RandomCrop(size=(h, w)), transforms.ToTensor()])
        resize_transform = transforms.Compose([transforms.Resize(size=(h, w)), transforms.ToTensor()])
        
        if mhrnid_image_w >= w and mhrnid_image_h >= h:
            labels = crop_transform(mhrnid_image)
        else:
            labels = resize_transform(mhrnid_image)
        
        if 'unpaired_real_moire_dataset' in self.args and self.args['unpaired_real_moire_dataset'] == "TIP":
            unpaired_real_moire_index = random.randint(0, len(self.unpaired_real_moire_list) - 1)
            unpaired_real_moire_path = self.unpaired_real_moire_list[unpaired_real_moire_index]
            image_in_gt = unpaired_real_moire_path[:-10] + 'target.png'
            number = image_in_gt[:-11]
            unpaired_real_moire_path = self.args["tip_dataset_path"] + '/source/' + unpaired_real_moire_path
            # unpaired_real_moire = Image.open(unpaired_real_moire_path).convert('RGB')
            unpaired_real_moire = self.default_loader(unpaired_real_moire_path)            
            w, h = unpaired_real_moire.size
            i = random.randint(-6, 6)
            j = random.randint(-6, 6)
            unpaired_real_moire = unpaired_real_moire.crop((int(w / 6), int(h / 6), int(w * 5 / 6), int(h * 5 / 6)))
            # unpaired_real_moire = unpaired_real_moire.resize((256, 256), Image.BILINEAR)
            unpaired_real_moire = unpaired_real_moire.resize((self.size, self.size), Image.BILINEAR)
            moire_imgs = unpaired_real_moire
            moire_imgs = self.default_toTensor(moire_imgs)
        else:
            moire_imgs = labels
            number = os.path.split(path_tar)[-1]
        
        return moire_imgs, labels, number