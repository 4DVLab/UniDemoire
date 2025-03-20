import os
import random
from PIL import Image
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_natural_image_list_and_moire_pattern_list(args, mode='train', add_clean_percent=1.0):
    moire_pattern_files = _list_moire_pattern_files_recursively(data_dir=args["moire_pattern_path"])
    if args.natural_dataset_name == 'UHDM':
        uhdm_natural_files    = _list_image_files_recursively(data_dir=args["uhdm_dataset_path"])
        return uhdm_natural_files, moire_pattern_files
            
    elif args.natural_dataset_name == 'FHDMi':
        fhdmi_natural_files = sorted([file for file in os.listdir(args["fhdmi_dataset_path"] + '/target') if file.endswith('.png')])
        return fhdmi_natural_files, moire_pattern_files

    elif args.natural_dataset_name == 'TIP':
        tip_natural_files = sorted([file for file in os.listdir(args["tip_dataset_path"] + '/source') if file.endswith('.png')])
        return tip_natural_files, moire_pattern_files

    elif args.natural_dataset_name == 'AIM':
        if mode=='train':
            aim_natural_files = sorted([file for file in os.listdir(args["aim_dataset_path"] + '/moire') if file.endswith('.jpg')])
        else:
            aim_natural_files = sorted([file for file in os.listdir(args["aim_dataset_path"] + '/moire') if file.endswith('.png')])
        return aim_natural_files, moire_pattern_files

    elif args.natural_dataset_name == 'MHRNID':
        mhrnid_files = get_paths_from_images(path=args["mhrnid_dataset_path"])
        return mhrnid_files, moire_pattern_files

    elif args.natural_dataset_name == 'UHDM and FHDMi':
        uhdm_natural_files  = _list_image_files_recursively(data_dir=args["uhdm_dataset_path"])
        fhdmi_natural_files = sorted([file for file in os.listdir(args["fhdmi_dataset_path"] + '/target') if file.endswith('.png')])
        
        print(f'Clean image percentage: {add_clean_percent*100}%')
        fhdmi_size = len(fhdmi_natural_files)
        fhdmi_sublist_size = int(fhdmi_size * add_clean_percent)
        fhdmi_sublist_files = fhdmi_natural_files[:fhdmi_sublist_size]
        
        return uhdm_natural_files + fhdmi_sublist_files, moire_pattern_files

    else:
        print('Unrecognized data_type!')
        raise NotImplementedError


def get_unpaired_moire_images(args):
    if args.unpaired_real_moire_dataset == 'TIP':
        tip_real_moire_files = sorted([file for file in os.listdir(args["tip_dataset_path"] + '/source') if file.endswith('.png')])
        return tip_real_moire_files
    else:
        print('Unrecognized data_type!')
        raise NotImplementedError


def _list_image_files_recursively(data_dir):
    file_list = []
    for home, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('gt.jpg'):
                file_list.append(os.path.join(home, filename))
    file_list.sort()
    return file_list

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

def _list_moire_pattern_files_recursively(data_dir):
    assert os.path.isdir(data_dir), '{:s} is not a valid directory'.format(data_dir)
    images = []
    for dirpath, _, fnames in sorted(os.walk(data_dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(data_dir)
    return sorted(images)



def default_loader(path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs
    
def crop_loader(crop_size, x, y, path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.crop((x, y, x + crop_size, y + crop_size))
         img = default_toTensor(img)
         imgs.append(img)
    return imgs
    
def resize_loader(resize_size, path_set=[]):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.resize((resize_size,resize_size),Image.BICUBIC)
         img = default_toTensor(img)
         imgs.append(img)
    
    return imgs

def resize_then_crop_loader(crop_size, resize_size, path_set=[]):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        if resize_size == 1920:
            img = img.resize((1920,1080),Image.BICUBIC)
            x = random.randint(0, 1920 - crop_size)
            y = random.randint(0, 1080 - crop_size)
        else:
            img = img.resize((resize_size,resize_size),Image.BICUBIC)
            x = random.randint(0, resize_size - crop_size)
            y = random.randint(0, resize_size - crop_size)
        img = img.crop((x, y, x + crop_size, y + crop_size))
        img = default_toTensor(img)
        imgs.append(img)
    return imgs


def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)
    return composed_transform(img)