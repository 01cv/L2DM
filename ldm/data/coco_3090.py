import os
from PIL import Image, ImageEnhance
from glob import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    """return img as nparray format"""
    noise_level = random.randint(noise_level1, noise_level2)
    img = np.array(img)
    img = img + np.random.normal(0, noise_level, img.shape).astype(np.float32)
    img = np.clip(img, 0., 255.)

    return img


def img_process(img):
    img = Image.open(img)
    # img dark
    bright = ImageEnhance.Brightness(img)
    fac = random.uniform(0.05, 0.5)
    b_img = bright.enhance(factor=fac)
    # add noise
    b_n_img = add_Gaussian_noise(b_img)
    return b_n_img


def get_img(data_dir):
    """return the list of absolute img path"""
    data = os.listdir(data_dir)
    data_list = []
    for i in data:
        data_list.append(os.path.join(data_dir, i))
    return data_list


class COCODataset(Dataset):

    def __init__(self, train_dir=None, val_dir=None, mode='train'):
        self.mode = mode
        self.to_tensor = transforms.ToTensor()

        if mode == 'train':
            train_dir.sort()
            self.path_high_train = train_dir
            # self.low_train = img_process(train_dir)
            print('[ * ] We got %d normal images and %d low light images for TRAIN' % (len(self.path_high_train), len(self.path_high_train)))

        elif mode == "eval":
            val_dir.sort()
            self.path_high_val = val_dir
            # self.low_val = img_process(val_dir)
            print('[ * ] We got %d normal images and %d low light images for EVAL' % (len(self.path_high_val), len(self.path_high_val)))



    def __getitem__(self, idx):
        """
        :param idx: idx for dataloader
        :return: paired image low light and normal light image

        if training, random crop and flip are employed.
        if testing, original image data will be used.
        """
        example = dict()
        img_resize = 256
        # lr_img_resize = 256
        if self.mode == 'train':
            img_low = img_process(self.path_high_train[idx])
            img_high_name = self.path_high_train[idx]
        else:
            img_low = img_process(self.path_high_val[idx])
            img_high_name = self.path_high_val[idx]
        # print("[ * ]",img_low_name,img_high_name)
        img_high = Image.open(img_high_name)
        # img_low = Image.fromarray(img_low.astype('uint8'))
        # img_high.show()
        # img_low.show()
        img_low = ((img_low / 255.0) - 0.5) / 0.5  # 归一化
        img_high = ((np.asarray(img_high) / 255.0) - 0.5) / 0.5  # 归一化
        H, W , C = img_low.shape

        i = random.randint(0, (H - img_resize - 2) // 2) * 2
        j = random.randint(0, (W - img_resize - 2) // 2) * 2

        img_low_crop = img_low[i:i + img_resize, j:j + img_resize]
        img_low_crop = torch.from_numpy(img_low_crop).float()

        # img_low = img_low.permute(2, 0, 1)
        # img_high = img_high.resize((img_resize, img_resize), Image.ANTIALIAS)

        img_high_crop = img_high[i:i + img_resize, j:j + img_resize]
        img_high_crop = torch.from_numpy(img_high_crop).float()

        example["ll_image"] = img_low_crop
        example["hl_image"] = img_high_crop

        return example

    def __len__(self):
        if self.mode == 'train':
            return len(self.path_high_train)
        if self.mode == 'eval':
            return len(self.path_high_val)


class COCODatasetTrain(COCODataset):
    def __init__(self, **kwargs):
        super().__init__(train_dir=get_img('/media/ahu/Storage/Drone/dataset/COCO/train2017'), mode='train', **kwargs)


class COCODatasetVal(COCODataset):
    def __init__(self, **kwargs):
        super().__init__(val_dir=get_img('/media/ahu/Storage/dong/latent-diffusion/data/coco/val'), mode='eval', **kwargs)


# x = COCODatasetTrain()
# y = COCODatasetVal()
# img1 = x[3]
# img2 = y[3]
# print(x[3]["hl_image"], x[3]['ll_image'])
# print(x[3]["hl_image"].shape, x[3]['ll_image'].shape)
#
# print(y[3]['hl_image'], y[3]['ll_image'])
# print(y[3]['hl_image'].shape, y[3]['ll_image'].shape)
# img = Image.fromarray(y[3]['ll_image'].astype('uint8'))
# img.show()