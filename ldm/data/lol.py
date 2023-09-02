import os
from PIL import Image
from glob import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class LOLDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        """
        init a LOL dataset obj
        :param data_dir: data location of LOL dataset
        :param transform: you should know what it means
        :param mode: 'train'/'eval'
        """
        self.mode = mode

        self.to_tensor = transforms.ToTensor()
        self.root_dir = data_dir
        self.path_real = os.path.join(self.root_dir, 'our485')
        self.path_syn = os.path.join(self.root_dir, 'syn')
        self.path_eval = os.path.join(self.root_dir, 'eval15')
        if mode == "train":
            self.high_names = glob(self.path_real + '/high/*.png')  # + glob(self.path_syn + '/high/*.png')
            self.low_names = glob(self.path_real + '/low/*.png')  # + glob(self.path_syn + '/low/*.png')
            print('[ * ] We got %d normal images and %d low light images for TRAIN' % (
            len(self.high_names), len(self.low_names)))
        elif mode == "eval":
            self.high_names = glob(self.path_eval + '/high/*.png')
            self.low_names = glob(self.path_eval + '/low/*.png')
            print('[ * ] We got %d normal images and %d low light images for EVAL' % (
            len(self.high_names), len(self.low_names)))
        self.high_names.sort()
        self.low_names.sort()
        # random.shuffle(self.low_names)
        assert len(self.high_names) == len(self.low_names), \
            "num of input must equal to that of gt, but %d != %d." % (len(self.low_names), len(self.high_names))
        # print('We got %d normal images and %d low light images' % (len(self.high_names), len(self.low_names)))

    def __getitem__(self, idx):
        """
        :param idx: idx for dataloader
        :return: paired image low light and normal light image
        """
        example = dict()
        img_resize = 256
        lr_img_resize = 256


        img_low_name = self.low_names[idx]
        img_high_name = self.high_names[idx]
        # print("[ * ]",img_low_name,img_high_name)
        img_low = Image.open(img_low_name)
        img_high = Image.open(img_high_name)

        img_low = ((np.asarray(img_low) / 255.0) - 0.5) / 0.5  # 归一化了
        img_high = ((np.asarray(img_high) / 255.0) - 0.5) / 0.5  # 归一化了
        H, W , C = img_low.shape

        """
        if training, random crop and flip are employed.
        if testing, original image data will be used.
        """

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
        return max(len(self.high_names), len(self.low_names))

class LOLDatasetTrain(LOLDataset):
    def __init__(self, **kwargs):
        super().__init__(data_dir='data/LOL/LOLdataset/', mode='train', **kwargs)


class LOLDatasetVal(LOLDataset):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(data_dir='data/LOL/LOLdataset/', mode='eval', **kwargs)
