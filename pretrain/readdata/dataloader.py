import imp
import torchvision
import cv2
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
# augmentation在forward中定义，这里只加载数据（归一化）

class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


class Transforms:
    def __init__(self, size, s=1.0, mean=None, std=None, blur=False):
        self.train_transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=23))
        # self.train_transform.append(torchvision.transforms.ToTensor())
        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            torchvision.transforms.ToTensor(),
        ]
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)



class NCT_CRC_Data(Dataset):
    def __init__(self, blur =False, train=True) -> None:
        self.image_label_pair = [] #元组列表 
        if train:
            a = pd.read_csv('./readdata/NCT_CRC_train.csv')
            pa = list(a['path'])
            la = (a['label'])
            self.image_label_pair = list(zip(pa, la))
            self.transforms = Transforms(150, blur=blur).train_transform
        else:
            a = pd.read_csv('./readdata/NCT_CRC_test.csv')
            pa = list(a['path'])
            la = (a['label'])
            self.image_label_pair = list(zip(pa, la))
            self.transforms = torchvision.transforms.ToTensor()
        # # folder = 'C:\\Users\\86136\\Desktop\\sample'
        # tissues_name = os.listdir(self.Imgfolder)
        # for tis in tissues_name:
        #     tissues_dir = os.path.join(self.Imgfolder, tis)
        #     tissues_imgs = os.listdir(tissues_dir)
        #     for img in tissues_imgs:
        #         image_dir = os.path.join(tissues_dir,img)
        #         self.image_label_pair.append([image_dir, tis])
    def __getitem__(self, index):
        img_path = self.image_label_pair[index][0]
        label = self.image_label_pair[index][1]
        img = cv2.imread(img_path)
        img_trans1 = self.transforms(img)
        img_trans2 = self.transforms(img)
        return img_trans1, img_trans2, label
    def __len__(self):
        return len(self.image_label_pair)
    
