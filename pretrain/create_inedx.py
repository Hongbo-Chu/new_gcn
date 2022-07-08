import pandas as pd
import cv2
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import shuffle
Imgfolder = "/data_sda/lf/tissuse_project/pretrain_datasets/NCT-CRC-HE-100K"
image_ = []
label_ = []
label_mapping = {
    "ADI": 0,
    "BACK": 1,
    "DEB": 2,
    "LYM": 3,
    "MUC": 4,
    "MUS": 5,
    "NORM": 6,
    "STR": 7,
    "TUM": 8,
}
tissues_name = os.listdir(Imgfolder)
for tis in tissues_name:
  if(tis == 'MUS (1)'):
    continue
  tissues_dir = os.path.join(Imgfolder, tis)
  tissues_imgs = os.listdir(tissues_dir)
  lable_map = label_mapping[tis]
  for img in tissues_imgs:
      image_dir = os.path.join(tissues_dir,img)
      image_.append(image_dir)
      label_.append(lable_map)
ll = {"path":image_, "label":label_}            
df = pd.DataFrame(ll)
df = shuffle(df)
train = df[1:90000]
test = df[90001:]
# train.to_csv('/content/gdrive/MyDrive/tissue_segmentation/readdata/NCT_CRC_train.csv')
# test.to_csv('/content/gdrive/MyDrive/tissue_segmentation/readdata/NCT_CRC_test.csv')
train.to_csv('./readdata/NCT_CRC_train.csv', encoding="utf-8-sig", mode="a")
test.to_csv('./readdata/NCT_CRC_test.csv', encoding="utf-8-sig", mode="a")
# print(test)