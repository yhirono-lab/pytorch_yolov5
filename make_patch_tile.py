import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import re
import torch
import torchvision
import pprint
import sys
from pathlib import Path

args = sys.argv
color=(10, 128, 10)
txt_color=(255, 255, 255)
name_color = (0, 0, 0)
lw = 2
split = 24
row = 6

def save_images(save_dir, images, subtype, title):
    images_tensor = torch.as_tensor(images)
    joined_images_tensor = torchvision.utils.make_grid(images_tensor, nrow=row, padding=10)
    joined_images = joined_images_tensor.numpy()
    jointed = np.transpose(joined_images, [1,2,0])
    plt.tick_params(color='white', labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.title(title)
    plt.imshow(jointed)
    if subtype:
        plt.savefig(f'{save_dir}/{subtype}/{title}.tif', bbox_inches='tight', pad_inches=0.1, format='tif', dpi=300)
    else:
        plt.savefig(f'./{save_dir}/{title}.tif', bbox_inches='tight', pad_inches=0.1, format='tif', dpi=300)
    plt.clf()

img_dir = Path(args[1]) / '*.tif'
save_dir = Path('./runs/detect') / Path(f'patch_tile/{args[1]}')
img_list = glob.glob(str(img_dir))
img_list.sort()
print(len(img_list))


### サブタイプ別に作成する
file_dic = {}
for img in img_list:
    file_name = os.path.split(img)[-1]
    if len(file_name.split('_')[0]) == 6:
        subtype = os.path.split(img)[-1].split('_')[2]
    else:
        subtype = os.path.split(img)[-1].split('_')[3]
        
    if subtype not in file_dic:
        file_dic[subtype] = []
    file_dic[subtype].append(img)

img_count = 0
for subtype in file_dic:
    file_list = file_dic[subtype]
    images = np.zeros((len(file_list), 224, 224, 3), np.int64)
    for i, img in enumerate(file_list):
        img = cv2.imread(img)
        images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images = np.transpose(images, [0,3,1,2]) # NHWC -> NCHW に変換
    img_count += images.shape[0]
    pages = images.shape[0]//split if images.shape[0]%split == 0 else images.shape[0]//split+1
    
    if not os.path.exists(save_dir / subtype):
        os.makedirs(save_dir / subtype)
    for p in range(pages):
        img_split = images[p*split:(p+1)*split]
        title = f'{subtype} No.{p*split+1}...No.{p*split+img_split.shape[0]-1}'
        save_images(save_dir, img_split, subtype=subtype, title=title)
print(img_count)
        

### 全サブタイプまとめて作成する
# count = 0
# images = np.zeros((len(img_list), 224, 224, 3), np.int64)
# for i, (img, txt) in enumerate(zip(img_list, txt_list)):
#     images[i] = bbox_label(img, txt)

# images = np.transpose(images, [0,3,1,2]) # NHWC -> NCHW に変換
# print(images.shape)
# pages = images.shape[0]//split if images.shape[0]%split == 0 else images.shape[0]//split+1

# for p in range(pages):
#     img_split = images[p*split:(p+1)*split]
#     title = f'No.{p*split+1}...No.{(p+1)*split+1}'
#     save_images(img_split, subtype='', title=title)
    
