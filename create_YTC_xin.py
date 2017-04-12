#!/usr/bin/env python
import numpy as np
from skimage.transform import resize
import h5py
import os
from PIL import Image
from time import time
import cv2 as cv
import drawFromDataset

Num_img = 15488 #30976

def create_YTC(pathfolder, size):
    imgs = load_ytc_datasets(pathfolder, size)
    # idxs = np.random.permutation(np.arange(imgs.shape[0]))
    # imgs = imgs[idxs]
    print imgs.shape
    return imgs

def load_ytc_datasets(pathfolder, size):
    imgs = np.zeros((Num_img,3,size,size),dtype=np.float32)
    count = 0
    for root, dirs, files in os.walk(pathfolder):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() != '.jpg':
                continue

            filepath = os.path.join(root, filename)
            img = cv.imread(filepath)
            # img = np.array(Image.open(filepath))
            img = crop_image(img)
            # img = cv.resize(img,(32, 32),interpolation=cv.INTER_CUBIC)
            img = cv.resize(img, (size, size), interpolation=cv.INTER_CUBIC)/255.0
            # img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
            img = np.transpose(img, (2, 0, 1))
            # imgs.append(img)
            if count >= Num_img:
                break
            imgs[count, :, :, :] = img
            count += 1


    # imgs = np.array(imgs, dtype=np.float32)
    # imgs = np.transpose(imgs, (0, 3, 1, 2))
    return imgs

def crop_image(img):
    x, y , W, H = 35, 55, 100, 130
    img_new = img[y:(y+H), x:(x+W), :]
    return img_new

if __name__ == '__main__':
    pathfolder = os.path.join(os.getcwd(), 'img_align_celeba')
    start_time = time()
    x = create_YTC(pathfolder, 16)
    end_time = time()
    # drawFromDataset.draw_save_images(x,10)
    print('Time using: %f'%(end_time-start_time))
    f = h5py.File('YTC_LR_ext.hdf5', 'w')
    f.create_dataset('YTC', data=x)
    f.close()

    start_time = time()
    x = create_YTC(pathfolder, 128)
    end_time = time()
    # drawFromDataset.draw_save_images(x,10)
    print('Time using: %f'%(end_time-start_time))
    f = h5py.File('YTC_HR_ext.hdf5', 'w')
    f.create_dataset('YTC', data=x)
    f.close()

