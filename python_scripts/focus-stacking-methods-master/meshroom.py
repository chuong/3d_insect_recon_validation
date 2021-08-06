# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:38:54 2020

@author: Jiayuan Liu
"""

import os
import cv2

path = 'E:/ANU/ENGN4200/synthetic_images/test_7/truth'
folders = os.listdir(path)
outfolder = 'E:/ANU/ENGN4200/synthetic_images/test_7/all_img'

for i in folders:
    f2 = os.listdir(os.path.join(path,i))
    for j in f2:
        img_dir = os.path.join(path, i, j)
        img_name = os.listdir(img_dir)    
        img = cv2.imread(os.path.join(img_dir,img_name[0]))
        save_name = img_name[0]
        out_dir = os.path.join(outfolder, save_name)
        cv2.imwrite(out_dir, img)
    #shutil.copyfile(img, out_dir)
    
#%%
import pyexiv2
import glob
import os

imgdir = 'E:/ANU/ENGN4200/synthetic_images/meshroom_input_4k'
extensions = ['*.png', '*.jpg']
files = [file for ext in extensions for file in glob.glob(os.path.join(imgdir, ext))]
for j in files:
    metadata = pyexiv2.ImageMetadata(j)
    metadata.read()
    pyexiv2.ImageData(j)