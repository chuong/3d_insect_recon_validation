# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:10:34 2020

@author: Jiayuan Liu

This script removes the background from frontlight images

"""

import cv2
import os
import matplotlib.pyplot as plt

#%% For single image

maskdir = 'E:/ANU/ENGN4200/synthetic_images/SCANT/mask/x=0/y=0/fusion_ECC_AFFINE_pyramid.jpg'
inputdir = 'E:/ANU/ENGN4200/synthetic_images/SCANT/stack/x=0/y=0/fusion_ECC_AFFINE_pyramid.jpg'

mask = cv2.imread(maskdir)
img = cv2.imread(inputdir)
mask_grays = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
mask_binarys = cv2.threshold(mask_grays, 190, 255, cv2.THRESH_BINARY)[1]

mask = cv2.cvtColor(mask_binarys, cv2.COLOR_GRAY2BGR)

img_out = cv2.bitwise_or(img, mask)
img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
plt.imshow(img_out)

#%% For multiple images

# directory for frontlight and backlight images
maskdir = 'E:/ANU/ENGN4200/synthetic_images/SCANT/mask/x=%d'
inputdir = 'E:/ANU/ENGN4200/synthetic_images/SCANT/stack/x=%d'
mask_x = [maskdir % i for i in range(0,13)]
img_x = [inputdir % i for i in range(0,13)]

maskfolders = [os.listdir(i) for i in mask_x]
imgfolders = [os.listdir(i) for i in img_x]

maskdirs = []
img_dirs = []
for i in range(0,13):
    mask_y = [os.path.join(mask_x[i], j, 'fusion_ECC_AFFINE_pyramid.jpg') for j in maskfolders[i]]
    maskdirs.append(mask_y)

for i in range(0,13):
    img_y = [os.path.join(img_x[i], j, 'fusion_ECC_AFFINE_pyramid.jpg') for j in imgfolders[i]]
    img_dirs.append(img_y)

flat_img = []
flat_mask = []
for i in img_dirs:
    for j in i:
        flat_img.append(j)

for i in maskdirs:
    for j in i:
        flat_mask.append(j)

# read frontlight and backlight images to a lits of array
image_BGRs = [cv2.imread(x) for x in flat_img]
mask_BGRs = [cv2.imread(y) for y in flat_mask]

# convert mask images to binary
mask_grays = [cv2.cvtColor(z, cv2.COLOR_BGR2GRAY) for z in mask_BGRs]
mask_binarys = [cv2.threshold(n, 190, 255, cv2.THRESH_BINARY)[1] for n in mask_grays]

mask = [cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) for a in mask_binarys]

# remove image background using mask image
img_out = [cv2.bitwise_or(b, c) for b, c in zip(image_BGRs, mask)]


#%% Save images without background    

outfolder = '/flush5/liu220/images/AA/meshroom_4k/'
x = ['x=%d' % x for x in range(0,13)]
ns = []
for i in range(0,13):
    n = [x[i] + '_' + j for j in maskfolders[i]]
    ns.append(n)
names = []
for i in ns:
    for j in i:
        j = j + '.jpg'
        names.append(j)

out_dirs = [os.path.join(outfolder, f) for f in names]
for f in range(0,len(img_out)):
    cv2.imwrite(out_dirs[f], img_out[f])
    
    
    
