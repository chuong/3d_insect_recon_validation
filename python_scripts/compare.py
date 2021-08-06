# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:42:40 2020

@author: Jiayuan Liu

This script compares the focus stacking results with ground truth images

"""

from skimage.measure import compare_ssim
from matplotlib import pyplot as plt
import cv2
import os, glob

# read in focus stacking results
inputFolder = 'E:/ANU/ENGN4200/synthetic_images/test_7/Theo/y=%d'
extensions = ['*.png', '*.jpg']
files = []
for i in range(2,24,4):
    files.append([file for ext in extensions for file in glob.glob(os.path.join(inputFolder % i, ext))])

image_BGRs = [cv2.imread(file[0]) for file in files]

grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in image_BGRs]

# read in ground truth images
references = 'E:/ANU/ENGN4200/synthetic_images/test_7/truth/x=5/y=%d'

ref_dir = []
for i in range(2,24,4):
    ref_dir.append([file for ext in extensions for file in glob.glob(os.path.join(references % i, ext))])
ref_imgs = [cv2.imread(file[0]) for file in ref_dir]
ref_grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in ref_imgs]    

# compare the two using SSIM algorithm
# score from -1 to 1, the closer to 1, the more similar two images are
# diff shows the differences between two images
scores = []
diffs = []
for i in range(0, len(grays)):
    (score, diff) = compare_ssim(grays[i], ref_grays[i], full=True)
    diff = (diff * 255).astype("uint8")
    scores.append(score)
    diffs.append(diff)

plt.imshow(diffs[0], cmap='rainbow')

