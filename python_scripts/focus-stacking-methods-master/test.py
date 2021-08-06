# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:37:56 2020

@author: Jiayuan Liu
"""

import os, glob
import cv2
import numpy as np
import logging
from matplotlib import pyplot as plt

#outputdir = 'D:/ANU/ENGN4200/synthetic_images/calibration/ver2/'
inputdir = 'E:/ANU/ENGN4200/synthetic_images/test_3/Angle 0/'
#califolder = 'D:/ANU/ENGN4200/synthetic_images/calibration/ORB/'

inputFolder = inputdir
#outputFolder = outputdir
#os.makedirs(outputFolder)

extensions = ['*.png', '*.jpg']
files = [file for ext in extensions for file in glob.glob(os.path.join(inputFolder, ext))]
def num(x):
    if 'p' in x[-7:-4]:
        y = x[-7:-4].replace('p', ' ')
    else:
        y = x[-7:-4]
    return(y)

files.sort(key = num)   
# remove fusion image
files = [file for file in files if 'fusion' not in file]
logging.info('Found:\n' + '\n'.join(files))

image_BGRs = [cv2.imread(file) for file in files]

#H_file = os.path.join(califolder, 'Hs_%s_%s.npz' % ('ORB', 'pyramid'))
#homographies = []
#affines = []
#if os.path.isfile(H_file):
    #homographies = np.load(H_file)['homographies']
    #affines = np.load(H_file)['affines']


#%%

grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in image_BGRs]
#base_index = len(grays)//2
#base_gray = grays[base_index]
#criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,
#                1e-10)
#homographies = [cv2.findTransformECC(gray, base_gray,
#                                         np.eye(3, dtype=np.float32),
#                                         cv2.MOTION_HOMOGRAPHY, criteria, inputMask=None, gaussFiltSize=5)[1]
#                    for gray in grays]
max_pnts = 128
base_index = len(grays)//2
base_gray = grays[base_index]
#if feature.upper() == 'ORB':
#    detector = cv2.ORB_create(max_pnts)  # tested with OpenCV 3.2.0
#    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#elif feature.upper() == 'SIFT':
detector = cv2.SIFT_create(max_pnts)  # tested with OpenCV 3.2.0
matcher = cv2.BFMatcher()
iterations=1000
epsilon=1e-6
base_kpnt, base_desc = detector.detectAndCompute(base_gray, None)
homographies = []

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,
                epsilon)
for i, gray in enumerate(grays):
    if i == base_index:
        H = np.eye(3, dtype=np.float32)
    else:
        curr_kpnt, curr_desc = detector.detectAndCompute(gray, None)
        matches = matcher.match(curr_desc, base_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        if i == 40:
            final_img = cv2.drawMatches(gray, curr_kpnt, base_gray, base_kpnt, matches, None)
            plt.imshow(final_img,cmap='gray')
#            matches = matches[:max_pnts]
        src_pts = np.zeros([len(matches), 1, 2], dtype=np.float32)
        dst_pts = np.zeros([len(matches), 1, 2], dtype=np.float32)
        for j in range(len(matches)):
            src_pts[j] = curr_kpnt[matches[j].queryIdx].pt
            dst_pts[j] = base_kpnt[matches[j].trainIdx].pt
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=2.0)
        except:
            print(i)
            H = cv2.findTransformECC(gray, base_gray,
                                         np.eye(3, dtype=np.float32)[:2, :3],
                                         cv2.MOTION_AFFINE, criteria, inputMask=None, gaussFiltSize=5)[1]
        if H is None:
            print(i)
            H = cv2.findTransformECC(gray, base_gray,
                                         np.eye(3, dtype=np.float32)[:2, :3],
                                         cv2.MOTION_AFFINE, criteria, inputMask=None, gaussFiltSize=5)[1]
        elif np.isnan(H).any():
            print(i)
            H = cv2.findTransformECC(gray, base_gray,
                                         np.eye(3, dtype=np.float32)[:2, :3],
                                         cv2.MOTION_AFFINE, criteria, inputMask=None, gaussFiltSize=5)[1]
            
    homographies.append(H.astype(np.float32))

row = np.array((0,0,1)).reshape((1,3))
for i in range(0,len(homographies)):
    if np.shape(homographies[i]) == (2,3):
        a = np.vstack((homographies[i],row))
        homographies[i] = a
#homographies = fix_homography(homographies)


