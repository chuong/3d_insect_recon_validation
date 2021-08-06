#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:44:12 2019

@author: chuong nguyen <chuong.nguyen@csiro.au>
"""

import cv2
import numpy as np
from timed import timed


@timed
def align_images(image_BGRs, algorithm='ECC', warpMatrices=[], base='MIDDLE'):
    '''Compute homography if missing and align image by warping.

    INPUT:
        - BGRs: list of BGR images
        - algorithm='ECC': ('ECC', 'ORB', 'SIFT', 'SURF')
        - homographies=[]: list of known homographies if available to speed up
        - base='MIDDLE': ('MIDDLE', 'START', 'END') position of reference image

    OUTPUT:
        - aligned_images: aligned images
        - holographies: list of homographies
    '''
    grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in image_BGRs]
#    grays = [cv2.blur(gray, (5, 5)) for gray in grays] # blur to smooth noise
    if len(warpMatrices) == 0:
        if algorithm.upper() == 'ECC_AFFINE':
            warpMatrices = get_affine_ECC(grays)
        elif algorithm.upper() == 'ECC_HOMOGRAPHY':
            warpMatrices = get_homography_ECC(grays)
        elif algorithm.upper() == 'ECC_PYRAMID':
            warpMatrices = get_homography_ECC_pyramid(grays)
        elif algorithm.upper() in ['ORB', 'SIFT', 'SURF']:
            warpMatrices = get_homography_feature(grays, algorithm)
        elif algorithm.upper() == 'HYBRID':
            warpMatrices = get_homography_hybrid(grays)
        elif algorithm.upper() == 'THEORETICAL':
            warpMatrices = theoretical_homography(grays)
        else:
            raise Exception('Invalid algorithm %s' % algorithm)

    if 'AFFINE' in algorithm:
        warped_images = warp_image_affine(image_BGRs, warpMatrices)
    else:
        warped_images = warp_image_homography(image_BGRs, warpMatrices)

    return warped_images, warpMatrices


@timed
def get_affine_ECC(grays, iterations=1000, epsilon=1e-6):
    '''Computer a list of affine transformations from list of images

    INPUT:
        - grays: list of gray images
        - iterations=5000: number of iterations for ECC algorithm
        - epsilon=1e-10: threshold for ECC algorithm

    OUTPUT:
        - homographies: list of homographies
    '''
    base_index = len(grays)//2
    base_gray = grays[base_index]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,
                epsilon)
    affines = [cv2.findTransformECC(gray, base_gray,
                                    np.eye(3, dtype=np.float32)[:2, :3],
                                    cv2.MOTION_AFFINE, criteria, inputMask=None, gaussFiltSize=5)[1]
               for gray in grays]
    return affines


@timed
def get_homography_ECC(grays, iterations=1000, epsilon=1e-6):
    '''Computer a list of homography transformation from list of images

    INPUT:
        - grays: list of gray images
        - iterations=5000: number of iterations for ECC algorithm
        - epsilon=1e-10: threshold for ECC algorithm

    OUTPUT:
        - homographies: list of homographies
    '''
    base_index = len(grays)//2
    base_gray = grays[base_index]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,
                epsilon)
    homographies = [cv2.findTransformECC(gray, base_gray,
                                         np.eye(3, dtype=np.float32),
                                         cv2.MOTION_HOMOGRAPHY, criteria, inputMask=None, gaussFiltSize=5)[1]
                    for gray in grays]
    
    return homographies

def theoretical_homography(grays):
    
    # create 3x3 homography matrices
    homographies = np.zeros((len(grays),3,3))
    homographies[:,2,2] = 1
    
    # parameters in blender:
    i = np.arange(0,61,1) # number of images in a stack
    f = 65/1000 # focal length in m
    d = 0.2 # distance of camera to object in m
    ft = 0.024 # forward translation of camera in m
    bt = -0.019 # backward translation of camera in m
    
    # camera location and focal length after translation
    max_d1 = d + ft
    min_d1 = d + bt
    max_f = min_d1 * f / (min_d1 - f)
    min_f = max_d1 * f / (max_d1 - f)
    
    # change in focal length in each image in the stack
    f_step = (max_f - min_f) / (len(i)-1)
    # distance of lens to image sensor for each image
    d2 = max_f - i * f_step

    base_index = np.size(i)//2
    scales = np.zeros((1,np.size(i)))
    x = np.zeros((1,np.size(i)))
    y = np.zeros((1,np.size(i)))
    
    x_off = np.shape(grays[0])[1]/2
    y_off = np.shape(grays[0])[0]/2
    
    for n in range(0,np.size(i)):
        # scale change of each image with respect to middle image
        scales[0,n] = d2[base_index] / (max_f - n * f_step)
        # offset alignment to image centre
        x[0,n] = (-scales[0,n] * x_off) + x_off
        y[0,n] = (-scales[0,n] * y_off) + y_off
     
    homographies[:,0,0] = scales
    homographies[:,1,1] = scales
    homographies[:,0,2] = x
    homographies[:,1,2] = y
    
    homographies = list(homographies)
    
    return homographies

@timed
def get_homography_ECC_pyramid(grays, iterations=1000, epsilon=1e-6):
    '''Computer a list of homography from list of images

    INPUT:
        - grays: list of gray images
        - iterations=5000: number of iterations for ECC algorithm
        - epsilon=1e-10: threshold for ECC algorithm

    OUTPUT:
        - homographies: list of homographies
    '''
    base_index = len(grays)//2
    no_levels = 4
    pyramid = []
    for l in range(no_levels):
        pyramid.append([cv2.resize(gray, (0, 0), fx=0.5**l, fy=0.5**l)
                       for gray in grays])

    scale_matrix = np.array([[1, 1, 2], [1, 1, 2], [1, 1, 1]],
                            dtype=np.float32)
    for lev in range(no_levels - 1, -1, -1):
        grays_p = pyramid[lev]
        print(lev)
        if lev == (no_levels - 1):
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        iterations, epsilon)
            homographies = [cv2.findTransformECC(gray_p, grays_p[base_index],
                                                 np.eye(3, dtype=np.float32),
                                                 cv2.MOTION_HOMOGRAPHY,
                                                 criteria, inputMask=None, gaussFiltSize=5)[1]
                            for gray_p in grays_p]
        else:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        iterations, epsilon)
            homographies = [cv2.findTransformECC(gray_p, grays_p[base_index],
                                                 H_p*scale_matrix,
                                                 cv2.MOTION_HOMOGRAPHY,
                                                 criteria, inputMask=None, gaussFiltSize=5)[1]
                            for (gray_p, H_p) in zip(grays_p, homographies)]

    return homographies


@timed
def get_homography_hybrid(grays, iterations=1000, epsilon=1e-6):
    '''Computer a list of homography from list of images

    INPUT:
        - grays: list of gray images
        - iterations=5000: number of iterations for ECC algorithm
        - epsilon=1e-10: threshold for ECC algorithm

    OUTPUT:
        - homographies: list of homographies
    '''
    homographies_init = get_homography_feature(grays, 'ORB')
    base_index = len(grays)//2
    base_gray = grays[base_index]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,
                epsilon)
    homographies = [cv2.findTransformECC(gray, base_gray, H,
                                         cv2.MOTION_HOMOGRAPHY, criteria, inputMask=None, gaussFiltSize=5)[1]
                    for gray, H in zip(grays, homographies_init)]
    return homographies

def get_homography_feature(grays, feature, max_pnts=128):
    '''Compute homography using feature matching.
    Only support ORB and SIFT features for now.
    Not working properly.
    '''
    base_index = len(grays)//2
    base_gray = grays[base_index]
    if feature.upper() == 'ORB':
        detector = cv2.ORB_create(max_pnts)  # tested with OpenCV 3.2.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif feature.upper() == 'SIFT':
        detector = cv2.SIFT_create(max_pnts)  # tested with OpenCV 3.2.0
        matcher = cv2.BFMatcher()

    base_kpnt, base_desc = detector.detectAndCompute(base_gray, None)
    homographies = []
    for i, gray in enumerate(grays):
        if i == base_index:
            H = np.eye(3, dtype=np.float32)
        else:
            curr_kpnt, curr_desc = detector.detectAndCompute(gray, None)
            matches = matcher.match(curr_desc, base_desc)
            matches = sorted(matches, key=lambda x: x.distance)
#            matches = matches[:max_pnts]
            src_pts = np.zeros([len(matches), 1, 2], dtype=np.float32)
            dst_pts = np.zeros([len(matches), 1, 2], dtype=np.float32)
            for i in range(len(matches)):
                src_pts[i] = curr_kpnt[matches[i].queryIdx].pt
                dst_pts[i] = base_kpnt[matches[i].trainIdx].pt
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                         ransacReprojThreshold=2.0)
        homographies.append(H.astype(np.float32))
    return homographies


@timed
def get_homography_feature(grays, feature, max_pnts=128):
    '''Compute homography using feature matching.
    Only support ORB feature for now.
    Not working properly.
    '''
    base_index = len(grays)//2
    base_gray = grays[base_index]
    if feature.upper() == 'ORB':
        detector = cv2.ORB_create(max_pnts)  # tested with OpenCV 3.2.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    base_kpnt, base_desc = detector.detectAndCompute(base_gray, None)
    homographies = []
    for i, gray in enumerate(grays):
        if i == base_index:
            H = np.eye(3, dtype=np.float32)
        else:
            curr_kpnt, curr_desc = detector.detectAndCompute(gray, None)
            matches = matcher.match(curr_desc, base_desc)
            matches = sorted(matches, key=lambda x: x.distance)
#            matches = matches[:max_pnts]
            src_pts = np.zeros([len(matches), 1, 2], dtype=np.float32)
            dst_pts = np.zeros([len(matches), 1, 2], dtype=np.float32)
            for i in range(len(matches)):
                src_pts[i] = curr_kpnt[matches[i].queryIdx].pt
                dst_pts[i] = base_kpnt[matches[i].trainIdx].pt
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                         ransacReprojThreshold=2.0)
        homographies.append(H.astype(np.float32))
    return homographies


@timed
def warp_image_affine(image_BGRs, affines):
    '''Warp images to align images from given homographies
    '''
    warped_BGRs = [cv2.warpAffine(
            BGR, H, BGR.shape[1::-1],
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for BGR, H in zip(image_BGRs, affines)]
    return warped_BGRs


@timed
def warp_image_homography(image_BGRs, homographies):
    '''Warp images to align images from given homographies
    '''
    warped_BGRs = [cv2.warpPerspective(
            BGR, H, BGR.shape[1::-1],
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) # borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255)
        for BGR, H in zip(image_BGRs, homographies)]
    return warped_BGRs
