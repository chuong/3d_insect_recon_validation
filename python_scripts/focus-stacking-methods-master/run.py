#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:35:56 2019

@author: pi
"""

import os, glob
import alignments
import fusions
import pyramid
import cv2
import numpy as np
import logging
import argparse
import math

#def num(x):
#    if 'p' in x[-7:-4]:
#        y = x[-7:-4].replace('p', ' ')
#    else:
#        y = x[-7:-4]
#    return(y)

def focus_stacking(inputFolder, outputFolder, califolder, scale, align_method, fuse_method, display):

    extensions = ['*.png', '*.jpg']
    files = [file for ext in extensions for file in glob.glob(os.path.join(inputFolder, ext))]
    # files.sort(key = num) 
    # remove fusion image
    files = [file for file in files if 'fusion' not in file]
    logging.info('Found:\n' + '\n'.join(files))

    image_BGRs = [cv2.imread(file) for file in files]
    if scale != 1.0:
        image_BGRs = [cv2.resize(img, (0, 0), fx=scale, fy=scale)
                      for img in image_BGRs]

    # load precomputed homographies if available
    H_file = os.path.join(califolder, 'Hs_%s_%s.npz' % (align_method, fuse_method))
    homographies = []
    affines = []
    if os.path.isfile(H_file):
        homographies = np.load(H_file)['homographies']
        affines = np.load(H_file)['affines']

    # compute homographies, if not available, and align images
    # once computed homographies can be reused for similar stacks
    if align_method.upper() == 'ECC_AFFINE': # recommended but slowest
        aligned_BGRs, affines = alignments.align_images(
                image_BGRs, 'ECC_AFFINE', affines)
    elif align_method.upper() == 'ECC_HOMOGRAPHY': # recommended but slowest
        aligned_BGRs, homographies = alignments.align_images(
                image_BGRs, 'ECC_HOMOGRAPHY', homographies)
    elif align_method.upper() == 'ECC_PYRAMID':
        aligned_BGRs, homographies = alignments.align_images(image_BGRs, 'ECC_PYRAMID', homographies)
    elif align_method.upper() == 'ORB':
        aligned_BGRs, homographies = alignments.align_images(image_BGRs, 'ORB', homographies)
    elif align_method.upper() == 'HYBRID':
        aligned_BGRs, homographies = alignments.align_images(image_BGRs, 'HYBRID', homographies)
    elif align_method.upper() == 'THEORETICAL':
        aligned_BGRs, homographies = alignments.align_images(image_BGRs, 'THEORETICAL', homographies)

    # save homographies if not yet done
    if not os.path.isfile(H_file):
        np.savez(H_file, align_method=align_method, fuse_method=fuse_method,
                 homographies=homographies, affines=affines)

    # methods of fusion with different speed and quality
    if fuse_method == 'pyramid': # recommended
        F = pyramid.get_pyramid_fusion(np.asarray(aligned_BGRs))
    elif fuse_method == 'guided_filter':
        F = fusions.fuse_guided_filter(aligned_BGRs)
    else:  # simple
        fuse_method = 'simple'
        F = fusions.fuse_simple(aligned_BGRs)

#    cv2.imwrite(os.path.join(folder, 'fusion_simple.jpg'), F)
#    cv2.imwrite(os.path.join(folder, 'fusion_guidedfilter.jpg'), F)
    fused_filename = 'fusion_%s_%s.jpg' % (align_method, fuse_method)
    cv2.imwrite(os.path.join(outputFolder, fused_filename), F)

    if display:

    #    cv2.imshow('im0', image_BGRs[0])
    #    cv2.imshow('im1', image_BGRs[1])
    #    cv2.imshow('im2', image_BGRs[2])
    #    cv2.imshow('im3', image_BGRs[3])
        cv2.imshow('aim0', aligned_BGRs[0])
        cv2.imshow('aim1', aligned_BGRs[1])
        cv2.imshow('aim2', aligned_BGRs[2])
        cv2.imshow('aim3', aligned_BGRs[3])
        cv2.imshow('fusion', F)
        cv2.waitKey()
        cv2.destroyAllWindows()

    logging.info('Focus stacking done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Query the status of nodes in a Graph of processes.')
    parser.add_argument('--input', '-i', metavar='FOLDER', type=str,
                        help='Folder with input images')
    parser.add_argument('--output', '-o', metavar='FOLDER', type=str, default=None,
                        help='Folder to create output images')
    parser.add_argument('--downscale', metavar='DOWNSCALE', type=float, default=1.0,
                        help='Downscale the image')
    parser.add_argument('--align_method', type=str, default='THEORETICAL',
                        help='Method to align images.',
                        choices=['ECC_AFFINE', 'ECC_HOMOGRAPHY', 'ECC_PYRAMID', 'ORB', 'HYBRID','THEORETICAL'])
    parser.add_argument('--fuse_method', type=str, default='pyramid',
                        help='Method to combine multi-focus images.',
                        choices=['pyramid', 'guided_filter', 'simple'])
    parser.add_argument('--verbose', '-v', help='Verbosity level', default='warning',
                        choices=['fatal', 'error', 'warning', 'info', 'debug', 'trace'])
    parser.add_argument('--display', help='Display results',
                        action='store_true')
    args = parser.parse_args()

    logStringToPython = {
        'fatal': logging.FATAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
        'trace': logging.DEBUG,
    }
    logging.getLogger().setLevel(logStringToPython[args.verbose])
    
    #outputdir = 'D:/ANU/ENGN4200/synthetic_images/change_d2/step 0.2/no_calibration/stacked angle %d'
    #inputdir = 'D:/ANU/ENGN4200/synthetic_images/change_d2/step 0.2/Angle %d'
    #calidir = 'D:/ANU/ENGN4200/synthetic_images/change_d2/step 0.2/no_calibration/stacked angle %d'
    
    outputdir = 'E:/ANU/ENGN4200/synthetic_images/test/calibration/ORB/'
    inputdir = 'E:/ANU/ENGN4200/synthetic_images/test/calibration/'
    califolder = 'E:/ANU/ENGN4200/synthetic_images/test/calibration/ORB/'

    #for i in range(0,360,90):
    inputFolder = inputdir #% i
    outputFolder = outputdir #% i
    #califolder = calidir #% i
    os.makedirs(outputFolder)
    focus_stacking(inputFolder, outputFolder, califolder, args.downscale, args.align_method, args.fuse_method, args.display)
    
    #distance = -0.124
    #outputdir = 'E:/ANU/ENGN4200/synthetic_images/test/theo/height %d rotate %d'
    #inputdir = 'E:/ANU/ENGN4200/synthetic_images/test/height %d rotate %d'
    #califolder = 'E:/ANU/ENGN4200/synthetic_images/test/height 0 rotate 0'
    #outputdir = '/flush5/liu220/images/stack_4k/height %d rotate %d'
    #inputdir = '/flush5/liu220/images/frontlight_4k/height %d rotate %d'
    #califolder = '/flush5/liu220/4k_homography/'
    #for j in range (0,10,15): #(-90,90,10)
        #deg = (90-abs(j)) / 10 * 4
        #if deg == 0:
            #deg = 1
        #gap = int(360//deg)
        #height = -(distance * 1000) * math.sin(-j*math.pi/180)
        #for i in range(0,140,130):
            #inputFolder = inputdir % (height, i)
            #outputFolder = outputdir % (height, i)
            #os.makedirs(outputFolder)
            #focus_stacking(inputFolder, outputFolder, califolder, args.downscale, args.align_method, args.fuse_method, args.display)
