# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:43:07 2021

@author: Jiayuan Liu

This script changes the pose matrix generated from Blendeer to Meshroom coordinates

"""

import numpy as np

H = np.load('E:/ANU/ENGN4200/synthetic_images/pose_est/T_matrix.npy')

R_bcam2cv = np.array([[1,0,0],[0,0,1],[0,-1,0]])

i = H.shape[0]
R = H[:,0:3,0:3]
T = H[::,0:3,3]
T = np.resize(T, (i,3,1))


R_world2bcam = R
T_world2bcam = T

R_world2cv = np.matmul(-1*R_bcam2cv, R_world2bcam)
T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)

R_world2cv[:,:,0] = -R_world2cv[:,:,0]

RT = np.append(R_world2cv, T_world2cv, axis=2)
np.save('E:/ANU/ENGN4200/synthetic_images/pose_est/fixed.npy',RT)

