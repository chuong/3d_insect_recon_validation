# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:48:26 2020

@author: Jiayuan Liu
"""


import numpy as np
from sklearn.linear_model import LinearRegression

affine_file = np.load('E:/ANU/ENGN4200/synthetic_images/test_7/calibration/ECC_A/Hs_ECC_AFFINE_pyramid.npz')

ECC_affines = affine_file['affines']

pic_index = np.array(range(0,len(ECC_affines)))

xsa = ECC_affines[:,0,0]
ysa = ECC_affines[:,1,1]
xta = ECC_affines[:,0,2]
yta = ECC_affines[:,1,2]


#%% Linearly fit the homography matrices

model_x = LinearRegression().fit(np.arange(30,70,1).reshape((-1,1)), xsa[30:70])
predictions_x = model_x.predict(pic_index.reshape((-1,1)))

model_y = LinearRegression().fit(np.arange(30,70,1).reshape((-1,1)), ysa[30:70])
predictions_y = model_y.predict(pic_index.reshape((-1,1)))

model_tx = LinearRegression().fit(np.arange(30,70,1).reshape((-1,1)), xta[30:70])
predictions_tx = model_tx.predict(pic_index.reshape((-1,1)))

model_ty = LinearRegression().fit(np.arange(30,70,1).reshape((-1,1)), yta[30:70])
predictions_ty = model_ty.predict(pic_index.reshape((-1,1)))


#%% Save the new homography matrices

H = np.zeros(ECC_affines.shape)
H[:,0,1] = 0
#H[:,2,1] = 0
H[:,1,0] = 0
#H[:,2,0] = 0
H[:,0,0] = predictions_x
H[:,1,1] = predictions_y
H[:,0,2] = predictions_tx
H[:,1,2] = predictions_ty

np.savez('Hs_ECC_AFFINE_pyramid', align_method='ECC_AFFINE', fuse_method='pyramid',
                 homographies=[], affines=H)


















