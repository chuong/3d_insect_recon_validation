# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:51:47 2021

@author: Jiayuan Liu

Modified from: https://github.com/evo-biomech/scAnt/blob/master/scripts/estimate_camera_positions.py

This script generate the sfm file using poses provided 

"""

import numpy as np
import os


def tab_x_REALTAB(n):
    # produces real tabs
    string = "\n"
    for i in range(n):
        string += "\t"
    return string


def tab_x(n):
    # produces 4 spaces instead of tabs
    string = "\n"
    for i in range(n):
        string += "    "
    return string


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def generate_sfm(T, imgname, use_cutouts=True):

    out = open('E:/ANU/ENGN4200/synthetic_images/pose_est/fixed.sfm', "w+")
    out.write("{")
    out.write(tab_x(1) + '"version": [')
    out.write(tab_x(2) + '"1",')
    out.write(tab_x(2) + '"0",')
    out.write(tab_x(2) + '"0"')
    out.write(tab_x(1) + '],')

    out.write(tab_x(1) + '"views": [')

    viewId = 10000000
    intrinsicId = 1000000000
    resectionId = 0
    locked = 1

    meshroom_img_dir = "E:\/ANU\/ENGN4200\/synthetic_images\/insect_3\/meshroom_Theo\/"
    meshroom_img_path = [meshroom_img_dir + i for i in imgname]

    """
    Enter all cameras and assign IDs to reference for each pose
    """

    for cam in range(len(T)):
        out.write(tab_x(2) + '{')
        viewId += 1
        resectionId += 1
        out.write(tab_x(3) + '"viewId": "' + str(viewId) + '",')
        out.write(tab_x(3) + '"poseId": "' + str(viewId) + '",')
        out.write(tab_x(3) + '"intrinsicId": "' + str(intrinsicId) + '",')
        out.write(tab_x(3) + '"resectionId": "' + str(resectionId) + '",')
        out.write(tab_x(3) + '"path": "' + meshroom_img_path[cam] + '",')
        out.write(tab_x(3) + '"width": "4320",')
        out.write(tab_x(3) + '"height": "2880",')
        out.write(tab_x(3) + '"metadata": {')
        out.write(tab_x(4) + '"AliceVision:SensorWidth": "36.000000",')
        out.write(tab_x(4) + '"Exif:FocalLength": "65",')
        out.write(tab_x(4) + '"Exif:FocalLengthIn35mmFilm": "98",')
        out.write(tab_x(4) + '"FNumber": "2.8",')
        out.write(tab_x(4) + '"Make": "Canon",')
        out.write(tab_x(4) + '"Model": "Canon EOS 5DS",')
        #out.write(tab_x(4) + '"Orientation": "1",')
        #out.write(tab_x(4) + '"PixelAspectRatio": "1",')
        out.write(tab_x(4) + '"ResolutionUnit": "in",')
        out.write(tab_x(4) + '"XResolution": "96",')
        out.write(tab_x(4) + '"YResolution": "96",')
        out.write(tab_x(4) + '"jpeg:subsampling": "4:2:0",')
        out.write(tab_x(4) + '"oiio:ColorSpace": "sRGB"')

        out.write(tab_x(3) + '}')

        out.write(tab_x(2) + '}')

        if cam != len(T) - 1:
            # add a comma to all but the last entry
            out.write(',')

    """
    Enter intrinsics of camera model(s)
    These values need to be calculated / estimated by Meshroom and entered here
    """

    out.write(tab_x(1) + '],')
    out.write(tab_x(1) + '"intrinsics": [')

    out.write(tab_x(2) + '{')
    out.write(tab_x(3) + '"intrinsicId": "' + str(intrinsicId) + '",')
    out.write(tab_x(3) + '"width": "4320",')
    out.write(tab_x(3) + '"height": "2880",')
    out.write(tab_x(3) + '"sensorWidth": "36",')
    out.write(tab_x(3) + '"sensorHeight": "24",')
    out.write(tab_x(3) + '"serialNumber": "E:\/ANU\/ENGN4200\/synthetic_images\/insect_3\/meshroom_Theo_Canon_Canon EOS 5DS",')
    out.write(tab_x(3) + '"type": "radial3",')
    out.write(tab_x(3) + '"initializationMode": "estimated",')
    out.write(tab_x(3) + '"pxInitialFocalLength": "11760",')
    out.write(tab_x(3) + '"pxFocalLength": "11760",')
    out.write(tab_x(3) + '"principalPoint": [')

    out.write(tab_x(4) + '"2160",')
    out.write(tab_x(4) + '"1440"')

    out.write(tab_x(3) + '],')
    out.write(tab_x(3) + '"distortionParams": [')

    out.write(tab_x(4) + '"0",')
    out.write(tab_x(4) + '"0",')
    out.write(tab_x(4) + '"0"')

    out.write(tab_x(3) + '],')
    out.write(tab_x(3) + '"locked": "' + str(locked) + '"')

    out.write(tab_x(2) + '}')

    """
    All estimated poses are 
    """

    out.write(tab_x(1) + '],')
    out.write(tab_x(1) + '"poses": [')

    # reset viewID to begin at the initial count and move sequentially through all cameras
    viewId = 10000000
    cam = 0
    for i in T:
        out.write(tab_x(2) + '{')
        viewId += 1
        out.write(tab_x(3) + '"poseId": "' + str(viewId) + '",')
        out.write(tab_x(3) + '"pose": {')
        out.write(tab_x(4) + '"transform": {')
        out.write(tab_x(5) + '"rotation": [')

        # rotation of camera
        out.write(tab_x(6) + '"' + str(round(i[0][0], ndigits=5)) + '",')
        out.write(tab_x(6) + '"' + str(round(i[0][1], ndigits=5)) + '",')
        out.write(tab_x(6) + '"' + str(round(i[0][2], ndigits=5)) + '",')

        out.write(tab_x(6) + '"' + str(round(i[1][0], ndigits=5)) + '",')
        out.write(tab_x(6) + '"' + str(round(i[1][1], ndigits=5)) + '",')
        out.write(tab_x(6) + '"' + str(round(i[1][2], ndigits=5)) + '",')

        out.write(tab_x(6) + '"' + str(round(i[2][0], ndigits=5)) + '",')
        out.write(tab_x(6) + '"' + str(round(i[2][1], ndigits=5)) + '",')
        out.write(tab_x(6) + '"' + str(round(i[2][2], ndigits=5)) + '"')

        out.write(tab_x(5) + '],')
        out.write(tab_x(5) + '"center": [')

        # translation of camera
        out.write(tab_x(6) + '"' + str(round(i[0][3], ndigits=5)) + '",')
        out.write(tab_x(6) + '"' + str(round(i[1][3], ndigits=5)) + '",')
        out.write(tab_x(6) + '"' + str(round(i[2][3], ndigits=5)) + '"')

        out.write(tab_x(5) + ']')
        out.write(tab_x(4) + '},')
        out.write(tab_x(4) + '"locked": "' + str(locked) + '"')
        out.write(tab_x(3) + '}')
        out.write(tab_x(2) + '}')

        if cam != len(T) - 1:
            # add a comma to all but the last entry
            out.write(',')
        cam += 1

    out.write(tab_x(1) + ']')
    out.write('\n}')

    out.close()


if __name__ == '__main__':

    T = np.load('E:/ANU/ENGN4200/synthetic_images/pose_est/fixed.npy')

    imgname = 'x=%d_y=%d.jpg'
    ns = []
    for i in range(0,13):
        if i == 0 or i == 12:
            n = imgname % (i, 0)
            ns.append(n)
        elif i == 1 or i == 11:
            for j in range(0,6):
                n = imgname % (i, j)
                ns.append(n)
        elif i == 2 or i == 10:
            for j in range(0,12):
                n = imgname % (i, j)
                ns.append(n)
        elif i == 3 or i == 9:
            for j in range(0,17):
                n = imgname % (i, j)
                ns.append(n)
        elif i == 4 or i == 8:
            for j in range(0,21):
                n = imgname % (i, j)
                ns.append(n)
        elif i == 5 or i == 7:
            for j in range(0,23):
                n = imgname % (i, j)
                ns.append(n)
        else:
            for j in range(0,24):
                n = imgname % (i, j)
                ns.append(n)
        
    
    generate_sfm(T, ns, use_cutouts=False)

    #exit()
