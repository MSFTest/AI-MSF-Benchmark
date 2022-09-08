#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:37:53 2017

@author: sulekahraman
"""
import os

import fire
from natsort import natsorted
from tqdm import tqdm

from common import my_utils
from fog_codes_public import SimplexNoise

"""
AVERAGE BETA OVER THE RAYYY

===============================================
Simplex Noise for Heterogenous Fog with Pinhole Camera Model
===============================================

Papers used: 
1. Haze Visibility Enhancement: A Survey and Quantitative Benchmarking
 Yu Li, Shaodi You, Michael S. Brown, and Robby T. Tan,

2. Vision and the Atmosphere
SRINIVASA G. NARASIMHAN AND SHREE K. NAYAR

3. Simplex noise demystified, Stefan Gustavson

Intrinsic calibration matrix of vkitti data from
http://www.europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds
       [[725,  0, 620.5],
    K = [  0,725, 187.0],
        [  0,  0,     1]] 

"""

# ------------------------------------------------------------
# MODULES
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


def _cal_vis(level):
    Ct = 0.05
    if level == "strong":
        beta = 0.0581
    elif level == "moderate":
        beta = 0.0374
    elif level == "low":
        beta = 0.0213
    else:
        raise ValueError()
    # cal visbility
    vis = -np.log(Ct) / beta
    return beta, vis


def fog(output, level, rgb_input=None, depth_input=None):
    assert level in ["strong", "moderate", "low"]
    beta, vis = _cal_vis(level)
    print("visibility", vis)
    rgb_input_dir = "./data/source/kitti/data_object/training/image_2"
    depth_input_dir = "./data/source/kitti/data_object/training/image_2/depth"
    # if "depth" in output:
    #     rgb_input_dir = "./data_depth/source/kitti/data_object/training/image_2"
    #     depth_input_dir = "./data_depth/source/kitti/data_object/training/image_2/depth"
    # elif "tracking" in output:
    #     rgb_input_dir = "./data_tracking/source/kitti/data_object/training/image_2"
    #     depth_input_dir = "./data_tracking/source/kitti/data_object/training/image_2/depth"
    # else:
    #     rgb_input_dir = "./data_object/source/kitti/data_object/training/image_2"
    #     depth_input_dir = "./data_object/source/kitti/data_object/training/image_2/depth"
    if rgb_input is not None:
        rgb_input_dir = rgb_input
    if depth_input is not None:
        depth_input_dir = depth_input
    print(output)
    print(depth_input_dir)
    rgb_input_fns = os.listdir(rgb_input_dir)
    rgb_input_fns = natsorted(rgb_input_fns)

    # depth_input_fns = os.listdir(depth_input_dir)
    # depth_input_fns = natsorted(depth_input_fns)
    for fn in tqdm(rgb_input_fns):
        fn1 = os.path.join(rgb_input_dir, fn)
        fn2 = os.path.join(depth_input_dir, fn)
        output_path = os.path.join(output, fn)
        image = cv2.imread(fn1)
        depth = cv2.imread(fn2, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        # print(depth.shape)
        depth = my_utils.fill_depth(depth, image.shape[0], image.shape[1])
        # print(depth.shape)
        if np.max(depth) > 255:
            depth = depth * 0.01  # convert to meters because beta is in meters
        # if not is_tracking:
        #     depth = depth * 0.01  # convert to meters because beta is in meters
        # -----------------------------------------------------------------
        # IMPLEMENTATION - ATMOSPHERIC SCATTERING MODEL WITH SIMPLEX NOISE
        # AND PINHOLE CAMERA MODEL
        # -----------------------------------------------------------------

        # PARAMETERS
        height, width = image.shape[:2]
        LInf = np.array([250, 255, 255])  # the brightest point in the image

        k = 0.5  # just a coefficient for the noise
        fu_inv = 1.0 / 725  # 725 - focal length of the camera in pixels (from vitti dataset)
        fv_inv = 1.0 / 725

        # image coordinates, (0,0) at the center of the image
        x_ = np.linspace(0., width, width, endpoint=False) - 620.5
        y_ = np.linspace(0., height, height, endpoint=False) + 187.0

        y_, x_ = np.meshgrid(y_, x_, indexing='ij')

        # ---------------------------------------------------------
        # SIMPLEX NOISE
        # ---------------------------------------------------------
        depthN = 10
        noise = np.zeros_like(x_)
        simplex = SimplexNoise.SimplexNoise()  # initialize once so that we're moving in the same volume of noise
        simplex.setup(depth)
        for i in range(depthN):
            Z = depth * i / depthN
            X = Z * x_ * fu_inv  # PINHOLE MODEL:image coords(x,y) to space coords (X,Y,Z)
            Y = Z * y_ * fv_inv
            noise += simplex.noise3d(X / 2000., Y / 2000., Z / 2000.) / depthN

        #
        transmission_noise = np.zeros_like(image, dtype=np.float64)
        direct_trans_noise = np.zeros_like(image)
        airlight_noise = np.zeros_like(image)

        beta_noise_ave = beta * (1 + k * noise)

        # ---------------------------------------------------------
        # HETEROGENEOUS FOG WITH 3D NOISE
        # ---------------------------------------------------------
        transmission_noise[:, :, 0] = np.exp(-beta_noise_ave * depth)
        transmission_noise[:, :, 1] = np.exp(-beta_noise_ave * depth)
        transmission_noise[:, :, 2] = np.exp(-beta_noise_ave * depth)
        direct_trans_noise = image * transmission_noise
        airlight_noise = LInf * (1 - transmission_noise)
        foggy_noise = direct_trans_noise + airlight_noise
        foggy_noise = np.asarray(foggy_noise, dtype=np.uint8)

        image_with_fog = foggy_noise

        # # ---------------------------------------------------------
        # # HOMOGENEOUS FOG
        # # ---------------------------------------------------------
        # direct_trans = np.zeros_like(image)
        # airlight = np.zeros_like(image)
        # transmission = np.zeros_like(image, dtype=np.float64)
        #
        # transmission[:, :, 0] = np.exp(-beta * depth)
        # transmission[:, :, 1] = np.exp(-beta * depth)
        # transmission[:, :, 2] = np.exp(-beta * depth)
        # direct_trans = image * transmission
        # airlight = LInf * (1 - transmission)
        # foggy = direct_trans + airlight
        # foggy = np.asarray(foggy, dtype=np.uint8)

        # diff_nois = (beta_noise_ave - beta)
        # print(diff_nois.max(), diff_nois.min())
        #    print(diff_nois)
        #    print(depth[55, 790:800])

        # ---------------------------------------------------------
        # PLOTS
        # ---------------------------------------------------------
        # print(np.min(foggy_noise), np.max(foggy_noise), foggy_noise.dtype)
        foggy_noise = foggy_noise[..., ::-1]
        # plt.figure(1)
        # plt.imshow(foggy_noise)
        # plt.show()
        plt.imsave(output_path, foggy_noise)


if __name__ == "__main__":
    # fog("", "strong")
    fire.Fire(fog)
