#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:37:53 2017

@author: sulekahraman
"""
from common import my_utils

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
import SimplexNoise
import time

plt.ion()

if __name__ == "__main__":
    # ------------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------------
    fn1, fn2 = "rgb_00001", "depth_00001"
    image = cv2.imread("samples/{}.png".format(fn1))
    # a pixel intensity of 1 in our single channel PNG16 depth images corresponds
    # to a distance of 1cm to the camera plane. The depth map is in centimeters.
    depth = cv2.imread("samples/{}.png".format(fn2), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = my_utils.fill_depth(depth, image.shape[0], image.shape[1])

    depth = depth * 0.01  # convert to meters because beta is in meters
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
    print("Computing simplex noise (%d steps)" % depthN)
    simplex = SimplexNoise.SimplexNoise()  # initialize once so that we're moving in the same volume of noise
    simplex.setup(depth)
    for i in range(depthN):
        print(" %d" % i)
        Z = depth * i / depthN
        X = Z * x_ * fu_inv  # PINHOLE MODEL:image coords(x,y) to space coords (X,Y,Z)
        Y = Z * y_ * fv_inv
        noise += simplex.noise3d(X / 2000., Y / 2000., Z / 2000.) / depthN

    #
    transmission_noise = np.zeros_like(image, dtype=np.float64)
    direct_trans_noise = np.zeros_like(image)
    airlight_noise = np.zeros_like(image)

    # beta_max = 100
    # beta_min = 0
    # steps = 10
    # betas = np.linspace(beta_min, beta_max, steps, endpoint=False)[::-1]
    # for i in betas:
    #     beta = i / 1000
    t0 = time.time()
    betas = [0.0581, 0.0374, 0.0032, 0.008, 0.08]
    # cal visbility
    Ct = 0.05

    for beta in betas:
        vis = -np.log(Ct) / beta
        print("beta %.4f max visibility %.4f" % (beta, vis))
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

        plt.figure(1)
        plt.imshow(foggy_noise[..., ::-1] / 255.)
        plt.show()
        # plt.draw()
        # plt.waitforbuttonpress(0)

        # SAVE IMAGES
        # cv2.imwrite('noise_map_b=80_k=0.5_n=100.jpg', noise_map)
        # cv2.imwrite('noisy_images/' + str(folder) + '/noisy_k=0.5_n=100_b=' + str(i) + '_' + str(folder) + '.png',
        #             foggy_noise)
        # cv2.imwrite('comparison_k=0.5_n=100.jpg', imvisu)

    t1 = time.time()

    print((t1 - t0) / len(betas), ' per image')
