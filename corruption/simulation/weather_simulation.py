''''
@Project: AI-MSF-benchmark
@Description: Please add Description       
@Time:2022/9/6 16:10       
@Author:NianGao    
 
'''
import os
from natsort import natsorted
from tqdm import tqdm
import numpy as np
from corruption.operator.image_operator import ImageOperator
from corruption.operator.weather_operator import WeatherOperator
from utils.utils import symlink


class WeatherSimulation(object):

    @staticmethod
    def image_format(inputdir_c, calibdir, depthdir):
        dir_c = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "3rd_parts", "camera",
                             "data", "source", "kitti", "data_object", "training")
        os.makedirs(dir_c, exist_ok=True)
        symlink(inputdir_c, os.path.join(dir_c, "image_2"))
        symlink(calibdir, os.path.join(dir_c, "calib"))
        symlink(depthdir, os.path.join(dir_c, "image_2", "depth"))

    @staticmethod
    def simulation(inputdir_c, outputdir_c, inputdir_l, outputdir_l, calibdir, depthdir, dop, sev, conflict_mode=0):
        assert sev in range(1, 6)
        # op_name = dop
        WeatherSimulation.image_format(inputdir_c, calibdir, depthdir)
        op = WeatherOperator.operator_map()[dop]
        op(inputdir_c, outputdir_c, inputdir_l, outputdir_l, sev, conflict_mode=conflict_mode)

    @staticmethod
    def simulation4tracking(tracking2object_dir, outputdir_c, inputbasedir_l, outputbasedir_l, depthdir, dop, sev,
                            conflict_mode=0):
        inputdir_c = os.path.join(tracking2object_dir, "image_2")
        calibdir = os.path.join(tracking2object_dir, "calib")
        WeatherSimulation.image_format(inputdir_c, calibdir, depthdir)
        l_inputbasedir_fns = natsorted(os.listdir(inputbasedir_l))
        op = WeatherOperator.operator_map()[dop]
        for l_subinputdir in tqdm(l_inputbasedir_fns):
            l_inputdir = os.path.join(inputbasedir_l, l_subinputdir)
            l_outputdir = os.path.join(outputbasedir_l, l_subinputdir)
            op(None, None, l_inputdir, l_outputdir, sev, tp="l")  #
        # covert tracking 2 object
        op(None, outputdir_c, None, None, sev, tp="c", conflict_mode=conflict_mode)  # iamge fix format

    @staticmethod
    def simulation4depth(depth2object_dir, output_dir_c, outputdir_l, depthdir, dop, sev, conflict_mode=0):
        op = WeatherOperator.operator_map()[dop]
        inputdir_c = os.path.join(depth2object_dir, "image_2")
        inputdir_l = os.path.join(depth2object_dir, "velodyne")
        calibdir = os.path.join(depth2object_dir, "calib")
        WeatherSimulation.image_format(inputdir_c, calibdir, depthdir)
        op(inputdir_c, output_dir_c, inputdir_l, outputdir_l, sev, conflict_mode=conflict_mode)
