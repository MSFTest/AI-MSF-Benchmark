''''
@Project: fusion
@Description: Please add Description
@Time:2022/1/13 17:26
@Author:NianGao

'''
import os
import shutil

import cv2
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm

from corruption.operator.image_operator import ImageOperator


class ImageSimulation(object):
    """
    :Author:  NianGao
    :Create:  2022/1/13
    """

    @staticmethod
    def simulation(input_dir, output_dir, dop, sev, conflict_mode=0):
        assert sev in range(1, 6)
        print("input_dir", input_dir)
        print("output_dir", output_dir)
        print("corruption", dop)
        print("severity", sev)
        # read input
        input_fns = os.listdir(input_dir)
        input_fns = natsorted(input_fns)
        for filename in tqdm(input_fns):
            input_fn = "{}/{}".format(input_dir, filename)
            output_fn = "{}/{}".format(output_dir, filename)
            if conflict_mode == 1 and os.path.exists(output_fn):
                continue
            ImageSimulation.simulation_one(input_fn, output_fn, dop, sev)

    @staticmethod
    def simulation_one(input_fn, output_fn, dop, sev):
        op_map = ImageOperator.operator_map()
        op = op_map[dop]
        # print(filepath)
        image = cv2.imread(input_fn)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # aug one
        aug_image = op(image=image, severity=sev)
        plt.imsave(output_fn, aug_image)
