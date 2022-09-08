''''
@Project: fusion
@Description: Calib error (Spatial misalignment)
@Time:2022/1/13 17:26
@Author:NianGao

'''
from corruption.operator.calib_operator import CalibOperator
import os
import shutil
import cv2
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
import numpy as np


class CalibSimulation(object):
    """
    :Author:  NianGao
    :Create:  2022/1/13
    """

    # calib format (object & tracking)
    # |-000001.txt
    # |-000002.txt
    @staticmethod
    def simulation(input_dir, output_dir, dop, sev, fm="%06d.txt"):
        assert sev in range(1, 4)
        print("input_dir", input_dir)
        print("output_dir", output_dir)
        print("corruption", dop)
        print("severity", sev)
        fn_arr = os.listdir(input_dir)
        index_arr = [int(fn.split(".")[0]) for fn in fn_arr]
        index_arr.sort()
        for data_idx in tqdm(index_arr):
            input_fn = os.path.join(input_dir, fm % (data_idx))
            output_fn = os.path.join(output_dir, fm % (data_idx))
            CalibSimulation.simulation_one(input_fn, output_fn, dop, sev)

    @staticmethod
    def simulation_one(input_fn, output_fn, dop, sev):
        # if isinstance(dop, str):
        op = CalibOperator.operator_map()[dop]
        # else:
        #     op = dop
        line_arr = []
        with open(input_fn, "r") as f:
            flag = False
            for _line in f.readlines():
                if len(_line.rstrip()) == 0:
                    continue
                if "Tr_velo_" in _line:
                    if "Tr_velo_to_cam: " in _line:
                        key = "Tr_velo_to_cam: "
                        flag = True
                    elif "Tr_velo_cam " in _line:
                        key = "Tr_velo_cam "
                        flag = True
                    value = _line.replace(key, "")
                    V2C = np.array([float(x) for x in value.split()])
                    V2C = np.reshape(V2C, [3, 4])
                    V2C_noise = op(V2C, sev)
                    V2C_noise = numpy2str(V2C_noise)
                    _line = key + V2C_noise + "\n"
                line_arr.append(_line)
            assert flag == True
        with open(output_fn, "w") as f:
            for line in line_arr:
                f.write(line)

    # calib format (Depth complete)
    # 2011_0928
    # |-format calib_cam_to_cam.txt
    # |-calib_velo_to_cam.txt
    @staticmethod
    def simulation4depth(input_dir, output_dir, dop, sev):
        op = CalibOperator.operator_map()[dop]
        calib_dirs = natsorted(os.listdir(input_dir))
        for sub_dir in calib_dirs:
            calib_input_dir = os.path.join(input_dir, sub_dir)
            calib_output_dir = os.path.join(output_dir, sub_dir)
            # print(calib_output_dir)
            os.makedirs(calib_output_dir, exist_ok=True)
            for fn in natsorted(os.listdir(calib_input_dir)):
                calib_input_txt = os.path.join(calib_input_dir, fn)
                calib_output_txt = os.path.join(calib_output_dir, fn)
                # print(calib_output_txt)
                if fn == "calib_velo_to_cam.txt":
                    with open(calib_input_txt, "r") as f:
                        lines = f.readlines()
                    R_key, R_value = lines[1].strip().split(":")
                    T_key, T_value = lines[2].strip().split(":")
                    R = np.array(list(map(float, R_value.strip().split(' '))))
                    T = np.array(list(map(float, T_value.strip().split(' '))))
                    velo2cam = np.hstack((R.reshape(3, 3), T[..., np.newaxis]))
                    assert velo2cam.shape == (3, 4)
                    V2C_noise = op(velo2cam, sev)
                    assert len(V2C_noise) == 12
                    R_noise = np.concatenate([V2C_noise[:3], V2C_noise[4:7], V2C_noise[8:11]])
                    T_noise = np.array([V2C_noise[3], V2C_noise[7], V2C_noise[11]])
                    R_noise_txt = R_key + ": " + numpy2str(R_noise) + "\n"
                    T_noise_txt = T_key + ": " + numpy2str(T_noise) + "\n"
                    lines[1] = R_noise_txt
                    lines[2] = T_noise_txt
                    with open(calib_output_txt, "w") as f:
                        f.writelines(lines)
                else:
                    shutil.copyfile(calib_input_txt, calib_output_txt)


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def numpy2str(arr):
    arr = np.array2string(arr, separator=" ")[1:-1].strip()
    arr = arr.replace(",", "")
    arr = arr.replace("\n", "")
    while True:
        arr = arr.replace("  ", " ")
        if "  " not in arr:
            break
    return arr
