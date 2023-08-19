import os

import fire
import numpy as np
from tqdm import tqdm


def read_id_list(p1):
    with open(p1, "r") as f:
        data = f.read().splitlines()
    return data


# Image ID, Category ID, Score, Box (x1, y1, x2, y2)
# det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
str2det_id = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3}


def conver_2D_result(from_dir, to_dir, id_list, id_type):
    lines_format = "{} {} {} {} {} {} {}\n"
    output_path = "{}/rgb_detection_{}.txt".format(to_dir, id_type)
    with open(output_path, "w") as fo:
        lines_arr = []
        # for id in tqdm(id_list):
        for id in id_list:
            input_path = "{}/{}.txt".format(from_dir, id)
            with open(input_path, "r") as f:
                lines_input = f.read().splitlines()
                for line_input in lines_input:
                    # data =  # drop the \n
                    arr = line_input.split(" ")
                    c_id = str2det_id[arr[0]]
                    score = arr[-1]
                    x1, y1, x2, y2 = int(float(arr[4])), int(float(arr[5])), int(float(arr[6])), int(float(arr[7]))
                    line = lines_format.format(id, c_id, score, x1, y1, x2, y2)
                    lines_arr.append(line)
        fo.writelines(lines_arr)
        # print(output_path)


def convert(from_dir, multi_insert=False):
    if multi_insert:  # mark adaptive msf insert project
        convert_msf_insert(from_dir)
    else:
        convert_msf_benchmark(from_dir)


def convert_msf_benchmark(from_dir):
    # train_id_path = "../frustum-convnet/kitti/image_sets/train.txt"
    train_id_path = "./kitti/image_sets/train.txt"
    train_id_list = read_id_list(train_id_path)
    test_id_path = "./kitti/image_sets/val.txt"
    # test_id_path = "../frustum-convnet/kitti/image_sets/val.txt"
    test_id_list = read_id_list(test_id_path)

    # to_dir = "../frustum-convnet/kitti/rgb_detections"
    to_dir = "./kitti/rgb_detections"

    id_type_arr = ["train", "val"]
    id_list_arr = [train_id_list, test_id_list]
    for id_type, id_list in zip(id_type_arr, id_list_arr):
        conver_2D_result(from_dir, to_dir, id_list, id_type)


def convert_msf_insert(from_dir):
    train_id_list = []
    test_id_path = "./kitti/image_sets/val.txt"
    test_id_list = read_id_list(test_id_path)
    to_dir = "./kitti/rgb_detections"
    id_type_arr = ["train", "val"]
    id_list_arr = [train_id_list, test_id_list]
    for id_type, id_list in zip(id_type_arr, id_list_arr):
        conver_2D_result(from_dir, to_dir, id_list, id_type)


if __name__ == '__main__':
    fire.Fire(convert)
