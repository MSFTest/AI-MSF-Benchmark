import os

import cv2
from natsort import natsorted
from tqdm import tqdm
from skimage import io


def assert_tracking(sub_dir_path):
    sub_sub_dirs = natsorted(os.listdir(sub_dir_path))
    num = 0
    for sub_sub_dir in tqdm(sub_sub_dirs):
        d = os.path.join(sub_dir_path, sub_sub_dir)
        fns = natsorted(os.listdir(d))
        num += len(fns)
        for fn in fns:
            f = os.path.join(d, fn)
            try:
                img = cv2.imread(f)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                os.remove(f)
                print(f)
    print(sub_dir_path.split("/")[-1], "len", num)


def assert_tacking_all(dir_tracking):
    skip_arr = []
    for dir in [dir_tracking]:
        sub_dirs = natsorted(os.listdir(dir))
        print(len(sub_dirs))
        for sub_dir in sub_dirs:
            print(sub_dir)
            if sub_dir in skip_arr:
                continue
            if "_object" in sub_dir:
                continue
            sub_dir_path = os.path.join(dir, sub_dir)
            assert_tracking(sub_dir_path)


def assert_object_all(dir_object):
    skip_arr = []
    for dir in tqdm([dir_object]):
        sub_dirs = natsorted(os.listdir(dir))
        # length_map = {}
        for sub_dir in tqdm(sub_dirs):
            if sub_dir in skip_arr:
                continue
            assert_object(dir, sub_dir)


def assert_object(dir, sub_dir):
    d = os.path.join(dir, sub_dir)
    fns = natsorted(os.listdir(d))
    print(sub_dir, "len", len(fns))
    # length_map[d] = fns
    for fn in tqdm(fns):
        f = os.path.join(d, fn)
        try:
            img = cv2.imread(f)
            img3 = io.imread(f).shape[:2]
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            os.remove(f)
            print(f)


def assert_depth_all(dir_depth):
    skip_arr = []
    for dir in tqdm([dir_depth]):
        print(dir)
        sub_dirs = natsorted(os.listdir(dir))
        # length_map = {}
        for sub_dir in tqdm(sub_dirs):
            print(sub_dir)
            if sub_dir in skip_arr:
                continue
            if "_object" in sub_dir:
                continue
            if ".zip" in sub_dir:
                continue
            d = os.path.join(dir, sub_dir)
            fns = natsorted(os.listdir(d))
            print(sub_dir, "len", len(fns))
            # length_map[d] = fns
            for fn in fns:
                f = os.path.join(d, fn)
                try:
                    img = cv2.imread(f)
                    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    print(f)


import numpy as np


def assert_tracking_lidar_all(dir):
    skip_arr = []
    sub_dirs = natsorted(os.listdir(dir))
    print(len(sub_dirs))
    for sub_dir in sub_dirs:
        print(sub_dir)
        if sub_dir in skip_arr:
            continue
        if "_object" in sub_dir:
            continue
        sub_dir_path = os.path.join(dir, sub_dir)
        sub_sub_dirs = natsorted(os.listdir(sub_dir_path))
        num = 0
        for sub_sub_dir in tqdm(sub_sub_dirs):
            d = os.path.join(sub_dir_path, sub_sub_dir)
            fns = natsorted(os.listdir(d))
            num += len(fns)
            for fn in fns:
                f = os.path.join(d, fn)
                try:
                    points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
                    assert len(points) > 0
                except:
                    # os.remove(f)
                    print(f)
        print(sub_dir_path.split("/")[-1], "len", num)


# Check if the generated data can be read properly
if __name__ == '__main__':
    dir_fconv = "kitti_corruption/training/noise/lidar_fov/"
    assert_object(dir_fconv, "loss75")
