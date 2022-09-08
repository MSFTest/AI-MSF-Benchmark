import argparse

from natsort import natsorted
from tqdm import tqdm

from python_old.atmos_models import LISA
import numpy as np
import os
import fire


def rain(input, output, intensity, conflict_mode=0):
    print(intensity)
    input_fns = os.listdir(input)
    input_fns = natsorted(input_fns)
    lisa = LISA(atm_model='rain')
    n_vec = 4
    dtype = np.float32
    for i, data_fn in tqdm(enumerate(input_fns)):
        filepath = "{}/{}".format(input, data_fn)
        output_path = "{}/{}".format(output, data_fn)
        if conflict_mode == 1 and os.path.exists(output_path) and len(
                np.fromfile(output_path, dtype=dtype)) != 0:
            continue
        else:
            data = np.fromfile(filepath, dtype=dtype)
            scan = data.reshape((-1, n_vec))
            data_aug = lisa.augment(scan, intensity)
            data_aug = data_aug[:, :n_vec]
            data_aug = data_aug.astype(dtype)
            data_aug.tofile(output_path)


def fog(input, output, level, conflict_mode=0):
    assert level in ["strong", "moderate", "low"]
    atm_model = '{}_advection_fog'.format(level)
    lisa = LISA(atm_model=atm_model)
    # lisa = LISA(atm_model='strong_advection_fog') # 0.05817631785531164 0.038836152219934264
    # lisa = LISA(atm_model='moderate_advection_fog')  # 0.03748166059893389 0.02529398153886468
    # lisa = LISA(atm_model='low_advection_fog')  # 0.021299962403234472 0.014625713419299496
    # lisa = LISA(atm_model='chu_hogg_fog')  # 0.00327451753051638 0.0022463636788279335
    n_vec = 4
    dtype = np.float32
    input_fns = os.listdir(input)
    input_fns = natsorted(input_fns)

    for data_fn in tqdm(input_fns):
        filepath = "{}/{}".format(input, data_fn)
        output_path = "{}/{}".format(output, data_fn)
        if conflict_mode == 1 and os.path.exists(output_path):
            continue
        data = np.fromfile(filepath, dtype=dtype)
        scan = data.reshape((-1, n_vec))
        data_aug = lisa.augment(scan)
        data_aug = data_aug[:, :n_vec]
        data_aug = data_aug.astype(dtype)
        data_aug.tofile(output_path)


if __name__ == '__main__':
    fire.Fire()
