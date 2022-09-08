''''
@Project: fusion
@Description: Siganl Delay (Temporal misalignment)
@Time:2022/9/6 15:01       
@Author:NianGao    
'''
import os
import shutil

from natsort import natsorted
from tqdm import tqdm

from corruption.operator.delay_operator import DelayOperator
from utils.utils import symlink


class DelaySimulation(object):

    # type c or l
    # delay a source of signal (camera or Lidar)
    @staticmethod
    def simulation(inputbasedir, outputbasedir, dop, sev, copy_mode=0):  # type,
        assert sev in range(1, 6)
        print("input_dir", inputbasedir)
        print("output_dir", outputbasedir)
        print("corruption", dop)
        print("severity", sev)

        # inputbasedir = os.path.join(input_dir, suffix_d[type], "clean_{}".format(suffix_d[type]))
        # outputbasedir = os.path.join(output_dir, suffix_d[type], "{}{}_{}".format(dop, sev, suffix_d[type]))

        inputbasedir_fns = natsorted(os.listdir(inputbasedir))
        op = DelayOperator.operator_map()[dop]
        for subinputdir in tqdm(inputbasedir_fns):
            inputdir = os.path.join(inputbasedir, subinputdir)
            outputdir = os.path.join(outputbasedir, subinputdir)
            os.makedirs(outputdir, exist_ok=True)
            op(inputdir, outputdir, sev, copy_mode)
