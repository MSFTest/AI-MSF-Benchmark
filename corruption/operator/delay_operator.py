''''
@Project: AI-MSF-benchmark
@Description: Please add Description       
@Time:2022/9/6 15:18       
@Author:NianGao    
 
'''

import os
import shutil

from natsort import natsorted
from tqdm import tqdm

from utils.utils import symlink


class DelayOperator(object):
    @staticmethod
    def operator_map():
        operator_map = {
            "delay": DelayOperator.delay,
        }
        return operator_map

    # delay a source of signal (camera or Lidar)
    @staticmethod
    def delay(inputbasedir, outputbasedir, severity, copy_mode=0):
        t = [1, 2, 3, 4, 5][severity - 1]
        base_input_fns = natsorted(os.listdir(inputbasedir))
        output_fns = base_input_fns.copy()
        input_fns = [base_input_fns[0]] * t + base_input_fns[:-t]
        assert len(input_fns) == len(output_fns)
        for frame_i, frame_o in zip(input_fns, output_fns):
            source = os.path.join(inputbasedir, frame_i)
            target = os.path.join(outputbasedir, frame_o)
            # print(source, "->", target)
            if copy_mode == 0:
                symlink(source, target)
            else:
                shutil.copy(source, target)
