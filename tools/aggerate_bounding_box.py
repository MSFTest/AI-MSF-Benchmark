import glob
import os

from natsort import natsorted
import numpy as np
import numba
from ops.second.my_eval import get_label_anno
from ops.second.utils.eval import calculate_iou_partly

def merge_dp(dir1, dir2, output_dir, iou_th=None, score=0.0, metric=None):
    # 0 box  1 bev  2 3d
    if metric is None:
        raise ValueError()
    dts1 = glob.glob(dir1 + "/*")
    dts2 = glob.glob(dir2 + "/*")
    # print(dts2)
    # print(len(dts1), len(dts2))
    if len(dts1) != len(dts2):
        raise ValueError(dir1, len(dts1), dir2, len(dts2))
    assert len(dts1) == len(dts2)
    assert len(dts1) != 0
    dts1 = natsorted(dts1)
    dts2 = natsorted(dts2)
    if iou_th is None:
        for p1, p2 in zip(dts1, dts2):
            with open(p1, "r") as f:
                p1_txt = f.readlines()
            with open(p2, "r") as f:
                p2_txt = f.readlines()
            if len(p1_txt) > 0:
                p1_txt[-1] = p1_txt[-1].strip() + "\n"
            if len(p2_txt) > 0:
                p2_txt[-1] = p2_txt[-1].strip() + "\n"
            p_txt = p1_txt + p2_txt
            p_txt.sort()
            p_txt[-1] = p_txt[-1].strip()
            # print(len(p1_txt), len(p2_txt))
            fn = os.path.basename(p1)
            po = os.path.join(output_dir, fn)
            with open(po, "w") as f:
                f.writelines(p_txt)
    else:
        add_count = 0
        # score = 0.8
        obj_interesting = "Car"
        for p1, p2 in zip(dts1, dts2):
            with open(p1, "r") as f:
                p1_txt = f.readlines()  # fusion
            with open(p2, "r") as f:
                p2_txt = f.readlines()  # others
            if len(p1_txt) > 0:
                p1_txt[-1] = p1_txt[-1].strip() + "\n"
            if len(p2_txt) > 0:
                p2_txt[-1] = p2_txt[-1].strip() + "\n"
            p1_txt_car = list(filter(lambda x: x[:3] == "Car", p1_txt))
            p2_txt_car = list(filter(lambda x: x[:3] == "Car", p2_txt))
            if len(p1_txt_car) == 0:
                p2_txt_car = list(filter(lambda obj: float(obj.strip().split(" ")[-1]) > score, p2_txt_car))
                p_txt = p2_txt_car
            elif len(p2_txt_car) == 0:
                p_txt = p1_txt_car
            else:
                fusion_results = get_label_anno(p1, obj_interesting)
                signal_results = get_label_anno(p2, obj_interesting)
                rets = calculate_iou_partly(signal_results, fusion_results, metric=metric, num_parts=1)
                overlaps, parted_overlaps, total_gt_num, total_dt_num = rets
                overlaps_array = overlaps[0]
                try:
                    overlaps_array_max = np.max(overlaps_array, axis=1)
                except:
                    print(overlaps_array)
                    print(p1_txt_car, p2_txt_car)
                    print(dir1)
                    print(len(dts1))
                    raise ValueError()
                    # overlaps_array_max = np.max(overlaps_array, axis=1)
                select_idx = np.where(overlaps_array_max < iou_th)[0]
                # p2_txt_selected_car = p2_txt_car[select_idx]
                # p2_txt_selected_car = operator.itemgetter(select_idx)(p2_txt_car)
                p2_txt_selected_car = []
                for idx in select_idx:
                    obj = p2_txt_car[idx]
                    if float(obj.strip().split(" ")[-1]) < score:  # confidence socre
                        continue
                    p2_txt_selected_car.append(obj)
                p_txt = p1_txt_car + p2_txt_selected_car
            if len(p_txt) > 0:
                p_txt[-1] = p_txt[-1].strip()
            fn = os.path.basename(p1)
            po = os.path.join(output_dir, fn)
            if len(p_txt) != len(p1_txt_car):
                add_count += len(p_txt) - len(p1_txt_car)
                # print("=============", p1, "=============")
                # print(p1_txt_car)
                # print(p2_txt_car)
                # print(p_txt)
                # print("=============")
            with open(po, "w") as f:
                f.writelines(p_txt)
        print("add", add_count, "objects")

