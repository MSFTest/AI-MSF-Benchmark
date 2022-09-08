import glob
import operator
import os
import pathlib
import pickle
import shutil

from google.protobuf import text_format
from natsort import natsorted
from tqdm import tqdm

from second.builder import anchor_generator_builder
from second.protos import pipeline_pb2
from second.utils.eval import get_official_eval_result, calculate_iou_partly
import second.data.kitti_common as kitti
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning, \
    NumbaWarning
import warnings
import numpy as np

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


def get_label_anno(label_path, obj_interesting):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    # print(content)
    if obj_interesting is not None:
        content = [x for x in content if x[0] == obj_interesting]
    # print(content)
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
        -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0],))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return [annotations]


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
            if len(p1_txt) == 0:
                p2_txt_car = list(filter(lambda obj: float(obj.strip().split(" ")[-1]) > score, p2_txt))
                p_txt = p2_txt_car
            elif len(p2_txt) == 0:
                p_txt = p1_txt
            else:
                fusion_results = get_label_anno(p1, obj_interesting)
                signal_results = get_label_anno(p2, obj_interesting)
                rets = calculate_iou_partly(signal_results, fusion_results, metric=metric, num_parts=1)
                overlaps, parted_overlaps, total_gt_num, total_dt_num = rets
                overlaps_array = overlaps[0]
                overlaps_array_max = np.max(overlaps_array, axis=1)
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
            with open(po, "w") as f:
                f.writelines(p_txt)


def evaluate(dt_path):
    data_path = "/home/niangao/disk1/kitti4cloc"
    root_path = pathlib.Path(data_path)
    info_path = root_path / 'kitti_infos_val.pkl'
    dt_annos = kitti.get_label_annos(dt_path)

    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    gt_annos = [info["annos"] for info in kitti_infos]
    # config_path = './configs/car.fhd.config'
    config_path = '/home/niangao/PycharmProjects/fusion/CLOCs/second/configs/car.fhd.config'
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    model_cfg = config.model.second
    target_assigner_cfg = model_cfg.target_assigner
    anchor_cfg = target_assigner_cfg.anchor_generators
    anchor_generators = []
    for a_cfg in anchor_cfg:
        anchor_generator = anchor_generator_builder.build(a_cfg)
        anchor_generators.append(anchor_generator)

    class_names = [a.class_name for a in anchor_generators]

    result = get_official_eval_result(gt_annos, dt_annos, class_names)
    result_path = dt_path + ".txt"
    with open(result_path, "w") as f:
        f.write(result)


# def merge_image():
#     # img_op_arr = ["bright1", "bright3", "bright5",
#     #               "dark1", "dark3", "dark5",
#     #               "motionblur1", "motionblur3", "motionblur5",
#     #               "defocusblur1", "defocusblur3", "defocusblur5",
#     #               "distortion1", "distortion", "distortion5",
#     #               "loss10", "loss25", "loss50", "loss75", "black",
#     #               "noise_gauss1", "noise_gauss3", "noise_gauss5",
#     #               "noise_impulse1", "noise_impulse3", "noise_impulse5",
#     #               ]
#     # img_op_arr = ["bright3", "dark3", "loss50"]
#     img_op_arr = ["distortion1"]
#     # for img_op in tqdm(img_op_arr):
#     for img_op in tqdm(img_op_arr):
#         lidar_op = "clean"
#         dt_dir_clocs = "/home/niangao/PycharmProjects/fusion/result/CLOCs/img_{}_lidar_{}_calib_clean" \
#             .format(img_op, lidar_op)
#         dt_dir_second = "/home/niangao/PycharmProjects/fusion/result/second/img_clean_lidar_{}_calib_clean/data" \
#             .format(lidar_op)
#         output_dir = "/home/niangao/PycharmProjects/fusion/CLOCs/second/merge_dt/img_{}_lidar_{}_calib_clean" \
#             .format(img_op, lidar_op)
#         # if os.path.exists(output_dir):
#         #     shutil.rmtree(output_dir)
#         os.makedirs(output_dir, exist_ok=True)
#
#         merge_dp(dt_dir_clocs, dt_dir_second, output_dir)  # , iou_th=0.8
#         evaluate(output_dir)


def my_merge_and_eval(noise):
    img_op = noise["img"]
    lidar_op = noise["lidar"]
    if img_op is None:
        img_op = "clean"
    if lidar_op is None:
        lidar_op = "clean"
    dt_dir_clocs = "/home/niangao/PycharmProjects/fusion/result/CLOCs/img_{}_lidar_{}_calib_clean" \
        .format(img_op, lidar_op)
    dt_dir_second = "/home/niangao/PycharmProjects/fusion/result/second/img_{}_lidar_{}_calib_clean/data" \
        .format(img_op, lidar_op)
    output_dir = "/home/niangao/PycharmProjects/fusion/CLOCs/second/merge_dt/img_{}_lidar_{}_calib_clean" \
        .format(img_op, lidar_op)
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    merge_dp(dt_dir_clocs, dt_dir_second, output_dir, iou_th=0.7, score=0.8, metric=2)  #
    evaluate(output_dir)


# CLOCS-Rb
if __name__ == '__main__':
    noise_arr = [
        # # light
        # {"img": "bright1", "lidar": None, "calib": None},
        # {"img": "bright3", "lidar": None, "calib": None},
        # {"img": "bright5", "lidar": None, "calib": None},
        # {"img": "dark1", "lidar": None, "calib": None},
        # {"img": "dark3", "lidar": None, "calib": None},
        # {"img": "dark5", "lidar": None, "calib": None},
        # {"img": "motionblur1", "lidar": None, "calib": None},
        # {"img": "motionblur3", "lidar": None, "calib": None},
        # {"img": "motionblur5", "lidar": None, "calib": None},
        # {"img": "defocusblur1", "lidar": None, "calib": None},
        # {"img": "defocusblur3", "lidar": None, "calib": None},
        # {"img": "defocusblur5", "lidar": None, "calib": None},
        # {"img": "distortion1", "lidar": None, "calib": None},
        # {"img": "distortion", "lidar": None, "calib": None},
        # {"img": "distortion5", "lidar": None, "calib": None},
        # {"img": "noise_gauss1", "lidar": None, "calib": None},
        # {"img": "noise_gauss3", "lidar": None, "calib": None},
        # {"img": "noise_gauss5", "lidar": None, "calib": None},
        # {"img": "noise_impulse1", "lidar": None, "calib": None},
        # {"img": "noise_impulse3", "lidar": None, "calib": None},
        # {"img": "noise_impulse5", "lidar": None, "calib": None},
        # # # image loss
        # {"img": "loss10", "lidar": None, "calib": None},
        # {"img": "loss25", "lidar": None, "calib": None},
        # {"img": "loss50", "lidar": None, "calib": None},
        # {"img": "loss75", "lidar": None, "calib": None},
        # {"img": "black", "lidar": None, "calib": None},
        # # # lidar loss
        # {"img": None, "lidar": "loss10", "calib": None},
        # {"img": None, "lidar": "loss25", "calib": None},
        # {"img": None, "lidar": "loss50", "calib": None},
        # {"img": None, "lidar": "loss75", "calib": None},
        # {"img": None, "lidar": "black", "calib": None},
        # # lidar noise
        # {"img": None, "lidar": "noise_gauss1", "calib": None},
        # {"img": None, "lidar": "noise_gauss3", "calib": None},
        # {"img": None, "lidar": "noise_gauss5", "calib": None},
        # {"img": None, "lidar": "noise_impulse1", "calib": None},
        # {"img": None, "lidar": "noise_impulse3", "calib": None},
        # {"img": None, "lidar": "noise_impulse5", "calib": None},
        # weather
        {"img": "fog_low", "lidar": "fog_low", "calib": None},
        {"img": "fog_moderate", "lidar": "fog_moderate", "calib": None},
        {"img": "fog_strong", "lidar": "fog_strong", "calib": None},
        {"img": "rain10", "lidar": "rain10", "calib": None, },
        {"img": "rain25", "lidar": "rain25", "calib": None, },
        {"img": "rain50", "lidar": "rain50", "calib": None, },
    ]
    # 11 *3 + 5*2 = 43
    for noise in tqdm(noise_arr):
        my_merge_and_eval(noise)
