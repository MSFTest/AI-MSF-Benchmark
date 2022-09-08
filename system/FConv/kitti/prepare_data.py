''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017

Modified by Zhixin Wang
'''

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

import kitti_util as utils
from kitti_object import kitti_object
from draw_util import get_lidar_in_image_fov

from ops.pybind11.rbbox_iou import bbox_overlaps_2d


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds


def random_shift_box2d(box2d, img_height, img_width, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    assert xmin < xmax and ymin < ymax

    while True:
        cx2 = cx + w * r * (np.random.random() * 2 - 1)
        cy2 = cy + h * r * (np.random.random() * 2 - 1)
        h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
        new_box2d = np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])

        new_box2d[[0, 2]] = np.clip(new_box2d[[0, 2]], 0, img_width - 1)
        new_box2d[[1, 3]] = np.clip(new_box2d[[1, 3]], 0, img_height - 1)

        if new_box2d[0] < new_box2d[2] and new_box2d[1] < new_box2d[3]:
            return new_box2d


def extract_boxes(objects, type_whitelist):
    boxes_2d = []
    boxes_3d = []

    filter_objects = []

    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]
        if obj.type not in type_whitelist:
            continue

        boxes_2d += [obj.box2d]
        boxes_3d += [np.array([obj.t[0], obj.t[1], obj.t[2], obj.l, obj.w, obj.h, obj.ry])]
        filter_objects += [obj]

    if len(boxes_3d) != 0:
        boxes_3d = np.stack(boxes_3d, 0)
        boxes_2d = np.stack(boxes_2d, 0)

    return filter_objects, boxes_2d, boxes_3d


def extract_frustum_det_data(idx_filename, split, output_filename, det_filename,
                             perturb_box2d=False, augmentX=1, type_whitelist=['Car']):
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)

    all_boxes_2d = {}

    for i, det_idx in enumerate(det_id_list):
        if det_idx not in all_boxes_2d:
            all_boxes_2d[det_idx] = []

        all_boxes_2d[det_idx] += [
            {
                'type': det_type_list[i],
                'box2d': det_box2d_list[i],
                'prob': det_prob_list[i]
            }
        ]

    id_list = []  # int number
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    gt_box2d_list = []

    calib_list = []

    pos_cnt = 0
    all_cnt = 0
    thresh = 0.5 if 'Car' in type_whitelist else 0.25
    for data_idx in tqdm(data_idx_list):
        # print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        gt_objects = dataset.get_label_objects(data_idx)

        gt_objects, gt_boxes_2d, gt_boxes_3d = extract_boxes(gt_objects, type_whitelist)

        if len(gt_objects) == 0:
            continue

        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        det_objects = all_boxes_2d.get(data_idx)
        if det_objects is None:
            continue

        for obj_idx in range(len(det_objects)):

            cur_obj = det_objects[obj_idx]

            if cur_obj['type'] not in type_whitelist:
                continue

            overlap = bbox_overlaps_2d(cur_obj['box2d'].reshape(-1, 4), gt_boxes_2d)
            overlap = overlap[0]
            max_overlap = overlap.max(0)
            max_idx = overlap.argmax(0)

            if max_overlap < thresh:
                continue

            assign_obj = gt_objects[max_idx]

            # 2D BOX: Get pts rect backprojected
            box2d = cur_obj['box2d']
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d, img_height, img_width, 0.1)
                else:
                    xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                               (pc_image_coord[:, 0] >= xmin) & \
                               (pc_image_coord[:, 1] < ymax) & \
                               (pc_image_coord[:, 1] >= ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds, :]

                pc_box_image_coord = pc_image_coord[box_fov_inds]

                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                box2d_center_rect[0, 0])
                # 3D BOX: Get pts velo in 3d box
                obj = assign_obj
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1

                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])
                gt_box2d = obj.box2d
                # Reject too far away object or object without points
                if (gt_box2d[3] - gt_box2d[1]) < 25 or np.sum(label) == 0:
                    # print(gt_box2d[3] - gt_box2d[1], np.sum(label))
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov.astype(np.float32, copy=False))
                label_list.append(label)
                type_list.append(obj.type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)

                gt_box2d_list.append(gt_box2d)
                calib_list.append(calib.calib_dict)
                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

    print('total_objects %d' % len(id_list))
    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / len(id_list)))

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp, -1)
        pickle.dump(box2d_list, fp, -1)
        pickle.dump(box3d_list, fp, -1)
        pickle.dump(input_list, fp, -1)
        pickle.dump(label_list, fp, -1)
        pickle.dump(type_list, fp, -1)
        pickle.dump(heading_list, fp, -1)
        pickle.dump(box3d_size_list, fp, -1)
        pickle.dump(frustum_angle_list, fp, -1)
        pickle.dump(gt_box2d_list, fp, -1)
        pickle.dump(calib_list, fp, -1)

    print('save in {}'.format(output_filename))


def extract_frustum_data(idx_filename, split, output_filename,
                         perturb_box2d=False, augmentX=1, type_whitelist=['Car']):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = []  # int number
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    gt_box2d_list = []

    calib_list = []

    pos_cnt = 0
    all_cnt = 0
    for data_idx in tqdm(data_idx_list):
        # print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist:
                continue

            # 2D BOX: Get pts rect backprojected
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d, img_height, img_width, 0.1)
                else:
                    xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                               (pc_image_coord[:, 0] >= xmin) & \
                               (pc_image_coord[:, 1] < ymax) & \
                               (pc_image_coord[:, 1] >= ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds, :]

                pc_box_image_coord = pc_image_coord[box_fov_inds]

                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                box2d_center_rect[0, 0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1

                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if (box2d[3] - box2d[1]) < 25 or np.sum(label) == 0:
                    # print(box2d[3] - box2d[1], np.sum(label))
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov.astype(np.float32, copy=False))
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)

                gt_box2d_list.append(box2d)
                calib_list.append(calib.calib_dict)
                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

    print('total_objects %d' % len(id_list))
    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / len(id_list)))

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp, -1)
        pickle.dump(box2d_list, fp, -1)
        pickle.dump(box3d_list, fp, -1)
        pickle.dump(input_list, fp, -1)
        pickle.dump(label_list, fp, -1)
        pickle.dump(type_list, fp, -1)
        pickle.dump(heading_list, fp, -1)
        pickle.dump(box3d_size_list, fp, -1)
        pickle.dump(frustum_angle_list, fp, -1)
        pickle.dump(gt_box2d_list, fp, -1)
        pickle.dump(calib_list, fp, -1)

    print('save in {}'.format(output_filename))


def get_box3d_dim_statistics(idx_filename):
    ''' Collect 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'))
    dimension_list = []
    type_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in tqdm(data_idx_list):
        # print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type == 'DontCare':
                continue
            dimension_list.append(np.array([obj.l, obj.w, obj.h]))
            type_list.append(obj.type)

    print("number of objects: {} ".format(len(type_list)))
    print("categories:", set(type_list))

    # Get average box size for different categories
    for class_type in sorted(set(type_list)):
        box3d_list = []
        for i in range(len(dimension_list)):
            if type_list[i] == class_type:
                box3d_list.append(dimension_list[i])

        # m_box3d = np.median(box3d_list, 0)
        m_box3d = np.mean(box3d_list, 0)
        print("\'%s\': np.array([%f,%f,%f])," %
              (class_type, m_box3d[0], m_box3d[1], m_box3d[2]))


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}  # default definition in rgb_detection_train/val.txt by rqi
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        try:
            cls_type = det_id2str[int(t[1])]
        except ValueError:
            assert t[1] in det_id2str.values()
            cls_type = t[1]
        type_list.append(cls_type)
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list


def read_det_pkl_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    with open(det_filename, 'r') as fn:
        results = pickle.load(fn)

    id_list = results['id_list']
    type_list = results['type_list']
    box2d_list = results['box2d_list']
    prob_list = results['prob_list']

    return id_list, type_list, box2d_list, prob_list


def extract_frustum_data_rgb_detection(det_filename, split, output_filename,
                                       type_whitelist=['Car'],
                                       img_height_threshold=5,
                                       lidar_point_threshold=1):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split=split)
    if det_filename.split('.')[-1] == 'pkl':
        det_id_list, det_type_list, det_box2d_list, det_prob_list = \
            read_det_pkl_file(det_filename)
    else:
        det_id_list, det_type_list, det_box2d_list, det_prob_list = \
            read_det_file(det_filename)
    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis
    calib_list = []

    for det_idx in tqdm(range(len(det_id_list))):
        data_idx = det_id_list[det_idx]
        # print('det idx: %d/%d, data idx: %d' %
        #       (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
            pc_velo = dataset.get_lidar(data_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
            cache = [calib, pc_rect, pc_image_coord, img_fov_inds]
            cache_id = data_idx
        else:
            calib, pc_rect, pc_image_coord, img_fov_inds = cache

        if det_type_list[det_idx] not in type_whitelist:
            continue

        # 2D BOX: Get pts rect backprojected
        det_box2d = det_box2d_list[det_idx].copy()
        det_box2d[[0, 2]] = np.clip(det_box2d[[0, 2]], 0, img_width - 1)
        det_box2d[[1, 3]] = np.clip(det_box2d[[1, 3]], 0, img_height - 1)

        xmin, ymin, xmax, ymax = det_box2d
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                       (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & \
                       (pc_image_coord[:, 1] >= ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds, :]

        pc_box_image_coord = pc_image_coord[box_fov_inds, :]

        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                        box2d_center_rect[0, 0])

        # Pass objects that are too small
        if ymax - ymin < img_height_threshold or xmax - xmin < 1 or \
                len(pc_in_box_fov) < lidar_point_threshold:
            continue

        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov.astype(np.float32, copy=False))
        frustum_angle_list.append(frustum_angle)
        calib_list.append(calib.calib_dict)

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp, -1)
        pickle.dump(box2d_list, fp, -1)
        pickle.dump(input_list, fp, -1)
        pickle.dump(type_list, fp, -1)
        pickle.dump(frustum_angle_list, fp, -1)
        pickle.dump(prob_list, fp, -1)
        pickle.dump(calib_list, fp, -1)

    print('total_objects %d' % len(id_list))
    print('save in {}'.format(output_filename))


def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format.

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)

    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'data/kitti'), split)

    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {}
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results:
            results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_train', action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')

    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')

    parser.add_argument('--gen_val_rgb_detection', action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')

    parser.add_argument('--gen_trainval', action='store_true',
                        help='Generate trainval split frustum data with perturbed GT 2D boxes')

    parser.add_argument('--gen_test_rgb_detection', action='store_true',
                        help='Generate test split frustum data with RGB detection 2D boxes')

    parser.add_argument('--car_only', action='store_true', help='Only generate cars')
    parser.add_argument('--people_only', action='store_true', help='Only generate peds and cycs')
    parser.add_argument('--save_dir', default=None, type=str, help='data directory to save data')

    parser.add_argument('--gen_avg_dim', action='store_true', help='get average dimension of each class')

    args = parser.parse_args()

    np.random.seed(3)

    if args.gen_avg_dim:
        get_box3d_dim_statistics(os.path.join(BASE_DIR, 'image_sets/train.txt'), )

    if args.save_dir is None:
        save_dir = 'kitti/data/pickle_data'
    else:
        save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'

    elif args.people_only:
        type_whitelist = ['Pedestrian', 'Cyclist']
        output_prefix = 'frustum_pedcyc_'

    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.gen_train:
        extract_frustum_data(
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            os.path.join(save_dir, output_prefix + 'train.pickle'),
            perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist)

    if args.gen_val:
        extract_frustum_data(
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(save_dir, output_prefix + 'val.pickle'),
            perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist)

    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection(
            os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
            'training',
            os.path.join(save_dir, output_prefix + 'val_rgb_detection.pickle'),
            type_whitelist=type_whitelist)
