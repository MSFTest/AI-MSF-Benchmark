import argparse
from datetime import datetime

import tqdm
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from lib.net.pointnet2_msg import Pointnet2MSG

# Pointnet2MSG

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import os
import torch
import re
import numpy as np
import logging
from lib.net.point_rcnn import PointRCNN
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from torch.utils.data import DataLoader
import tools.train_utils.train_utils as train_utils

# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py
from lib.utils import kitti_utils
from lib.utils.bbox_transform import decode_bbox_target
from lib.utils.iou3d import iou3d_utils

np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--cfg_file', type=str, default='cfgs/LI_Fusion_with_attention_use_ce_loss.yaml',
                    help='specify the config for evaluation')

parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
parser.add_argument('--test', action='store_true', default=False, help='evaluate without ground truth')
parser.add_argument("--ckpt", type=str, default=None, help="specify a checkpoint to be evaluated")
parser.add_argument("--rpn_ckpt", type=str, default=None,
                    help="specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt", type=str, default=None,
                    help="specify the checkpoint of rcnn if trained separated")

parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--extra_tag", type=str, default='default', help="extra tag for multiple evaluation")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument("--ckpt_dir", type=str, default=None,
                    help="specify a ckpt directory to be evaluated if needed")

parser.add_argument('--save_result', action='store_true', default=False, help='save evaluation results to files')
parser.add_argument('--save_rpn_feature', action='store_true', default=False,
                    help='save features for separately rcnn training and evaluation')

parser.add_argument('--random_select', action='store_true', default=True,
                    help='sample to the same number of points')
parser.add_argument('--start_epoch', default=0, type=int, help='ignore the checkpoint smaller than this epoch')
parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
parser.add_argument("--rcnn_eval_roi_dir", type=str, default=None,
                    help='specify the saved rois for rcnn evaluation when using rcnn_offline mode')
parser.add_argument("--rcnn_eval_feature_dir", type=str, default=None,
                    help='specify the saved features for rcnn evaluation when using rcnn_offline mode')
parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                    help='set extra config keys if needed')

parser.add_argument('--model_type', type=str, default='base', help='model type')

args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def create_dataloader(logger):
    mode = 'TEST' if args.test else 'EVAL'
    # DATA_PATH = os.path.join('../../', 'data')
    DATA_PATH = os.path.join('/home/niangao/PycharmProjects/fusion/EPNet/data')

    # create dataloader
    test_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TEST.SPLIT, mode=mode,
                                random_select=args.random_select,
                                rcnn_eval_roi_dir=args.rcnn_eval_roi_dir,
                                rcnn_eval_feature_dir=args.rcnn_eval_feature_dir,
                                classes=cfg.CLASSES,
                                logger=logger)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers, collate_fn=test_set.collate_batch)

    return test_loader


def load_ckpt_based_on_args(model, logger):
    if args.ckpt is not None:
        train_utils.load_checkpoint(model, filename=args.ckpt, logger=logger)

    total_keys = model.state_dict().keys().__len__()
    if cfg.RPN.ENABLED and args.rpn_ckpt is not None:
        load_part_ckpt(model, filename=args.rpn_ckpt, logger=logger, total_keys=total_keys)

    if cfg.RCNN.ENABLED and args.rcnn_ckpt is not None:
        load_part_ckpt(model, filename=args.rcnn_ckpt, logger=logger, total_keys=total_keys)


def load_part_ckpt(model, filename, logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


# def vis_2D(img, object_preds, gts_objects, save_path=None):
#     if object_preds is None:
#         d2_dts_objects = None
#     else:
#         conf = 1
#         d2_dts_objects = [MyObject2d(object_pred, "Car", conf) for object_pred in object_preds]
#     show_image_with_dt_dt_boxes(img, d2_dts_objects, gts_objects, hull=None, save_path=save_path)


def attack(model, dataloader, e=0.2):
    th = int(e * 255)
    assert cfg.RPN.ENABLED and cfg.RCNN.ENABLED
    # ret_dict = eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger)
    model.eval()
    # model.train()
    for data in dataloader:
        box_target = torch.from_numpy(data['gt_boxes3d'][0]).cuda(non_blocking=True).float()
        if len(box_target) == 0:  # TODO:
            return
        input_data, batch_size = get_input_data(data)
        input_data['gt_boxes3d'] = torch.from_numpy(data['gt_boxes3d']).cuda(non_blocking=True).float()
        assert batch_size == 1
        img_tensor = input_data["img"]
        img = img_tensor.detach().cpu().numpy()
        ori_img = img
        img_min_th = ori_img.astype("int32") - th
        img_min_th = np.clip(img_min_th, 0, 255).astype("uint8")

        img_max_th = ori_img.astype("int32") + th
        img_max_th = np.clip(img_max_th, 0, 255).astype("uint8")

        iteration = 50
        is_vis = False
        base_save_path = None

        for i in range(iteration):
            if base_save_path is not None:
                save_path = base_save_path.format(i)
            else:
                save_path = None
            result = attack_detail(input_data, model, box_target, batch_size, is_vis=is_vis)
            # if result is None:
            #     vis_2D(img, None, box_target, save_path=save_path)
            #     break
            adv_img, iou_max_num, object_preds = result

            l1_change_iter = np.sum(abs(adv_img - img))
            l1_change_all = np.sum(abs(adv_img - ori_img))

            # print("l1_change_iter", l1_change_iter, "l1_change_all", l1_change_all, "iou_max_num", iou_max_num)
            # clip
            adv_img = np.where(adv_img > img_min_th, adv_img, img_min_th)
            adv_img = np.where(adv_img < img_max_th, adv_img, img_max_th)
            img = adv_img

        break
        # vis_2D(img, object_preds, gts_objects, save_path=save_path)

        # if img_save_dir is not None:
        #     adv_path = img_save_dir + "/{}.png".format(get_file_name(data_idx))
        #     plt.imsave(adv_path, img)

    # return ret_dict


def attack_detail(input_data, model, box_target, batch_size, is_vis=False):
    np.random.seed(666)
    img_tensor = input_data["img"]
    pts_tensor = input_data['pts_input']
    pts_origin_xy_tensor = input_data['pts_origin_xy']


    img_tensor.requires_grad = True
    pts_tensor.requires_grad = True
    pts_origin_xy_tensor.requires_grad = True
    ret_dict = model.forward_with_grad(input_data)
    roi_boxes3d = ret_dict['rois']
    rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
    raw_scores = rcnn_cls  # (B, M, 1)
    norm_scores = torch.sigmoid(raw_scores)
    pred_boxes3d = get_pred_box_tensor(ret_dict, batch_size, roi_boxes3d)
    box_pred = select_box(norm_scores, pred_boxes3d, raw_scores)
    # from torchviz import make_dot
    # dot = make_dot(img_tensor)
    # # dot.r
    # # display(dot)
    # dot.render(filename="graph",view=False)
    # print("===========")
    # print( scores_selected.size())  # [x,7] [x,1]
    # print(box_pred.size(), box_target.size())
    ious = iou3d_utils.boxes_iou3d_gpu(box_pred, box_target)  # pred * gt
    # print(ious)
    iou_max_num, iou_max_index = torch.max(ious, dim=1)
    iou_max_num = iou_max_num.detach().cpu().numpy()
    box_target_overlap = box_target[iou_max_index]
    loss_func = My_loss3D()  # loss
    # cal loss based on iou
    loss = None
    for i in range(len(iou_max_index)):
        temp_loss = loss_func(box_pred[i], box_target_overlap[i])
        if loss is None:
            loss = temp_loss
        else:
            loss += temp_loss
    loss.requires_grad_(True)
    # 将所有现有的渐变归零
    model.zero_grad()
    # 计算后向传递模型的梯度
    loss.backward()

    img_tensor_detach = img_tensor.detach()
    img_tensor_grad = img_tensor.grad.data.detach()  # grad
    img_tensor_detach = img_tensor_detach[0]
    img_tensor_grad = img_tensor_grad[0]
    # perb_tensor = fgsm_attack(img_tensor_detach, 0.1, img_tensor_grad)  # attack
    perb_tensor = my_attack(img_tensor_detach, 0.01, img_tensor_grad)  # attack
    # perb_tensor = fgsm_attack(img_tensor_detach, 0.3, img_tensor_grad)  # attack
    # perturbed_data = img_tensor_detach + perb
    # adv_img = convert_tensor2_img(img_tensor_detach, unorm)
    # perb_numpy = convert_tensor2_img(perb_tensor * 255, unorm=None)
    perb_numpy = perb_tensor.detach().cpu().numpy()
    perb_numpy = np.transpose(perb_numpy, (1, 2, 0))[:, :, ::-1] * 255
    # print(np.max(perb_numpy), np.min(perb_numpy))
    img = img_tensor.detach().cpu().numpy()
    adv_img = img.astype("int32") + perb_numpy.astype("int32")
    adv_img = np.clip(adv_img, 0, 255).astype("uint8")

    object_preds = box_pred.cpu().detach().numpy()
    # vis
    if is_vis:
        plt.imshow(img)
        plt.title("ori_img")
        plt.show()

        # plt.imshow(
        #     convert_tensor2_img(img_tensor_detach, unorm)
        # )
        # plt.title("ori_unorm")
        # plt.show()

        plt.imshow(adv_img)
        plt.title("adv")
        plt.show()

        plt.imshow(
            img_tensor_grad
        )
        plt.title("grad")
        plt.show()

        plt.imshow(
            perb_numpy
        )
        plt.title("perb")
        plt.show()

    return adv_img, iou_max_num, object_preds

    # model inference
    # pts_origin_xy 雷达投影到图像上的点坐标
    # img  图像
    # pts_input 雷达的点云坐标
    # print(input_data)
    # print(input_data.keys())
    # print(input_data["pts_input"].shape)
    # print(input_data["pts_origin_xy"].shape)
    # print(input_data["img"].shape)

    # print(ret_dict.keys())
    # roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
    # (B, M, 7)
    # seg_result = ret_dict['seg_result'].long()  # (B, N)

    # print(type(pred_boxes3d), pred_boxes3d.size())  # torch.Size([1, 100, 7])

    # # scoring
    # if rcnn_cls.shape[2] == 1:
    #
    #     # pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
    # else:
    #     raise ValueError()

    # print(norm_scores.shape) # torch.Size([1, 100, 1])
    # print(pred_classes, pre
    # d_classes.shape)

    # get adv_img

    # corners3d = kitti_utils.boxes3d_to_corners3d(pred_boxes3d_selected)

    # norm_scores_selected = norm_scores[0, inds]
    # print("======================")
    # print(pred_boxes3d_selected.size(), raw_scores_selected.size(), )  # norm_scores_selected.size()
    # print(raw_scores_selected)  # , norm_scores_selected

    # [x,7] [x,1] [x,1]
    #         if cur_inds.sum() == 0:
    #             continue
    #
    #         pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
    #         raw_scores_selected = raw_scores[k, cur_inds]
    #         norm_scores_selected = norm_scores[k, cur_inds]
    # inds = norm_scores > cfg.RCNN.SCORE_THRESH
    # print(inds)

    # evaluation
    recalled_num = gt_num = rpn_iou = 0
    # if not args.test:
    #     if not cfg.RPN.FIXED:
    #         rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
    #         rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking=True).long()
    #
    #     gt_boxes3d = data['gt_boxes3d']
    #
    #     for k in range(batch_size):
    #         # calculate recall
    #         cur_gt_boxes3d = gt_boxes3d[k]
    #         tmp_idx = cur_gt_boxes3d.__len__() - 1
    #
    #         while tmp_idx >= 0 and cur_gt_boxes3d[tmp_idx].sum() == 0:
    #             tmp_idx -= 1
    #
    #         if tmp_idx >= 0:
    #             cur_gt_boxes3d = cur_gt_boxes3d[:tmp_idx + 1]
    #
    #             cur_gt_boxes3d = torch.from_numpy(cur_gt_boxes3d).cuda(non_blocking=True).float()
    #             iou3d = iou3d_utils.boxes_iou3d_gpu(pred_boxes3d[k], cur_gt_boxes3d)
    #             gt_max_iou, _ = iou3d.max(dim=0)
    #             refined_iou, _ = iou3d.max(dim=1)
    #
    #             for idx, thresh in enumerate(thresh_list):
    #                 total_recalled_bbox_list[idx] += (gt_max_iou > thresh).sum().item()
    #             recalled_num += (gt_max_iou > 0.7).sum().item()
    #             gt_num += cur_gt_boxes3d.shape[0]
    #             total_gt_bbox += cur_gt_boxes3d.shape[0]
    #
    #             # original recall
    #             iou3d_in = iou3d_utils.boxes_iou3d_gpu(roi_boxes3d[k], cur_gt_boxes3d)
    #             gt_max_iou_in, _ = iou3d_in.max(dim=0)
    #
    #             for idx, thresh in enumerate(thresh_list):
    #                 total_roi_recalled_bbox_list[idx] += (gt_max_iou_in > thresh).sum().item()
    #
    #         if not cfg.RPN.FIXED:
    #             fg_mask = rpn_cls_label > 0
    #             correct = ((seg_result == rpn_cls_label) & fg_mask).sum().float()
    #             union = fg_mask.sum().float() + (seg_result > 0).sum().float() - correct
    #             rpn_iou = correct / torch.clamp(union, min=1.0)
    #             total_rpn_iou += rpn_iou.item()


#     disp_dict = {'mode': mode, 'recall': '%d/%d' % (total_recalled_bbox_list[3], total_gt_bbox)}
#     progress_bar.set_postfix(disp_dict)
#     progress_bar.update()
#
#     if args.save_result:
#         # save roi and refine results
#         roi_boxes3d_np = roi_boxes3d.cpu().numpy()
#         pred_boxes3d_np = pred_boxes3d.cpu().numpy()
#         roi_scores_raw_np = roi_scores_raw.cpu().numpy()
#         raw_scores_np = raw_scores.cpu().numpy()
#
#         rpn_cls_np = ret_dict['rpn_cls'].cpu().numpy()
#         rpn_xyz_np = ret_dict['backbone_xyz'].cpu().numpy()
#         seg_result_np = seg_result.cpu().numpy()
#         output_data = np.concatenate((rpn_xyz_np, rpn_cls_np.reshape(batch_size, -1, 1),
#                                       seg_result_np.reshape(batch_size, -1, 1)), axis=2)
#
#         for k in range(batch_size):
#             cur_sample_id = sample_id[k]
#             calib = dataset.get_calib(cur_sample_id)
#             image_shape = dataset.get_image_shape(cur_sample_id)
#             save_kitti_format(cur_sample_id, calib, roi_boxes3d_np[k], roi_output_dir,
#                               roi_scores_raw_np[k], image_shape)
#             save_kitti_format(cur_sample_id, calib, pred_boxes3d_np[k], refine_output_dir,
#                               raw_scores_np[k], image_shape)
#
#             output_file = os.path.join(rpn_output_dir, '%06d.npy' % cur_sample_id)
#             np.save(output_file, output_data.astype(np.float32))
#
#     # scores thresh
#     inds = norm_scores > cfg.RCNN.SCORE_THRESH
#     # print('cfg.RCNN.SCORE_THRESH:',cfg.RCNN.SCORE_THRESH)
#     # print('cfg.RCNN.NMS_THRESH:',cfg.RCNN.NMS_THRESH)
#
#     for k in range(batch_size):
#         cur_inds = inds[k].view(-1)
#         if cur_inds.sum() == 0:
#             continue
#
#         pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
#         raw_scores_selected = raw_scores[k, cur_inds]
#         norm_scores_selected = norm_scores[k, cur_inds]
#
#         # NMS thresh
#         # rotated nms
#         boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
#         keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
#         pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
#         scores_selected = raw_scores_selected[keep_idx]
#         pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().numpy(), scores_selected.cpu().numpy()
#
#         cur_sample_id = sample_id[k]
#         calib = dataset.get_calib(cur_sample_id)
#         final_total += pred_boxes3d_selected.shape[0]
#         image_shape = dataset.get_image_shape(cur_sample_id)
#         save_kitti_format(cur_sample_id, calib, pred_boxes3d_selected, final_output_dir, scores_selected,
#                           image_shape)
#
# progress_bar.close()
# # dump empty files
# split_file = os.path.join(dataset.imageset_dir, '..', '..', 'ImageSets', dataset.split + '.txt')
# split_file = os.path.abspath(split_file)
# image_idx_list = [x.strip() for x in open(split_file).readlines()]
# empty_cnt = 0
# for k in range(image_idx_list.__len__()):
#     cur_file = os.path.join(final_output_dir, '%s.txt' % image_idx_list[k])
#     if not os.path.exists(cur_file):
#         with open(cur_file, 'w') as temp_f:
#             pass
#         empty_cnt += 1
#         logger.info('empty_cnt=%d: dump empty file %s' % (empty_cnt, cur_file))
#
# ret_dict = {'empty_cnt': empty_cnt}
#
# logger.info('-------------------performance of epoch %s---------------------' % epoch_id)
# logger.info(str(datetime.now()))
#
# avg_rpn_iou = (total_rpn_iou / max(cnt, 1.0))
# avg_cls_acc = (total_cls_acc / max(cnt, 1.0))
# avg_cls_acc_refined = (total_cls_acc_refined / max(cnt, 1.0))
# avg_det_num = (final_total / max(len(dataset), 1.0))
# logger.info('final average detections: %.3f' % avg_det_num)
# logger.info('final average rpn_iou refined: %.3f' % avg_rpn_iou)
# logger.info('final average cls acc: %.3f' % avg_cls_acc)
# logger.info('final average cls acc refined: %.3f' % avg_cls_acc_refined)
# ret_dict['rpn_iou'] = avg_rpn_iou
# ret_dict['rcnn_cls_acc'] = avg_cls_acc
# ret_dict['rcnn_cls_acc_refined'] = avg_cls_acc_refined
# ret_dict['rcnn_avg_num'] = avg_det_num
#
# for idx, thresh in enumerate(thresh_list):
#     cur_roi_recall = total_roi_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
#     logger.info('total roi bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_roi_recalled_bbox_list[idx],
#                                                                       total_gt_bbox, cur_roi_recall))
#     ret_dict['rpn_recall(thresh=%.2f)' % thresh] = cur_roi_recall
#
# for idx, thresh in enumerate(thresh_list):
#     cur_recall = total_recalled_bbox_list[idx] / max(total_gt_bbox, 1.0)
#     logger.info('total bbox recall(thresh=%.3f): %d / %d = %f' % (thresh, total_recalled_bbox_list[idx],
#                                                                   total_gt_bbox, cur_recall))
#     ret_dict['rcnn_recall(thresh=%.2f)' % thresh] = cur_recall
#
# if cfg.TEST.SPLIT != 'test':
#     logger.info('Averate Precision:')
#     name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
#     from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate
#     ap_result_str, ap_dict = kitti_evaluate(dataset.label_dir, final_output_dir, label_split_file=split_file,
#                                             current_class=name_to_class[cfg.CLASSES])
#     logger.info(ap_result_str)
#     ret_dict.update(ap_dict)
#
# logger.info('result is saved to: %s' % result_dir)
# return ret_dict


class My_loss3D(nn.Module):

    def __init__(self):
        super().__init__()

    # x,y,z,....
    def forward(self, pred_box, target_box):
        x_center, y_center, z_center = self.get_center(pred_box)
        x_center2, y_center2, z_center2 = self.get_center(target_box)
        return (x_center - x_center2) ** 2 + (y_center - y_center2) ** 2 + (z_center - z_center2) ** 2

    def get_center(self, box):
        return box[0], box[1], box[2]


def get_input_data(data):
    sample_id, pts_rect, pts_features, pts_input = \
        data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input']
    batch_size = len(sample_id)
    inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()

    input_data = {'pts_input': inputs}
    # img feature
    if cfg.LI_FUSION.ENABLED:
        pts_origin_xy, img = data['pts_origin_xy'], data['img']
        pts_origin_xy = torch.from_numpy(pts_origin_xy).cuda(non_blocking=True).float()
        img = torch.from_numpy(img).cuda(non_blocking=True).float().permute((0, 3, 1, 2))  # similar np.transpose
        input_data['pts_origin_xy'] = pts_origin_xy
        input_data['img'] = img

    if cfg.RPN.USE_RGB or cfg.RCNN.USE_RGB:
        pts_rgb = data['rgb']
        pts_rgb = torch.from_numpy(pts_rgb).cuda(non_blocking=True).float()
        input_data['pts_rgb'] = pts_rgb
    return input_data, batch_size


def get_pred_box_tensor(ret_dict, batch_size, roi_boxes3d):
    rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
    # bounding box regression
    anchor_size = MEAN_SIZE

    pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                      anchor_size=anchor_size,
                                      loc_scope=cfg.RCNN.LOC_SCOPE,
                                      loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                      num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                      get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                      loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                      get_ry_fine=True).view(batch_size, -1, 7)
    return pred_boxes3d


def select_box(norm_scores, pred_boxes3d, raw_scores):
    inds = norm_scores > cfg.RCNN.SCORE_THRESH  # select box by conf
    inds = inds[0].view(-1)
    box_pred = pred_boxes3d[0, inds]
    raw_scores_selected = raw_scores[0, inds]
    boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(box_pred)
    #  select box by nms
    keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
    box_pred = box_pred[keep_idx]
    return box_pred


def my_attack(image, epsilon, data_grad):
    # print(torch.max(image), torch.min(image))
    # 收集数据梯度的元素符号
    # 通过调整输入图像的每个像素来创建扰动图像
    perb = epsilon * data_grad
    # perturbed_image = image + perb
    # perturbed_image = image + data_grad * 10 #pgd
    # 添加剪切以维持[0,1]范围
    # perturbed_image = torch.clamp(perturbed_image, 0, 255)
    # 返回被扰动的图像
    return perb


def save_kitti_format(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file=f)


def eval_single_ckpt(root_result_dir):
    # print(root_result_dir,"==========")
    root_result_dir = os.path.join(root_result_dir, 'eval')
    # set epoch_id and output dir
    num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    root_result_dir = os.path.join(root_result_dir, 'epoch_%s' % epoch_id, cfg.TEST.SPLIT)
    print(root_result_dir, "==========")
    if args.test:
        root_result_dir = os.path.join(root_result_dir, 'test_mode')

    if args.extra_tag != 'default':
        root_result_dir = os.path.join(root_result_dir, args.extra_tag)
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger=logger)

    # create dataloader & network
    test_loader = create_dataloader(logger)
    # model = PointRCNN(num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST')
    model = PointRCNN(num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST')
    model.cuda()

    # load checkpoint
    load_ckpt_based_on_args(model, logger)

    # attack
    attack(model, test_loader)


# --set  LI_FUSION.ENABLED True
# LI_FUSION.ADD_Image_Attention True
# RCNN.POOL_EXTRA_WIDTH 0.2
# RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False

# CUDA_VISIBLE_DEVICES=2 python eval_rcnn.py
# --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml
# --eval_mode rcnn_online
# --eval_all  --output_dir ./log/Car/full_epnet_without_iou_branch/eval_results/  --ckpt_dir ./log/Car/full_epnet_without_iou_branch/ckpt --set  LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention
# True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False


# CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --
# cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml
# --eval_mode rcnn_online
# --output_dir ./log/Car/models/full_epnet_without_iou_branch/eval_results/
# --ckpt ./log/Car/models/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_45.pth
# --set  LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2
# RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False


if __name__ == '__main__':
    # export PYTHONPATH=$PYTHONPATH:'/home/niangao/PycharmProjects/fusion/EPNet'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    args.cfg_file = "cfgs/LI_Fusion_with_attention_use_ce_loss.yaml"
    args.eval_mode = "rcnn_online"
    args.output_dir = "./log/Car/models/full_epnet_without_iou_branch/eval_results/"
    # args.ckpt_dir = "./log/Car/models/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_45.pth"
    args.ckpt = "./log/Car/models/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_45.pth"
    args.set_cfgs = ['LI_FUSION.ENABLED', 'True', 'LI_FUSION.ADD_Image_Attention', 'True', 'RCNN.POOL_EXTRA_WIDTH',
                     '0.2', 'RPN.SCORE_THRESH', '0.2', 'RCNN.SCORE_THRESH', '0.2', 'USE_IOU_BRANCH', 'False']
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    cfg.RCNN.ENABLED = True
    cfg.RPN.ENABLED = True
    cfg.RPN.FIXED = False

    cfg.LI_FUSION.ENABLED = True
    cfg.LI_FUSION.ADD_Image_Attention = True
    cfg.RCNN.POOL_EXTRA_WIDTH = 0.2
    cfg.RCNN.SCORE_THRESH = 0.2
    cfg.RCNN.USE_IOU_BRANCH = False

    # root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
    # ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')

    # ckpt_dir = args.ckpt_dir
    root_result_dir = args.output_dir
    os.makedirs(root_result_dir, exist_ok=True)

    print(args)
    print(cfg)
    eval_single_ckpt(root_result_dir)
