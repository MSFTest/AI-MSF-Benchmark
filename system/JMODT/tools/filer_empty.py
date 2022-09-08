import argparse
import os
import shutil

from jmodt.config import TRAIN_SEQ_ID, VALID_SEQ_ID, SMALL_VAL_SEQ_ID, TEST_SEQ_ID

if __name__ == '__main__':
    data_root = "/home/niangao/PycharmProjects/fusion/JMODT/data/KITTI/tracking_object"
    # seq_arr = ["val"]
    l = len(os.listdir("/home/niangao/PycharmProjects/fusion/JMODT/data/KITTI/tracking_object/training/image_2"))
    content = ["%06d" % n for n in list(range(0, l))]
    # for seq in seq_arr:
    #     tracking_seq = "{}/ImageSets/{}.txt".format(data_root, seq)
    res_seq = "{}/ImageSets/{}_filter.txt".format(data_root, "val")
    #     with open(tracking_seq, 'r') as f:
    #         lines = f.readlines()
    #         content = [line.strip() for line in lines]
    # print(content)
    content_res = content.copy()
    for fn_idx in content:
        label_fn = os.path.join(data_root, "training", "label_2", '{}.txt'.format(fn_idx))
        with open(label_fn, 'r') as f2:
            lb = f2.readline()
            if len(lb) == 0:
                content_res.remove(fn_idx)
            # print(lb)
    content_res = [c + "\n" for c in content_res]
    print(len(content_res),len(content))
    with open(res_seq, 'w') as f3:
        f3.writelines(content_res)
