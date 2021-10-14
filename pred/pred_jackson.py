# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import os 
import json
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


def to_json(out, out_dict):
    cls = [int(c) for c in out[0]['class_ids']]
    score = [float(s) for s in out[0]['scores']]
    st = {'class': cls, 'score': score}
    out_dict[len(out_dict)] = st


def main(i):
    compound_coef = i
    force_input_size = None  # set None to use default size

    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = 0.2
    iou_threshold = 0.2

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

    out_dict = dict()
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                    ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()

    base_dir = '/data/jiashenc/jackson/'

    print('Processing Det-' + str(i))

    for k in range(300000, 400000):
        if k % 1000 == 0:
            print('    Finish {} frames'.format(k + 1))
            
        img_path = os.path.join(base_dir, 'frame{}.jpg'.format(k))
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)


        out = invert_affine(framed_metas, out)
        to_json(out, out_dict)

    with open(os.path.join(base_dir, '3', 'res-{:d}.json'.format(i)), 'w') as f:
        json.dump(out_dict, f)
        out_dict = dict()


if __name__ == '__main__':
    for m in range(0, 8):
        main(m)
