# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch import cuda
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


def main():
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    models = []
    for cp_coef in range(8):
        model = EfficientDetBackbone(compound_coef=cp_coef, num_classes=90, ratios=anchor_ratios, scales=anchor_scales)
        model.load_state_dict(torch.load(f'weights/efficientdet-d{cp_coef}.pth', map_location='cpu'))
        model.requires_grad_(False)
        model.eval()
        model = model.cuda()
        models.append(model)

    while True:
        time.sleep(1)


if __name__ == '__main__':
    main()
