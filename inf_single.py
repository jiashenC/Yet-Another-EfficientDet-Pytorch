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

LEN = 50
PICK_MODELS = [7]


class RandomDataset(Dataset):
    def __init__(self, size):
        super(RandomDataset, self).__init__()
        self.size = size

    def __len__(self):
        return LEN

    def __getitem__(self, index):
        return np.random.random_sample(self.size).astype(np.float32), 0


def run_inf(model, size, model_name, start_bs=64, logging=True):
    bs = start_bs
    finish = False

    if logging:
        print('Dataset Size:', size)

    while not finish:
        try:
            start = time.perf_counter()
            total_lat = 0
            num_iter = 0

            data_loader = DataLoader(dataset=RandomDataset(size),
                                     batch_size=bs,
                                     num_workers=1)
            with torch.no_grad():
                for _, (img, lb) in enumerate(data_loader):
                    iter_start = time.perf_counter()

                    img = img.cpu()
                    out = model(img)
                    # cuda.synchronize()

                    total_lat += time.perf_counter() - iter_start
                    num_iter += 1

            finish = True

            if logging:
                print('Batch Size:', bs)
                print('Latency(s): {:.2f}'.format(total_lat / num_iter))
                print('FPS: {:.2f}'.format(LEN /
                                           (time.perf_counter() - start)))

        except Exception as e:
            bs -= 2


def main():
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)]
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    print('Single models in memory ... ')
    for cp_coef in PICK_MODELS:
        print()
        print('Model: d' + str(cp_coef), '>>>')
        start = time.perf_counter()
        model = EfficientDetBackbone(compound_coef=cp_coef,
                                     num_classes=90,
                                     ratios=anchor_ratios,
                                     scales=anchor_scales)
        model.load_state_dict(
            torch.load(f'weights/efficientdet-d{cp_coef}.pth',
                       map_location='cpu'))
        model.requires_grad_(False)
        model.eval()
        model = model.cpu()
        print('Loading(s): {:.2f}'.format(time.perf_counter() - start))

        dim = input_sizes[cp_coef]
        size = (3, dim, dim)

        run_inf(model, size, cp_coef, start_bs=1)


if __name__ == '__main__':
    main()
