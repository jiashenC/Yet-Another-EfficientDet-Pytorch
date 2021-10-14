import cv2
import math
from skimage.measure import compare_ssim
import time
import json

import util

from profile_selectivity import process as best_process


def main():
    diff_res = dict()

    for path in util.read_path():

        best_acc = best_process(path, use_gt=util.use_gt)
        print(path, best_acc)
        if best_acc <= util.majority:
            continue

        prev_img_arr = None

        s_t = time.perf_counter()
        for k in range(10000):
            if k % 100 == 0:
                print(path, k, time.perf_counter() - s_t)

            path_idx = int(path.split('/')[-1])
            img_arr = cv2.imread('/data/jiashenc/jackson/frame{}.jpg'.format(path_idx * 10000 + k))
            if prev_img_arr is None:
                diff = math.inf
            else:
                diff = compare_ssim(prev_img_arr, img_arr, multichannel=True)

            diff_res[k] = diff
            prev_img_arr = img_arr
        print(time.perf_counter() - s_t)

        with open('/data/jiashenc/jackson_short/{}/framediff.json'.format(path.split('/')[-1]), 'w') as f:
            json.dump(diff_res, f)
        diff_res = dict()


if __name__ == '__main__':
    main()
