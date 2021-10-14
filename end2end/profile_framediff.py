import os
import cv2
import json
from itertools import combinations
from skimage.measure import compare_ssim

import util

from profile_selectivity import process as best_process


def get_img_diff(img_res, video_path, idx):
    path_idx = int(video_path.split('/')[-1])
    path = '/data/jiashenc/jackson/frame{}.jpg'.format(path_idx * 10000 + idx)
    diff = float(img_res[str(idx)])
    return diff


def process(video_path, use_gt=True, verbose=False):
    cls = util.cls
    count = util.count

    threshold = 0.9

    with open(os.path.join(video_path, 'framediff.json')) as f:
        diff_res = json.load(f)

    skip_res = {'class': [], 'score': []}

    res = []
    length = 0
    for i in range(8):
        with open(os.path.join(video_path, 'res-{:d}.json'.format(i))) as f:
            out_dict = json.load(f)
            res.append(out_dict)
            length = len(out_dict)

    gt = res[-1]

    exec_plan, exec_res = [], {}
    
    # execution
    model_invoke = True
    for k in range(length):
        img_diff = get_img_diff(diff_res, video_path, k)

        if img_diff < threshold:
            exec_res[str(k)] = gt[str(k)]
            exec_plan.append(7)
            model_invoke = util.evaluate(cls, count, exec_res[str(k)])
        else:
            if model_invoke:
                exec_res[str(k)] = gt[str(k)]
                exec_plan.append(7)
            else:
                exec_res[str(k)] = skip_res

    exec_time = 600
    for p in exec_plan:
        exec_time += util.time_dict[p]
    speedup = util.time_dict[7] / (exec_time / length)

    if use_gt:
        with open(os.path.join(video_path, 'gt.json')) as f:
            gt = json.load(f)
    else:
        with open(os.path.join(video_path, 'res-7.json')) as f:
            gt = json.load(f)

    tp, p = 0, 0
    for k in range(length):
        if util.evaluate(cls, count, exec_res[str(k)]):
            p += 1
            if str(k) in gt and util.evaluate(cls, count, gt[str(k)], use_gt=use_gt):
                tp += 1
    precision = 0 if (p == 0) else tp / p

    p = 0
    for k in range(length):
        if str(k) in gt and util.evaluate(cls, count, gt[str(k)], use_gt=use_gt):
            p += 1
    recall = 0 if (p == 0) else tp / p

    if verbose:
        print('{} | {:.2f} | {:.2f} | {:.2f} |'.format(video_path, speedup, precision, recall))

    return exec_time, 2 / (1 / precision + 1 / recall) if (recall != 0 and precision != 0) else 0


def main():
    open('framediff.log', 'w').close()

    for path in util.read_path():
        best_acc = best_process(path, use_gt=util.use_gt)

        if not os.path.isfile(os.path.join(path, 'framediff.json')):
            continue

        if best_acc > util.majority:
            speedup, acc = process(path, use_gt=util.use_gt, verbose=True)
            with open('framediff.log', 'a') as f:
                f.write('{},{}\n'.format(speedup, acc))
                f.flush()


if __name__ == '__main__':
    main()
