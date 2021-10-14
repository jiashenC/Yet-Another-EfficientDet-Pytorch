import os
import json
from itertools import combinations

import util


def process(video_path, use_gt=True, verbose=False):
    cls = util.cls
    count = util.count

    skip_res = {'class': [], 'score': []}

    length = 0
    with open(os.path.join(video_path, 'res-7.json')) as f:
        exec_res = json.load(f)
        length = len(exec_res)

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
        print('{} | {:.2f} | {:.2f} |'.format(video_path, precision, recall))

    return 2 / (1 / precision + 1 / recall) if (recall != 0 and precision != 0) else 0


def main():
    tmp_working = []

    base_dir = '/data/jiashenc/ua_detrac/test/'
    for video_path in os.listdir(base_dir):

        if len(tmp_working) != 0 and video_path not in tmp_working:
            continue

        skip = False
        for l in range(8):
            if not os.path.isfile(os.path.join(base_dir, video_path, 'res-{}.json'.format(l))):
                skip = True
        if skip:
            continue

        acc = process(os.path.join(base_dir, video_path), use_gt=util.use_gt, verbose=True)

        with open('best.log', 'a') as f:
            if acc != 0:
                f.write('{},{}\n'.format(1, acc))
            else:
                f.write('{},{}\n'.format(-1, -1))
            f.flush()
        f.close()


if __name__ == '__main__':
    main()
