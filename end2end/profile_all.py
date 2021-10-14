import os
import json
import statistics

from itertools import combinations

import util


def process(video_path, i, use_gt=True, verbose=False):
    cls = util.cls
    count = util.count

    skip_res = {'class': [], 'score': []}

    length = 0
    with open(os.path.join(video_path, 'res-{}.json'.format(i))) as f:
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
            if str(k) in gt and util.evaluate(
                    cls, count, gt[str(k)], use_gt=use_gt):
                tp += 1
    precision = 0 if (p == 0) else tp / p

    p = 0
    for k in range(length):
        if str(k) in gt and util.evaluate(
                cls, count, gt[str(k)], use_gt=use_gt):
            p += 1
    recall = 0 if (p == 0) else tp / p

    if verbose:
        print('{} | {:.2f} | {:.2f} |'.format(video_path, precision, recall))

    return 2 / (1 / precision + 1 / recall) if (recall != 0
                                                and precision != 0) else 0


def main():
    for path in util.read_path():
        res = []
        for i in range(8):
            res += ['{:.2f}'.format(process(path, i, use_gt=util.use_gt, verbose=False))]
        print(' '.join(res))


if __name__ == '__main__':
    main()
