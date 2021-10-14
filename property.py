import os
import json
from itertools import combinations

import util


def process(video_path):
    cls = util.cls
    count = util.count

    skip_res = {'class': [], 'score': []}

    length = 0
    with open(os.path.join(video_path, 'res-7.json')) as f:
        exec_res = json.load(f)
        length = len(exec_res)

    with open(os.path.join(video_path, 'gt.json')) as f:
        gt = json.load(f)

    tp = 0
    for k in range(length):
        if str(k) in gt and util.evaluate(cls, count, gt[str(k)], use_gt=True):
            tp += 1
    print(tp / length)


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

        process(os.path.join(base_dir, video_path))


if __name__ == '__main__':
    main()
