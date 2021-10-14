import os
import json
from itertools import combinations

import util


def process(video_path, use_gt=True, verbose=False, other_stats=False):
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

    pos = 0
    for k in range(length):
        if util.evaluate(cls, count, gt[str(k)]):
            pos += 1
    if other_stats:
        return pos / length, length
    else:
        return pos / length


def main():
    open('sel.log', 'w').close()

    for path in util.read_path():
        sel = process(path,
                      use_gt=util.use_gt,
                      verbose=True,
                      other_stats=True)
        print(path, sel)
        if sel[0] > util.majority:
            with open('sel.log' ,'a') as f:
                f.write('{},{}\n'.format(sel[0], sel[1]))
                f.flush()


if __name__ == '__main__':
    main()
