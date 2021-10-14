import os
import json
from itertools import combinations

import util
from profile_selectivity import process as best_process


def process(video_path, use_gt=True, verbose=False):
    cls = util.cls
    count = util.count

    skip_res = {'class': [], 'score': []}

    length = 0
    with open(os.path.join(video_path, 'res-7.json')) as f:
        exec_res = json.load(f)
        length = len(exec_res)

    res = []
    length = 0
    for i in range(8):
        with open(os.path.join(video_path, 'res-{:d}.json'.format(i))) as f:
            out_dict = json.load(f)
            res.append(out_dict)
            length = len(out_dict)

    if use_gt:
        with open(os.path.join(video_path, 'gt.json')) as f:
            gt = json.load(f)
    else:
        with open(os.path.join(video_path, 'res-7.json')) as f:
            gt = json.load(f)

    exec_plan = []
    for k in range(length):
        best_plan = -1
        for m in range(0, 8):
            if util.evaluate(cls, count, gt[str(k)]) and util.evaluate(
                    cls, count, gt[str(k)]) == util.evaluate(
                        cls, count, res[m][str(k)]):
                best_plan = m
                break
        exec_plan.append(best_plan)

    exec_time = 0
    for p in exec_plan:
        exec_time += util.time_dict[p]
    return exec_time


def main():
    open('dynamic.log', 'w').close()

    for path in util.read_path():
        best_acc = best_process(path, use_gt=util.use_gt)

        if best_acc > util.majority:
            sel = process(path, use_gt=util.use_gt, verbose=True)
            with open('dynamic.log', 'a') as f:
                f.write('{:.2f}\n'.format(sel))
                f.flush()


if __name__ == '__main__':
    main()
