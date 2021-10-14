import os
import json
from itertools import combinations

import util

from profile_selectivity import process as best_process


def process(video_path, use_gt=True, verbose=False):
    cls = util.cls
    count = util.count

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
    
    # optimize
    span = length // (10 * 10)
    sample_gt_res = {}
    for k in range(0, length, span):
        exec_plan.append(7)
        sample_gt_res[str(k)] = gt[str(k)]

    plan = -1

    for i in range(7):
        sample_res = {}
        for k in range(0, length, span):
            exec_plan.append(i)
            if util.evaluate(cls, [1 for _ in range(len(cls))], res[i][str(k)]):
                sample_res[str(k)] = gt[str(k)]

        tp, p = 0, 0
        for k in range(0, length, span):
            if str(k) in sample_res and util.evaluate(cls, count, sample_res[str(k)]):
                p += 1
                if str(k) in sample_gt_res and util.evaluate(cls, count, sample_gt_res[str(k)], use_gt=use_gt):
                    tp += 1
        precision = 0 if (p == 0) else tp / p

        p = 0
        for k in range(0, length, span):
            if str(k) in sample_gt_res and util.evaluate(cls, count, sample_gt_res[str(k)], use_gt=use_gt):
                p += 1
        recall = 0 if (p == 0) else tp / p

        plan = i
        if precision != 0 and recall != 0 and 2 / (1 / recall + 1 / precision) >= util.e_acc:
            break

    # execution
    for k in range(length):
        if k in list(range(0, length, span)):
            exec_res[str(k)] = gt[str(k)]
        else:
            exec_plan.append(plan)
            if util.evaluate(cls, [1 for _ in range(len(cls))], res[plan][str(k)]):
                exec_plan.append(7)
                exec_res[str(k)] = gt[str(k)]
            else:
                exec_res[str(k)] = skip_res

    exec_time = 0
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
    open('pp.log', 'w').close()

    for path in util.read_path():
        best_acc = best_process(path, use_gt=util.use_gt)

        if best_acc > util.majority:
            speedup, acc = process(path, use_gt=util.use_gt, verbose=True)
            with open('pp.log', 'a') as f:
                f.write('{},{}\n'.format(speedup, acc))
                f.flush()


if __name__ == '__main__':
    main()
