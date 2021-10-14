import os
import json
from itertools import combinations

import util
from profile_selectivity import process as best_process


def process(video_path, used_model, num_sample, use_gt=False):
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

    span = length // num_sample

    exec_plan, exec_res = [], {}
    exec_sample_plan = []
    sample_plan = []

    for i in range(0, length, span):
        best_plan = 7
        for m in range(7, -1, -1):
            if m not in used_model:
                continue

            exec_plan.append(m)
            exec_sample_plan.append(m)

            if not util.evaluate(cls, count, gt[str(i)]):
                best_plan = -1
                break
            elif util.evaluate(cls, count, res[m][str(i)]) and util.evaluate(cls, count, gt[str(i)]):
                best_plan = m
            else:
                break
        sample_plan.append(best_plan)

    sel_plan = max(sample_plan)
    for k in range(0, length):
        exec_plan.append(sel_plan)
        if k in [s for s in range(0, length, span)]:
            exec_res[str(k)] = res[-1][str(k)]
        elif sel_plan != -1:
            exec_res[str(k)] = res[sel_plan][str(k)]
        else:
            exec_res[str(k)] = skip_res

    assert len(exec_res) == length

    if use_gt:
        with open(os.path.join(video_path, 'gt.json')) as f:
            gt = json.load(f)

    exec_time = 0
    for e in exec_plan:
        if e == -1:
            continue
        exec_time += util.time_dict[e]

    exec_sample_time = 0
    for e in exec_sample_plan:
        if e == -1:
            continue
        exec_sample_time += util.time_dict[e]

    speedup = util.time_dict[-1] / (exec_time / length)

    tp, p = 0, 0
    for k in range(length):
        if util.evaluate(cls, count, exec_res[str(k)]):
            p += 1
            if str(k) in gt and util.evaluate(cls, count, gt[str(k)], use_gt=use_gt):
                tp += 1
    precision = 1 if (p == 0) else tp / p

    p = 0
    for k in range(length):
        if str(k) in gt and util.evaluate(cls, count, gt[str(k)], use_gt=use_gt):
            p += 1
    recall = 1 if (p == 0) else tp / p

    # model_str = '[' + ','.join([str(m) for m in used_model]) + ']'
    print('|{path:^{w}}|{plan:^{w}}|{speedup:^{w}.2f}|{sampling_ratio:^{w}.2f}|{opt_time_ratio:^{w}.2f}|{precision:^{w}.2f}|{recall:^{w}.2f}|'.format(path=video_path.split('/')[-1], plan=sel_plan, speedup=speedup, sampling_ratio=num_sample / length * 100, opt_time_ratio=exec_sample_time / exec_time * 100, precision=precision * 100, recall=recall * 100, w=11))

    return exec_time, 2 / (1 / precision + 1 / recall) if (precision != 0 and recall != 0) else 0


def main():
    res = []
    open('static.log', 'w').close()

    for path in util.read_path():
        with open(os.path.join(path, 'res-7.json')) as f:
            tmp_res = json.load(f)
            length = len(tmp_res)

        best_acc = best_process(path, use_gt=util.use_gt)

        if best_acc > util.majority:
            speedup, acc = process(path, [_ for _ in range(8)], length // 100, use_gt=util.use_gt)

            with open('static.log', 'a') as f:
                f.write('{},{}\n'.format(speedup, acc))
                f.flush()


if __name__ == '__main__':
    main()
