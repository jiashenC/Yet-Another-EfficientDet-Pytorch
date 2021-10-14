import os
import json

from functools import cmp_to_key

from collections import defaultdict
from itertools import combinations

import util
from profile_selectivity import process as best_process


def process(video_path, used_model, use_gt=False):
    cls = util.cls
    count = util.count
    gap = util.gap

    skip_res = {'class': [], 'score': []}

    res = []
    length = 0
    for i in range(8):
        with open(os.path.join(video_path, 'res-{:d}.json'.format(i))) as f:
            out_dict = json.load(f)
            res.append(out_dict)
            length = len(out_dict)

    gt = res[-1]

    exec_map, exec_sample_map = {}, defaultdict(lambda: [])

    exec_plan, exec_res = [], {}
    exec_sample_plan = []

    sample_time = 0

    m_list = [[0, i] for i in range(7)]

    def best_n_model(m_list, n=1):
        m_list = [(m[0], m[1]) for m in m_list]

        def cmp_func(x, y):
            if x[0] != y[0]:
                return -(x[0] - y[0])
            else:
                return x[1] - y[1]

        sorted_m_list = sorted(m_list, key=cmp_to_key(cmp_func))
        return [sorted_m_list[i][1] for i in range(n)]

    for i in range(0, length, 100):
        if i % 400 == 0:
            reward = [[0, i] for i in range(7)]
            use_model = list(range(7, -2, -1))
        else:
            use_model = [7] + best_n_model(m_list, n=2) + [-1]

        plan = []
        for k in range(i, min(i + 100, length), 10):
            for m in use_model:
                m_res = res[m][str(k)] if m != -1 else skip_res
                exec_plan.append(m)
                exec_sample_plan.append(m)
                if util.evaluate(cls, count, m_res) != util.evaluate(
                    cls, count, gt[str(k)]):
                    plan.append(m + 1)
                if i % 400 == 0 and m != 7:
                    reward = util.evaluate(cls, count,
                                           res[m][str(i)]) == util.evaluate(
                                               cls, count, gt[str(i)])
                    m_list[m][0] += reward

        sel_plan = max(plan) if len(plan) > 0 else -1
        for k in range(i, min(i + 100, length)):
            if k not in list(range(i, min(i + 100, length), 10)):
                exec_plan.append(sel_plan)
                exec_res[str(k)] = res[sel_plan][str(k)]
            else:
                exec_res[str(k)] = res[-1][str(k)]

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

    print(
        '|{path:^{w}}|{speedup:^{w}.2f}|{sampling_ratio:^{w}.2f}|{opt_time_ratio:^{w}.2f}|{precision:^{w}.2f}|{recall:^{w}.2f}|'
        .format(path=video_path.split('/')[-1],
                speedup=speedup,
                sampling_ratio=sample_time / length * 100,
                opt_time_ratio=exec_sample_time / exec_time * 100,
                precision=precision * 100,
                recall=recall * 100,
                w=11))

    return exec_time, 2 / (
        1 / precision + 1 / recall
    ) if precision != 0 and recall != 0 else 0, exec_sample_time, exec_time - exec_sample_time 


def main():
    res = []
    open('chameleon.log', 'w').close()
    open('chameleon_breakdown.log', 'w').close()

    for path in util.read_path():
        best_acc = best_process(path, use_gt=util.use_gt)

        if best_acc > util.majority:
            tmp_res = []
            for _ in range(1):
                tmp_res += [
                    process(path, [_ for _ in range(8)], use_gt=util.use_gt)
                ]
            speed = sum([tmp[0] for tmp in tmp_res])
            f1 = sum([tmp[1] for tmp in tmp_res])
            opt_t = sum([tmp[2] for tmp in tmp_res])
            exec_t = sum([tmp[3] for tmp in tmp_res])
            res += [(speed, f1, opt_t, exec_t)]
            print('-' * 73)

    with open('chameleon_breakdown.log', 'a') as f:
        for r in res:
            f.write('{},{},{},{}\n'.format(r[0], r[1], r[2], r[3]))
        f.flush()
    f.close()

    with open('chameleon.log', 'a') as f:
        for r in res:
            f.write('{},{}\n'.format(r[0], r[1]))
        f.flush()
    f.close()


if __name__ == '__main__':
    main()
