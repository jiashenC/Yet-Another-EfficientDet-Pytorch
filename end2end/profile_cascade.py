import os
import json
from itertools import combinations

import util
from profile_selectivity import process as best_process


def generate_conf_seq():
    for a in range(9, 4, -1):
        for b in range(a, 4, -1):
            for c in range(b, 4, -1):
                for d in range(c, 4, -1):
                    for e in range(d, 4, -1):
                        for f in range(e, 4, -1):
                            for g in range(f, 4, -1):
                                yield (a / 10, b / 10, c / 10, d / 10, e / 10,
                                       f / 10, g / 10)


def execute(res_arr, conf):
    cls = util.cls
    count = util.count

    res_count = [0 for _ in range(len(count))]
    stop_m = 7
    ret_res = None
    plan = []

    def greater(res, expected):
        for i, r in enumerate(res):
            if r < expected[i]:
                return False
        return True

    for i, res in enumerate(res_arr):
        for k, c in enumerate(res['class']):
            for e_idx in range(len(cls)):
                if i != 7 and c == cls[e_idx] and (res['score'][k] >= conf[i]):
                    res_count[e_idx] += 1

        plan.append(i)
        ret_res = res

        if greater(res_count, count):
            stop_m = i
            break

    golden_count = [0 for _ in range(len(cls))]
    for k, c in enumerate(res['class']):
        for e_idx in range(len(cls)):
            if c == cls[e_idx] and (res['score'][k] >= 0.5):
                golden_count[e_idx] += 1

    # calculate time
    time_dict = util.time_dict
    t = 0
    for i in range(0, stop_m + 1):
        t += time_dict[i]

    return ret_res, greater(res_count, count) == greater(golden_count, count), t, plan


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
        sample_res = []
        for m in range(7, -1, -1):
            if m not in used_model:
                continue

            exec_plan.append(m)
            exec_sample_plan.append(m)

            sample_res.append(res[m][str(i)])

        sample_plan.append(sample_res)

    exec_prof = dict()
    for c in generate_conf_seq():
        prof = dict()
        prof['correct'] = 0
        prof['t'] = 0

        for i, res_dict in enumerate(sample_plan):
            _, correct, t, _ = execute(res_dict, c)
            prof['correct'] += correct
            prof['t'] += t

        exec_prof[c] = prof

    # find most correct
    most_correct = 0
    for k, v in exec_prof.items():
        most_correct = max(most_correct, v['correct'])

    t = 0xffffffff
    conf = None
    for k, v in exec_prof.items():
        if v['t'] < t:
            conf = k
            t = v['t']

    for k in range(0, length):
        if k in [s for s in range(0, length, span)]:
            exec_res[str(k)] = res[-1][str(k)]
        else:
            res_dict = [res[i][str(k)] for i in range(8)]
            ret_res, _, _, plan = execute(res_dict, conf)
            exec_res[str(k)] = ret_res
            exec_plan += plan

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

    # model_str = '[' + ','.join([str(m) for m in used_model]) + ']'
    print(
        '|{path:^{w}}|{speedup:^{w}.2f}|{sampling_ratio:^{w}.2f}|{opt_time_ratio:^{w}.2f}|{precision:^{w}.2f}|{recall:^{w}.2f}|'
        .format(path=video_path.split('/')[-1],
                speedup=speedup,
                sampling_ratio=num_sample / length * 100,
                opt_time_ratio=exec_sample_time / exec_time * 100,
                precision=precision * 100,
                recall=recall * 100,
                w=11))

    return exec_time, 2 / (1 / precision + 1 / recall) if (
        precision != 0 and recall != 0) else 0


def main():
    res = []
    open('cascade.log', 'w').close()

    for path in util.read_path():
        with open(os.path.join(path, 'res-7.json')) as f:
            res = json.load(f)
            length = len(res)

        best_acc = best_process(path, use_gt=util.use_gt)

        if best_acc > util.majority:
            speedup, acc = process(path, 
                                   [_ for _ in range(8)], 
                                   length // 10,
                                   use_gt=util.use_gt)

            with open('cascade.log', 'a') as f:
                f.write('{},{}\n'.format(speedup, acc))
                f.flush()
        print('-' * 72)


if __name__ == '__main__':
    main()
