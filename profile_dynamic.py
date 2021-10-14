import os
import json
from itertools import combinations

import util
from profile_best import process as best_process


def process(video_path, used_model, chunk_size=60, use_gt=False):
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

    span = chunk_size // 4

    exec_plan, exec_res = [], {}
    exec_sample_plan = []
    plan_per_chunk = []

    sample_time = 0

    for i in range(0, length, chunk_size):
        start, end = i, min(length, i + chunk_size)

        sample_plan = []
        for k in range(start, end, span):
            sample_time += 1
            best_plan = 7
            for m in range(7, -1, -1):
                if m not in used_model:
                    continue

                exec_plan.append(m)
                exec_sample_plan.append(m)

                # if not util.evaluate(cls, count, gt[str(k)]):
                #     best_plan = -1
                #     break
                # elif util.evaluate(cls, count, res[m][str(k)]) and util.evaluate(cls, count, gt[str(k)]):
                #     best_plan = m
                # else:
                #     break

                if util.evaluate(cls, count, res[m][str(k)]) == util.evaluate(cls, count, gt[str(k)]):
                    best_plan = m
                else:
                    break

            sample_plan.append(best_plan)

        sel_plan = max(sample_plan)
        plan_per_chunk.append(sel_plan)
        for k in range(start, end):
            exec_plan.append(sel_plan)
            if k in [s for s in range(start, end, span)]:
                exec_res[str(k)] = gt[str(k)] if not use_gt else res[-1][str(k)]
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
    precision = 0 if (p == 0) else tp / p

    p = 0
    for k in range(length):
        if str(k) in gt and util.evaluate(cls, count, gt[str(k)], use_gt=use_gt):
            p += 1
    recall = 0 if (p == 0) else tp / p

    # model_str = '[' + ','.join([str(m) for m in used_model]) + ']'
    print('|{path:^{w}}|{speedup:^{w}.2f}|{sampling_ratio:^{w}.2f}|{opt_time_ratio:^{w}.2f}|{precision:^{w}.2f}|{recall:^{w}.2f}|'.format(path=video_path.split('/')[-1], speedup=speedup, sampling_ratio=sample_time / length * 100, opt_time_ratio=exec_sample_time / exec_time * 100, precision=precision * 100, recall=recall * 100, w=11))

    # plan distribution
    plan_count = dict()
    for i in range(-1, 8):
        plan_count[i] = 0
    for p in exec_plan:
        plan_count[p] += 1
    for p in exec_sample_plan:
        plan_count[p] -= 1
    # print(','.join(['{:.4f}'.format(plan_count[p] / length) for p in range(-1, 8)]))
    # print(','.join([str(p) for p in plan_per_chunk]))

    return speedup, 2 / (1 / precision + 1 / recall) if precision != 0 and recall != 0 else 0, sample_time / length * 100


def main():
    res = []
    open('dynamic.log', 'w').close()

    gap = util.gap

    use_gt = util.use_gt

    tmp_working = []

    # base_dir = '/data/jiashenc/ua_detrac/test/'
    base_dir = '/data/jiashenc/virat/'
    for video_path in os.listdir(base_dir):

        # if len(tmp_working) != 0 and video_path not in tmp_working:
        #     continue

        if len(tmp_working) != 0 and video_path not in tmp_working or 'VIRAT' not in video_path:
            continue

        skip = False
        for l in range(8):
            if not os.path.isfile(os.path.join(base_dir, video_path, 'res-{}.json'.format(l))):
                skip = True
        if skip:
            continue

        with open(os.path.join(base_dir, video_path, 'res-7.json')) as f:
            tmp_res = json.load(f)
            length = len(tmp_res)

        best_acc = best_process(os.path.join(base_dir, video_path), use_gt=use_gt)

        if best_acc > 0.1:
            res = []
            res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 25, use_gt=use_gt)]
            res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 50, use_gt=use_gt)]
            res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 100, use_gt=use_gt)]
            res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 200, use_gt=use_gt)]
            res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 500, use_gt=use_gt)]
            res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 1000, use_gt=use_gt)]
            res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], length, use_gt=use_gt)]

            final_res = (0, 0)
            for r in res:
                if r[0] > final_res[0] and best_acc - r[1] <= gap:
                    final_res = r

            with open('dynamic.log', 'a') as f:
                if best_acc - final_res[1] <= gap:
                    f.write('{},{}\n'.format(final_res[0], final_res[1]))
                else:
                    f.write('{},{}\n'.format(-1, -1))
                f.flush()
            f.close()

        print('-' * 73)

    # best_acc = best_process('/data/jiashenc/jackson/0/', use_gt=use_gt)

    # res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 20, use_gt=use_gt)]
    # res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 50, use_gt=use_gt)]
    # res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 100, use_gt=use_gt)]
    # res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 200, use_gt=use_gt)]
    # res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 500, use_gt=use_gt)]
    # res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 1000, use_gt=use_gt)]
    # res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 2000, use_gt=use_gt)]
    # res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 5000, use_gt=use_gt)]
    # res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 10000, use_gt=use_gt)]

    # final_res = (0, 0)
    # for r in res:
    #     if r[0] > final_res[0] and best_acc - r[1] <= gap:
    #         final_res = r

    # with open('dynamic.log', 'a') as f:
    #     if best_acc - final_res[1] <= gap:
    #         f.write('{},{}\n'.format(final_res[0], final_res[1]))
    #         f.flush()
    # f.close()


if __name__ == '__main__':
    main()
