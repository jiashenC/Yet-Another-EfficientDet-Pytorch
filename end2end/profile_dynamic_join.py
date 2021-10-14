import os
import json

from collections import defaultdict
from itertools import combinations

import util
from profile_selectivity import process as best_process


def process(cls, count, video_path, used_model, use_gt=False):
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

    accuracy_std = [[] for _ in range(8)]
    # positive_count = []

    est_list = [util.Esti() for i in range(7)]

    def generate_sample_idx(start, end, existing_sample):
        span = (end - start) // util.chunk_sample
        target_sample_idx = list(range(start, end, span))

        existing_sample_list = []
        for i in range(8):
            for k in existing_sample[i]:
                if start <= k < end and k not in existing_sample_list:
                    existing_sample_list.append(k)

        actual_sample_idx = []
        if len(existing_sample_list) == 0:
            actual_sample_idx = target_sample_idx
        elif 0 < len(existing_sample_list) < util.chunk_sample:
            actual_sample_idx += existing_sample_list
            for k in existing_sample_list:
                best_dis = -1
                best_idx = -1
                for idx in target_sample_idx:
                    if abs(idx -
                           k) > best_dis and idx not in existing_sample_list:
                        best_idx = idx
                        best_dis = abs(idx - k)
                actual_sample_idx.append(best_idx)
                target_sample_idx.remove(best_idx)
                if len(actual_sample_idx) == util.chunk_sample:
                    break
        else:
            for idx in target_sample_idx:
                best_dis = 0xffffffff
                best_idx = -1
                for k in existing_sample_list:
                    if abs(idx - k) < best_dis:
                        best_idx = k
                        best_dis = abs(idx - k)
                actual_sample_idx.append(best_idx)
                existing_sample_list.remove(best_idx)
                if len(actual_sample_idx) == util.chunk_sample:
                    break

        return sorted(actual_sample_idx)

    def traverse(depth, start, end):
        nonlocal exec_plan, exec_sample_plan, exec_map, exec_sample_map, est_list, accuracy_std

        if depth == 1:
            used_model = list(range(8))
        else:
            used_model = util.best_n(est_list, n=2) + [7]
            used_model = sorted(used_model)

        sample_idx = generate_sample_idx(start, end, exec_sample_map)

        sample_plan = []
        useful_sample = 0

        chunk_accuracy = [[] for _ in range(9)]

        for i in sample_idx:
            best_plan = 7
            for m in range(7, -1, -1):
                if m not in used_model:
                    continue
                if i not in exec_sample_map[m]:
                    exec_plan.append(m)
                    exec_sample_plan.append(m)
                    exec_sample_map[m].append(i)
                if m != 7:
                    reward = util.evaluate(cls, count,
                                           res[m][str(i)]) == util.evaluate(
                                               cls, count, gt[str(i)])
                    est_list[m].update(reward)
                if util.evaluate(cls, count, res[m][str(i)]) == util.evaluate(
                        cls, count, gt[str(i)]):
                    best_plan = m

                    # if len(positive_count) == 0 or (sum(positive_count) / len(positive_count)) > util.positive_fraction:
                    accuracy_std[m].append(1)

                    # if util.evaluate(cls, count, gt[str(i)]):
                    #     positive_count.append(1)
                    # else:
                    #     positive_count.append(0)

                    # chunk_accuracy[m].append(1)
                else:
                    for k in range(m, -1, -1):
                        accuracy_std[k].append(0)
                        # chunk_accuracy[m].append(0)
                    break
            if util.evaluate(cls, count, skip_res) == util.evaluate(
                    cls, count, gt[str(i)]):
                best_plan = -1
                # chunk_accuracy[-1].append(1)
            # else:
            # chunk_accuracy[-1].append(0)
            sample_plan.append(best_plan)
            if util.evaluate(cls, count, gt[str(i)]):
                useful_sample += 1

        level_accuracy_std = []
        for m in used_model:
            level_accuracy_std += accuracy_std[m]

        # cost model comparision
        perf_benefit = (end - start) * util.time_dict[max(sample_plan)] \
                - (end - start) / 2 * util.time_dict[max(sample_plan[:len(sample_plan) // 2])] \
                - (end - start) / 2 * util.time_dict[max(sample_plan[len(sample_plan) // 2:])]
        cost = 10 * sum([util.time_dict[m] for m in used_model])

        # print(util.sample_lower_bound_v2(level_accuracy_std, len(used_model)))
        if (util.sample_lower_bound_v2(level_accuracy_std, len(used_model)) <=
                util.small_chunk and perf_benefit < cost
            ) or (abs(end - start) // 2) < util.small_chunk:
            sel_plan = max(sample_plan)
            # for m in used_model + [-1]:
            #     correct = sum(chunk_accuracy[m])
            #     if correct == 0:
            #         continue
            #     elif correct / len(chunk_accuracy[m]) >= util.e_acc:
            #         sel_plan = m

            for k in range(start, end):
                if k in exec_sample_map[7]:
                    exec_map[k] = 7
                elif k in exec_sample_map[sel_plan]:
                    exec_map[k] = sel_plan
                else:
                    exec_plan.append(sel_plan)
                    exec_map[k] = sel_plan
        else:
            span = (end - start) // 2
            traverse(depth + 1, start, start + span)
            traverse(depth + 1, start + span, end)

    traverse(1, 0, length)

    for i in range(length):
        if exec_map[i] == -1:
            exec_res[str(i)] = skip_res
        else:
            exec_res[str(i)] = res[exec_map[i]][str(i)]

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

    frame_id = []
    for i in range(length):
        if str(i) in exec_res and util.evaluate(cls, count, exec_res[str(i)]):
            frame_id.append(i)

    gt_frame_id = []
    for i in range(length):
        if str(i) in gt and util.evaluate(cls, count, gt[str(i)]):
            gt_frame_id.append(i)

    return exec_time / length, frame_id, gt_frame_id, exec_time - exec_sample_time, exec_sample_time


def main():
    res = []
    open('dynamic_join.log', 'w').close()

    for path in util.read_path():
        best_acc = best_process(path, use_gt=util.use_gt)

        if best_acc > util.majority:
            tmp_res = []
            for _ in range(5):
                exec_time, frame_id, gt_frame_id = 0, None, None
                exec_t, opt_t = 0, 0
                for i in range(len(util.cls)):
                    exec_time_t, frame_id_t, gt_frame_id_t, t_exec_t, t_opt_t = process(
                        [util.cls[i]], [util.count[i]],
                        path, [_ for _ in range(8)],
                        use_gt=util.use_gt)
                    exec_time += exec_time_t
                    exec_t += t_exec_t
                    opt_t += t_opt_t
                    if frame_id is None:
                        frame_id = set(frame_id_t)
                    else:
                        frame_id.intersection(set(frame_id_t))
                    if gt_frame_id is None:
                        gt_frame_id = set(gt_frame_id_t)
                    else:
                        gt_frame_id.intersection(set(gt_frame_id_t))

                speedup = util.time_dict[7] / exec_time
                precision = len(
                    gt_frame_id.intersection(frame_id)) / len(frame_id)
                recall = len(
                    gt_frame_id.intersection(frame_id)) / len(gt_frame_id)
                f1 = 2 / (1 / precision + 1 / recall)
                tmp_res.append((speedup, f1, exec_t, opt_t))

            speed = sum([tmp[0] for tmp in tmp_res]) / 5
            f1 = sum([tmp[1] for tmp in tmp_res]) / 5
            exe = sum([tmp[2] for tmp in tmp_res]) / 5
            opt = sum([tmp[3] for tmp in tmp_res]) / 5
            res += [(speed, f1, exe, opt)]
            print('-' * 73)

    with open('dynamic_join.log', 'a') as f:
        for r in res:
            f.write('{},{},{},{}\n'.format(r[0], r[1], r[2], r[3]))
        f.flush()
    f.close()


if __name__ == '__main__':
    main()
