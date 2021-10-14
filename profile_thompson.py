import os
import json
import random 
import numpy as np
import scipy.stats as stats
from itertools import combinations
import matplotlib.pyplot as plt

import util
from profile_best import process as best_process


lower_bound = 1 - util.gap


def random_argmax(value_list):
    """ a random tie-breaking argmax"""
    values = np.asarray(value_list)
    for i, v in enumerate(values):
        if v >= lower_bound:
            return i
    return 7


class BernoulliThompson:
    def __init__(self):
        self.alpha = 1  # the number of times this socket returned a charge        
        self.beta = 1  # the number of times no charge was returned
        self.n = 0
        
    def charge(self, correct):        
        return correct
                    
    def update(self, R):
        """ increase the number of times this socket has been used and 
            update the counts of the number of times the socket has and 
            has not returned a charge (alpha and beta)"""
        self.n += 1    
        self.alpha += R
        self.beta += (1 - R)
        
    def sample(self):
        """ return a value sampled from the beta distribution """
        return np.random.beta(self.alpha, self.beta)


def process(video_path, used_model, sample_span, use_gt=False):
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

    if use_gt:
        with open(os.path.join(video_path, 'gt.json')) as f:
            gt = json.load(f)
    else:
        gt = res[-1]

    exec_plan, exec_sample_plan = [], []
    exec_res = dict()

    for start in range(0, length, util.chunk_size):

        est_list = [BernoulliThompson() for _ in range(8)]
        skip_est = BernoulliThompson()

        expl_range = list(range(start, min(start + util.chunk_size, length), sample_span * 3))

        # exploration
        for i in expl_range:
            for k in range(8):
                exec_sample_plan += [k]
                exec_plan += [k]
                reward = 1 if (util.evaluate(cls, count, res[k][str(i)]) == util.evaluate(cls, count, gt[str(i)], use_gt=use_gt)) else 0
                est_list[k].update(reward)

            reward = 1 if (util.evaluate(cls, count, skip_res) == util.evaluate(cls, count, gt[str(i)], use_gt=use_gt)) else 0
            skip_est.update(reward)

        # converged sampling: no exploration but exploitation
        for i in range(start, min(start + util.chunk_size, length), sample_span):
            if i in expl_range:
                continue
            est_idx = random_argmax([est.sample() for est in est_list] + [skip_est.sample()])
            if est_idx < 8:
                exec_sample_plan += [est_idx]
                exec_plan += [est_idx]
                reward = 1 if (util.evaluate(cls, count, res[est_idx][str(i)]) == util.evaluate(cls, count, gt[str(i)], use_gt=use_gt)) else 0
                est_list[est_idx].update(reward)
            else:
                reward = 1 if (util.evaluate(cls, count, skip_res) == util.evaluate(cls, count, gt[str(i)], use_gt=use_gt)) else 0
                skip_est.update(reward)

        est_str = []
        for est in est_list:
            est_str +=  ['{:.2f}'.format(est.alpha / (est.beta + est.alpha))]

        best_plan = random_argmax([est.sample() for est in est_list] + [skip_est.sample()])
        print('|'.join(est_str), best_plan)
        if best_plan < 8:
            for i in range(start, min(start + util.chunk_size, length)):
                if i in [range(start, start + util.chunk_size, sample_span)]:
                    exec_res[str(i)] = res[-1][str(i)]
                else:
                    exec_plan += [best_plan]
                    exec_res[str(i)] = res[best_plan][str(i)]
        else:
            for i in range(start, start + util.chunk_size):
                if i in [range(start, start + util.chunk_size, sample_span)]:
                    exec_res[str(i)] = res[-1][str(i)]
                else:
                    exec_res[str(i)] = skip_res

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
    print('|{path:^{w}}|{speedup:^{w}.2f}|{opt_time_ratio:^{w}.2f}|{precision:^{w}.2f}|{recall:^{w}.2f}|'.format(path=video_path.split('/')[-1], speedup=speedup, opt_time_ratio=exec_sample_time / exec_time * 100, precision=precision * 100, recall=recall * 100, w=11))


def main():
    res = []

    gap = util.gap

    use_gt = util.use_gt

    # tmp_working = []

    # base_dir = '/data/jiashenc/ua_detrac/test/'
    # for video_path in os.listdir(base_dir):

    #     if len(tmp_working) != 0 and video_path not in tmp_working:
    #         continue

    #     skip = False
    #     for l in range(8):
    #         if not os.path.isfile(os.path.join(base_dir, video_path, 'res-{}.json'.format(l))):
    #             skip = True
    #     if skip:
    #         continue

    #     with open(os.path.join(base_dir, video_path, 'res-7.json')) as f:
    #         tmp_res = json.load(f)
    #         length = len(tmp_res)

    #     best_acc = best_process(os.path.join(base_dir, video_path), use_gt=use_gt)

    #     if best_acc > 0.1:
    #         res = []
    #         # res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 1, use_gt=use_gt)]
    #         # res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 2, use_gt=use_gt)]
    #         # res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 5, use_gt=use_gt)]
    #         res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 5, use_gt=use_gt)]
    #         # res += [process(os.path.join(base_dir, video_path), [_ for _ in range(8)], 100, use_gt=use_gt)]

    best_acc = best_process('/data/jiashenc/jackson/0/', use_gt=use_gt)

    res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 10, use_gt=use_gt)]
    res += [process('/data/jiashenc/jackson/0/', [_ for _ in range(8)], 20, use_gt=use_gt)]


if __name__ == '__main__':
    main()
