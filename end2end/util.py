import os
import copy
import math
import statistics
import numpy as np
import cv2

from functools import cmp_to_key

cls = [2]
count = [1]
dataset = 'jackson'
majority = 0.1

gap = 0.03
e_acc = 0.95

small_chunk = 100
chunk_sample = 10

use_gt = False

chunk_size = 1000

positive_fraction = 0.3

time_dict = [
    1 / 36.20, 1 / 29.69, 1 / 26.50, 1 / 22.73, 1 / 14.75, 1 / 7.11, 1 / 5.30,
    1 / 3.73
]


def has_all_pred(path):
    for m in range(8):
        pred_path = os.path.join(path, 'res-{}.json'.format(m))
        if not os.path.isfile(pred_path):
            return False
    return True


def read_path():
    global dataset

    if dataset == 'ua_detrac':
        base_dir = '/data/jiashenc/ua_detrac/test/'
        for path in os.listdir(base_dir):
            pred_dir = os.path.join(base_dir, path)
            if has_all_pred(pred_dir):
                yield pred_dir
    elif dataset == 'bdd':
        base_dir = '/data/jiashenc/bdd/bdd100k/videos/test'
        for path in os.listdir(base_dir):
            pred_dir = os.path.join(base_dir, path)
            if os.path.isfile(pred_dir):
                continue
            if has_all_pred(pred_dir):
                yield pred_dir
    elif dataset == 'jackson':
        base_dir = '/data/jiashenc/jackson_short/'
        # base_dir = '/data/jiashenc/jackson/'
        for path in os.listdir(base_dir):
            pred_dir = os.path.join(base_dir, path)
            if has_all_pred(pred_dir) and \
                    '11' not in pred_dir and \
                    '88' not in pred_dir and \
                    '90' not in pred_dir and \
                    '23' not in pred_dir:
                yield pred_dir
    elif dataset == 'virat':
        base_dir = '/data/jiashenc/virat/'
        for path in os.listdir(base_dir):
            pred_dir = os.path.join(base_dir, path)
            if 'VIRAT' in path and has_all_pred(pred_dir):
                yield pred_dir


def evaluate(cls, count, res, use_gt=False):
    for i in range(len(cls)):
        per_cls, per_count = cls[i], count[i]
        cls_count = 0
        for i, c in enumerate(res['class']):
            if c == per_cls and (use_gt or res['score'][i] >= 0.5):
                cls_count += 1
        if cls_count < per_count:
            return False
    return True


def best_n(est_list, n=1):
    sample_list = []
    for i, est in enumerate(est_list):
        sample_list += [(est.sample(), i)]

    def cmp_func(x, y):
        if x[0] != y[0]:
            return -(x[0] - y[0])
        else:
            return x[1] - y[1]

    sorted_sample_list = sorted(sample_list, key=cmp_to_key(cmp_func))

    return [sorted_sample_list[i][1] for i in range(n)]


class Esti:
    def __init__(self):
        self.tau = 0.0001
        self.phi = 1

        self.n = 0
        self.Q = 0

    def update(self, R):
        self.n += 1
        self.Q = (1 - 1 / self.n) * self.Q + (1 / self.n) * R

        self.phi = ((self.tau * self.phi) +
                    (self.n * self.Q)) / (self.tau + self.n)
        self.tau += 1

    def sample(self):
        return np.random.rand() / np.sqrt(self.tau) + self.phi


def find_majority(k):
    myMap = {}
    maximum = ('', 0)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1
        if myMap[n] > maximum[1]: maximum = (n, myMap[n])
    return maximum


def sample_lower_bound(correct_arr, num_model):
    return 2 * math.log(num_model) * (statistics.stdev(correct_arr)**2) / (gap
                                                                           **2)


def sample_lower_bound_v2(correct_arr, num_model):
    avg_acc = sum(correct_arr) / len(correct_arr)
    avg_dev = sum([abs(x - avg_acc) for x in correct_arr]) / len(correct_arr)
    if (e_acc + gap - avg_acc) == 0:
        return 1
    k = (math.log(num_model) *
            (4 * (avg_dev) + 2 *
             (e_acc + gap - avg_acc))) / ((e_acc + gap - avg_acc)**2)
    return k
