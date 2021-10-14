import os
import json
from itertools import combinations


time_dict = [
    1/36.20,
    1/29.69,
    1/26.50,
    1/22.73,
    1/14.75,
    1/7.11,
    1/5.30,
    1/3.73
]


def evaluate(cls, count, res, use_gt=False):
    cls_count = 0
    for i, c in enumerate(res['class']):
        if c == cls and (use_gt or res['score'][i] >= 0.5):
            cls_count += 1 
    return cls_count >= count 


def process(video_path):
    cls = 5
    count = 1

    skip_res = {'class': [], 'score': []}

    length = 0
    with open(os.path.join(video_path, 'res-7.json')) as f:
        gt = json.load(f)
        length = len(gt)

    # find model
    sel_model = 7
    for i in range(6, -1, -1):
        with open(os.path.join(video_path, 'res-{}.json'.format(i))) as f:
            exec_res = json.load(f)

        tp, p = 0, 0
        for k in range(length):
            if evaluate(cls, count, exec_res[str(k)]):
                p += 1
                if str(k) in gt and evaluate(cls, count, gt[str(k)], use_gt=False):
                    tp += 1
        precision = 1 if (p == tp == 0) else tp / p

        p = 0
        for k in range(length):
            if str(k) in gt and evaluate(cls, count, gt[str(k)], use_gt=False):
                p += 1
        recall = 1 if (p == tp == 0) else tp / p

        if recall - 1 >= -0.03 and precision - 1 >= -0.03:
            sel_model = i

    exec_plan, exec_res = [], {}
    with open(os.path.join(video_path, 'res-{}.json'.format(0))) as f:
        hybrid_res = json.load(f)
    with open(os.path.join(video_path, 'res-{}.json'.format(sel_model))) as f:
        end_res = json.load(f)

    for i in range(length):
        # exec_plan.append(0)
        # if sel_model != 0 and evaluate(cls, 1, hybrid_res[str(i)]):
        #     exec_plan.append(sel_model)
        #     exec_res[str(i)] = end_res[str(i)]
        # elif sel_model == 0:
        #     exec_res[str(i)] = end_res[str(i)]
        exec_plan.append(sel_model)
        exec_res[str(i)] = end_res[str(i)]

    exec_time = 0
    for e in exec_plan:
        if e == -1:
            continue
        exec_time += time_dict[e]

    speedup = time_dict[-1] / (exec_time / length)

    tp, p = 0, 0
    for k in range(length):
        if str(k) in exec_res and evaluate(cls, count, exec_res[str(k)]):
            p += 1
            if str(k) in gt and evaluate(cls, count, gt[str(k)], use_gt=False):
                tp += 1
    precision = 1 if (p == tp == 0) else tp / p

    p = 0
    for k in range(length):
        if str(k) in gt and evaluate(cls, count, gt[str(k)], use_gt=False):
            p += 1
    recall = 1 if (p == tp == 0) else tp / p

    print('{},{},{:.2f},{:.4f},{:.4f}'.format(video_path, sel_model, speedup, precision, recall))


def main():
    tmp_working = []

    base_dir = '/data/jiashenc/ua_detrac/test/'
    for video_path in os.listdir(base_dir):

        if len(tmp_working) != 0 and video_path not in tmp_working:
            continue

        skip = False
        for l in range(8):
            if not os.path.isfile(os.path.join(base_dir, video_path, 'res-{}.json'.format(l))):
                skip = True
        if skip:
            continue

        process(os.path.join(base_dir, video_path))


if __name__ == '__main__':
    main()
