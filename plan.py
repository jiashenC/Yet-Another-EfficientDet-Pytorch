import json 
import os


def evaluate(cls, count, res):
    cls_count = 0
    for i, c in enumerate(res['class']):
        if c == cls and res['score'][i] >= 0.5:
            cls_count += 1
    return cls_count >= count 


def plan(path):
    cls = 2
    count = 4

    res = []
    length = 0
    for i in range(8):
        with open(os.path.join(path, 'res-{:d}.json'.format(i))) as f:
            out_dict = json.load(f)
            res.append(out_dict)
            length = len(out_dict)

    gt = res[-1]
    plan = []
    for i in range(length):
        best_plan = 7
        for k in range(7, -1, -1):
            if not evaluate(cls, count, gt[str(i)]):
                best_plan = -1
                break
            elif evaluate(cls, count, res[k][str(i)]) and evaluate(cls, count, gt[str(i)]):
                best_plan = k
            else:
                break
        
        plan.append(best_plan)
    
    count_dict = dict()
    for i in range(8):
        count_dict[i] = 0
    for p in plan:
        if p == -1:
            continue
        count_dict[p] += 1

    res_str = '|'
    for i in range(8):
        sec = '{:.2f}'.format(count_dict[i] / len(plan) * 100) if count_dict[i] != 0 else '-'
        res_str += '{sec: ^{width}}|'.format(sec=sec, width=9)
    print(res_str)
    print(plan)


def main():
    res_str = '|'
    for i in range(8):
        res_str += '{i: ^{width}}|'.format(i=i, width=9)
    print('-' * 81)
    print(res_str)
    print('-' * 81)

    base_dir = '/data/jiashenc/ua_detrac/test/'
    for video_path in os.listdir(base_dir):
        path = os.path.join(base_dir, video_path)

        skip = False
        for l in range(8):
            if not os.path.isfile(os.path.join(path, 'res-{}.json'.format(l))):
                skip = True
        if skip:
            continue

        plan(path)
    print('-' * 81)


if __name__ == '__main__':
    main()
