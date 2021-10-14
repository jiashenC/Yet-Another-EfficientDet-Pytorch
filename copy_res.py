import os
import glob


def copy_res(path):
    if 'ua_detrac' in path:
        copy_path = '/data/jiashenc/ua_detrac_res'
    elif 'virat' in path:
        copy_path = '/data/jiashenc/virat_res'
    elif 'bdd' in path:
        copy_path = '/data/jiashenc/bdd_res'
    elif 'jackson' in path:
        copy_path = '/data/jiashenc/jackson_res'
    else:
        raise Exception('No path')

    if not os.path.isdir(copy_path):
        os.system('mkdir ' + copy_path)

    for pred_path in os.listdir(path):
        if os.path.isfile(os.path.join(path, pred_path)):
            continue
        if 'virat' in path and 'VIRAT' not in pred_path:
            continue

        copy_from_path = os.path.join(path, pred_path)
        if not os.path.isdir(copy_from_path):
            os.system('mkdir ' + copy_from_path)

        copy_to_path = os.path.join(copy_path, pred_path)
        for res_path in glob.glob(os.path.join(copy_from_path, 'res-*.json')):
            file_path = res_path.split('/')[-1]
            res_full_path = os.path.join(copy_from_path, res_path)
            cp_full_path = os.path.join(copy_to_path, file_path)
            print(res_full_path, cp_full_path)
            # os.system('cp ' + res_full_path + ' ' + copy_to_path)


def main():
    copy_res('/data/jiashenc/ua_detrac/test/')


if __name__ == '__main__':
    main()
