import numpy as np

from sys import argv


def main():
    data = []
    with open(argv[1]) as f:
        for line in f.read().splitlines():
            if len(data) == 0:
                for _ in line.split(','):
                    data.append([])
            for i, n in enumerate(line.split(',')):
                data[i].append(float(n))
    data_str = ['{:.5f}'.format(np.median(d)) for d in data]
    print(','.join(data_str))


if __name__ == '__main__':
    main()
