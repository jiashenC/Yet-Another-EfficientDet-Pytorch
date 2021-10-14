import numpy as np

from sys import argv


def main():
    data = []
    with open(argv[1]) as f:
        for line in f.read().splitlines():
            if len(data) == 0:
                for n in line.split(','):
                    data.append([float(n)])
            else:
                for i, n in enumerate(line.split(',')):
                    data[i].append(float(n))
    avg_str = ['{:.5f}'.format(np.average(d)) for d in data]
    var_str = ['{:.5f}'.format(np.std(d)) for d in data]
    print('Average')
    print(','.join(avg_str))
    print('Variance')
    print(','.join(var_str))


if __name__ == '__main__':
    main()
