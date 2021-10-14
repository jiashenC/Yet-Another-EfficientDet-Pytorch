from sys import argv


def main():
    path = argv[1]

    t_opt_t, t_exec_t = 0, 0
    count = 0
    with open(path) as f:
        for line in f.read().splitlines():
            opt_t, opt_r = line.split(',')
            opt_t = float(opt_t)
            opt_r = float(opt_r)
            t_opt_t += opt_t
            t_exec_t += opt_t / opt_r - opt_t
            count += 1

    print(t_exec_t / count, t_opt_t / count)


if __name__ == '__main__':
    main()
