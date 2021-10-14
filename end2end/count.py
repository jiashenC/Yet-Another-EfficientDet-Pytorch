from collections import defaultdict

count = defaultdict(lambda: 0)
with open('model.log') as f:
    for line in f.read().splitlines():
        count[int(line)] += 1

for k, v in count.items():
    print('{},{}'.format(k, v))
