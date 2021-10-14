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

t_sum = 0
for i, t in enumerate(time_dict):
    t_sum += time_dict[i]
print(time_dict[-1] / t_sum)
