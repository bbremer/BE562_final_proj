import sys
import time

import h5py
import numpy as np

import lab
from lab import Lab, Node, Bench

arg_names = ['pp', 'sp', 'ps', 'd', 'b', 'n', 'br', 'pr', 'lab_fn']
# pp, sp, ps, d, b, n, br, pr, lab_fn, out_fn = sys.argv[1:]
# args = sys.argv[1:]

pp = 0.00001
sp = 0.001
ps = 0.001
d = 200
b = .01
n = 5
br = 100000
pr = 24*7200*7
lab_fn = 'Lab_v1.p'
out_fn = 'test5.hdf'
args = [pp, sp, ps, d, b, n, br, pr, lab_fn]

L = lab.load_lab(lab_fn)
L.events = []
L.movements = []

num_benches = len(L.benches)

metadata = {k: v for k, v in zip(arg_names, args)}

arr_len = 3*n + num_benches + 1  # 1 for day
# save_array = np.empty((0, arr_len), np.int16)
save_array = []

L.people = [lab.Person() for _ in range(n)]

# 60 timepoint simulation
ims = []

t = time.time()
for day in range(10):
    data_t = 0.0
    save_t = 0.0
    bench_t = 0.0
    people_t = 0.0

    print(day)
    for p in L.people:
        p.daily_init(L, b, pr, day)
    for bench in L.benches:
        bench.infected = 0
        bench.timer = 0

    i = 0
    while any(p.in_lab for p in L.people):
        print(i, '\r', end='', flush=True)
        i += 1
        if i > 72000:
            print('in lab longer than 72 hours, something\'s wrong')
            sys.exit()
        s = time.time()
        a = L.data(day)
        data_t += time.time() - s

        s = time.time()
        # save_array = np.append(save_array, np.expand_dims(a, axis=0), axis=0)
        save_array.append(a)
        save_t += time.time() - s

        s = time.time()
        for bench in L.benches:
            bench.update(L, day, i)

        bench_t += time.time() - s

        s = time.time()
        for p in L.people:
            p.update(L, pp, sp, ps, d, pr, br, day, i)
        people_t += time.time() - s

    print(f'data: {data_t}')
    print(f'save: {save_t}')
    print(f'bench: {bench_t}')
    print(f'people: {people_t}')

print(time.time() - t)
save_array = np.array(save_array, dtype=np.int16)


with h5py.File(out_fn, 'w') as f:
    g = f.create_group('g')
    dataset = g.create_dataset('default', data=save_array)
    e = g.create_dataset('events', data=L.events)
    v = g.create_dataset('movements', data=L.movements)

    # print(L.movements)

    for arg_name, arg in metadata.items():
        g.attrs[arg_name] = arg
