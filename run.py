import sys

import h5py
import numpy as np

import lab
from lab import Lab, Node, Bench

arg_names = ['pp', 'sp', 'ps', 'd', 'b', 'n', 'br', 'pr', 'lab_fn']
# pp, sp, ps, d, b, n, br, pr, lab_fn, out_fn = sys.argv[1:]

pp = 0.00001
sp = 0.0001
ps = 0.00001
d = 200
b = .001
n = 5
br = 100000
pr = 24*7200*7
lab_fn = 'Lab_v1.p'
out_fn = 'test.hdf'

L = lab.load_lab(lab_fn)

num_benches = len(L.benches)

metadata = {k: v for k, v in zip(arg_names, sys.argv[1:-1])}
arr_len = 3*n + num_benches + 1  # 1 for day
save_array = np.empty((0, arr_len), np.int16)

L.people = [lab.Person() for _ in range(n)]

# 60 timepoint simulation
ims = []
for day in range(30):
    import time
    s = time.time()
    print(day)
    for p in L.people:
        p.daily_init(L, b, pr)
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
        a = L.data(day)
        save_array = np.append(save_array, np.expand_dims(a, axis=0), axis=0)

        for bench in L.benches:
            bench.update(L)
        for p in L.people:
            p.update(L, pp, sp, ps, d, pr, br)

    print(time.time() - s)

with h5py.File(out_fn, 'w') as f:
    dataset = f.create_dataset('data', save_array)

    for arg_name, arg in metadata:
        dataset.attrs[arg_name] = arg
