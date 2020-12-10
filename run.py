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
lab_fn = 'Lab_v1.py'
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
    for p in L.people:
        p.daily_init(L)
    for b in L.benches:
        b.infected = 0
        b.timer = 0

    i = 0
    while any(p.in_lab for p in L.people):
        i += 1
        if i > 72000:
            print('in lab longer than 72 hours, something\'s wrong')
            sys.exit()
        save_array = np.append(save_array, L.data(day), axis=0)

        for b in L.benches:
            b.update(L, br)
        for p in L.people:
            p.update()

with h5py.File(out_fn, 'w') as f:
    dataset = f.create_dataset('data', save_array)

    for arg_name, arg in metadata:
        dataset.attrs[arg_name] = arg
