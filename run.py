import multiprocessing
import sys
import time

import h5py
import numpy as np

import lab
from lab import Lab, Node, Bench

arg_names = ['sp', 'ps', 'd', 'br', 'pr', 'lab_fn', 'pp', 'n', 'b']
# pp, sp, ps, d, b, n, br, pr, lab_fn, out_fn = sys.argv[1:]
# args = sys.argv[1:]


sp = 0.000001
ps = 0.0001
d = 200
br = 100000
pr = 24*7200*7
lab_fn = 'Lab_v1.p'
global_args = [sp, ps, d, br, pr, lab_fn]


def run(cond_i, rep_i, pp, n, b):
    out_fn = f'cond-{cond_i}_rep-{rep_i}.hdf'

    args = global_args + [pp, n, b]

    L = lab.load_lab(lab_fn)
    L.events = []
    L.movements = []

    metadata = {k: v for k, v in zip(arg_names, args)}

    save_array = []

    L.people = [lab.Person() for _ in range(n)]

    # t = time.time()
    for day in range(30):
        data_t = 0.0
        save_t = 0.0
        bench_t = 0.0
        people_t = 0.0

        print(cond_i, rep_i, day)
        for p in L.people:
            p.daily_init(L, b, pr, day)
        for bench in L.benches:
            bench.infected = 0
            bench.timer = 0

        i = 0
        while any(p.in_lab for p in L.people):
            # print(i, '\r', end='', flush=True)
            i += 1
            if i > 72000:
                print('in lab longer than 72 hours, something\'s wrong')
                sys.exit()
            s = time.time()
            a = L.data(day)
            data_t += time.time() - s

            s = time.time()
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

        # print(f'data: {data_t}')
        # print(f'save: {save_t}')
        # print(f'bench: {bench_t}')
        # print(f'people: {people_t}')

    # print(time.time() - t)
    save_array = np.array(save_array, dtype=np.int16)

    with h5py.File(out_fn, 'w') as f:
        g = f.create_group('g')
        g.create_dataset('default', data=save_array)
        g.create_dataset('events', data=L.events)
        g.create_dataset('movements', data=L.movements)

        for arg_name, arg in metadata.items():
            g.attrs[arg_name] = arg


def condition(params):
    cond_i, pp, n, b = params
    for i in range(1000):
        run(cond_i, i, pp, n, b)


pp_opts = .0001, .00001
n_opts = 5, 15
b_opts = .01, .05

default = [.00005, 10, .025]

opts = []
i = 0
for pp in pp_opts:
    print(i)
    de = [i] + default[:]
    de[1] = pp
    opts.append(de)
    i += 1
for n in n_opts:
    de = [i] + default[:]
    de[2] = n
    opts.append(de)
    i += 1
for b in b_opts:
    de = [i] + default[:]
    de[3] = b
    opts.append(de)
    i += 1
opts += [[i] + default]

with multiprocessing.Pool(7) as p:
    p.map(condition, opts)
