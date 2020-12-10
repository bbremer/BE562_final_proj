import sys

import h5py
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import numpy as np

import lab
from lab import Lab, Node, Bench

lab_fn, data_fn = sys.argv[1:]

arg_names = ['pp', 'sp', 'ps', 'd', 'b', 'n', 'br', 'pr', 'lab_fn']


with h5py.File(data_fn, 'r') as f:
    g = f['g']
    dataset = g['default']
    data = dataset[:]

    print(g['events'][:])

    movements = g['movements'][:]
    for m in movements:
        q = m.decode().split(',')
        if q[0] == '7' and q[2] == '1':
            print(m)

    attrs = {name: g.attrs[name] for name in arg_names}

L = lab.load_lab(attrs['lab_fn'])

n = attrs['n']
L.people = [lab.Person() for _ in range(n)]


# y1, x1 = data[0, 1:3]

g = iter(data)
prev = None
for i, d in enumerate(g):
    # if d[1] != y1 or d[2] != x1:
    if any(v == 1 for v in d[:3*n:3]):
        print(i)
        # thing = [prev, d]
        thing = [d]
        break
    prev = d
else:
    print('exited')

thing += [next(g) for _ in range(20)]
data = np.array(thing)

# print(data[:10000:1000])
ims = []
# for row in data[:10000:1000]:
for row in data:
    L.plot(ims, arr_row=row)

# plot each timepoint
frames = []
fig = plt.figure()
frames = [[plt.imshow(i, cmap='BuGn', animated=True)]
          for i in ims]
ani = ArtistAnimation(fig, frames, interval=100, blit=True,
                      repeat_delay=0)

plt.show()
