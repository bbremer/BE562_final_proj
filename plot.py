import sys

import h5py
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import numpy as np

import lab
from lab import Lab, Node, Bench

lab_fn, data_fn = sys.argv[1:]

arg_names = ['pp', 'sp', 'ps', 'd', 'b', 'n', 'br', 'pr', 'lab_fn']

L = lab.load_lab(lab_fn)

with h5py.File(data_fn, 'r') as f:
    dataset = f['data']

data = np.array(dataset)
attrs = {name: dataset.attrs[name] for name in arg_names}

n = attrs['n']
L.people = [lab.Person() for _ in range(n)]

ims = []
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
