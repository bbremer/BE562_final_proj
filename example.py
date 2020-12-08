import random

from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt

import lab
from lab import Lab, Node, Bench

L = lab.load_lab('Lab_v1.p')

# make a new person and add to lab
p = lab.Person()
L.people = [p]

# have person start at random bench and move to another bench
b1, b2 = random.sample(L.benches, 2)
p.pos = b1.node.pos
p.path = b1.p_dict[b2]

# 60 timepoint simulation
ims = []
for i in range(60):
    L.plot(ims)
    for b in L.benches:
        b.update()
    for p in L.people:
        p.update()


# plot each timepoint
frames = []
fig = plt.figure()
frames = [[plt.imshow(i, cmap='BuGn', animated=True)]
          for i in ims]
ani = ArtistAnimation(fig, frames, interval=100, blit=True,
                      repeat_delay=0)

plt.show()
