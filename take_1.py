import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation


class Rectangle:
    def __init__(self, position, size, v):
        self.pos = np.array(position)  # None if not in lab
        self.size = np.array(size)
        self.v = v

    def collision(self, rect):
        (y, x), (h, w) = self.pos, self.size
        (ry, rx), (rh, rw) = rect.pos, self.size
        return x < rx + rw and x + w > rx and y < ry + rh and y + h > ry

    def draw(self, arr):
        (y, x), (h, w) = self.pos, self.size
        v = self.v
        for i in range(y, y+h):
            for j in range(x, x+w):
                arr[i, j] = v

    @classmethod
    def from_pos(cls, pos1, pos2, v):
        (y1, x1), (y2, x2) = pos1, pos2
        h = y2 - y1
        w = x2 - x1
        return cls(pos1, (h, w), v)


class Person(Rectangle):
    def __init__(self, v):
        self.pos = None
        # self.current_node = None
        self.v = v
        self.size = np.array((50, 50))

    def enter_lab(self, y, x):
        self.pos = np.array((y, x))

    def move(self, dpos):
        if isinstance(dpos, np.ndarray):
            self.pos += dpos
        else:
            self.pos += np.array(dpos)

    '''
    def change_node(self, new_node):
        if self.current_node:
            self.current_node.set_value(0)
        self.current_node = new_node
        self.current_node.set_value(1)
    '''

    # infected or not
    # speed
    # destination or moving/not moving status
    # current path


class Node:
    def __init__(self, y, x):
        self.y = y
        self.x = x
        self.v = 0
        self.nodes = []

    def dist(self, n):
        return abs(n.x - self.x) + abs(n.y - self.y)  # Manhattan Distance (L1)

    def add_node(self, n):
        self.nodes.append(n)

    def set_value(self, v):
        self.v = v


class Lab:
    def __init__(self, h, w, walls, benches, people):
        self.h = h
        self.w = w

        self.im = np.zeros((h, w))
        for w in walls:
            y, x = w.pos
            dy, dx = w.size
            for r in range(y, y+dy):
                for c in range(x, x+dx):
                    self.im[r, c] = 3
        for b in benches:
            y, x = b.pos
            dy, dx = b.size
            for r in range(y, y+dy):
                for c in range(x, x+dx):
                    self.im[r, c] = 2
        self.benches = benches
        self.people = people

    def plot(self, ims):
        im = self.im.copy()
        for p in self.people:
            if p.pos is not None:
                p.draw(im)

        ims.append(im)


class Bench(Rectangle):
    pass

class Door:
    pass


def wall(y1, x1, y2, x2):
    return Rectangle.from_pos((y1, x1), (y2, x2), 1)


def make_walls():
    walls = []
    walls.append(wall(0, 25, 25, 1375))  # north wall
    walls.append(wall(25, 0, 1275, 25))  # west wall
    walls.append(wall(1275, 25, 1300, 875))

    # eastern walls
    walls.append(wall(25, 1375, 325, 1400))
    walls.append(wall(325, 1175, 350, 1375))
    walls.append(wall(350, 1175, 650, 1200))
    walls.append(wall(650, 875, 675, 1175))
    walls.append(wall(675, 875, 1275, 900))

    # internal walls
    walls.append(wall(225, 225, 250, 525))
    walls.append(wall(450, 225, 475, 550))
    walls.append(wall(25, 525, 450, 550))
    walls.append(wall(25, 750, 450, 775))

    return walls


if __name__ == '__main__':
    lab_h = 1300
    lab_w = 1400
    walls = make_walls()

    bench = Rectangle((200, 200), (50, 100), 3)

    p = Person(4)
    p.enter_lab(150, 100)

    L = Lab(lab_h, lab_w, walls, [bench], [p])

    dpos = np.array((1, 0))

    ims = []
    for i in range(30):
        L.plot(ims)
        p.move(dpos)

    frames = []
    fig = plt.figure()
    frames = [[plt.imshow(i, cmap='BuGn', animated=True)] for i in ims]

    ani = ArtistAnimation(fig, frames, interval=50, blit=True,
                          repeat_delay=0)

    plt.show()
