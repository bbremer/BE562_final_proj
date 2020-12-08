import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation
import networkx as nx

G = nx.Graph()


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

    def get_pos(self):
        return self.pos, self.pos + self.size

    @classmethod
    def from_pos(cls, pos1, pos2, v):
        (y1, x1), (y2, x2) = pos1, pos2
        h = y2 - y1
        w = x2 - x1
        return cls(pos1, (h, w), v)


class Person(Rectangle):
    def __init__(self):
        self.pos = None
        self.v = 4
        self.size = np.array((50, 50))
        self.path = []

    def enter_lab(self, y, x):
        self.pos = np.array((y, x))

    def move(self, dpos):
        if isinstance(dpos, np.ndarray):
            self.pos += dpos
        else:
            self.pos += np.array(dpos)

    def update(self):
        if self.path:
            curr_node = self.path.pop(0)
            self.pos = curr_node.pos

    def draw(self, arr):
        pos = self.pos - self.size // 2
        (y, x), (h, w) = pos, self.size
        v = self.v
        for i in range(y, y+h):
            for j in range(x, x+w):
                arr[i, j] = v


class Node(Rectangle):
    def __init__(self, y, x):
        super().__init__((y, x), (25, 25), 5)
        self.y = y
        self.x = x
        self.edges = []
        self.visited = False
        self.weight = 1000000000
        self.prev = None
        self.bench = None

    def dist(self, n):
        return abs(n.x - self.x) + abs(n.y - self.y)  # Manhattan Distance (L1)

    def __str__(self):
        return str(self.pos)

    def draw(self, arr):
        pos = self.pos - self.size // 2
        (y, x), (h, w) = pos, self.size
        v = self.v
        for i in range(y, y+h):
            for j in range(x, x+w):
                arr[i, j] = v


def good_candidate(n, n1, n2):
    if n.bench is None:
        return True
    if n.bench is n1.bench:
        return True
    if n.bench is n2.bench:
        return True
    return False


def dykstra(n1, n2, nodes):
    n1.visited = True
    n1.weight = 0

    candidates = nodes
    curr = n1

    while curr is not n2:
        for neighbor in random.sample(curr.edges, len(curr.edges)):
            if neighbor.visited or not good_candidate(neighbor, n1, n2):
                continue
            d = curr.weight + 1
            if d < neighbor.weight:
                neighbor.weight = d
                neighbor.prev = curr

        candidates = [n for n in candidates if not n.visited]
        # print([c.weight for c in candidates])
        curr = min(candidates, key=lambda n: n.weight)
        curr.visited = True

    # backtrace
    path = [curr]
    while curr.prev is not None:
        curr = curr.prev
        path.insert(0, curr)

    # reset nodes
    for n in nodes:
        n.visited = False
        n.weight = 1000000000
        n.prev = None

    return path


def point_in_rect(y1, x1, y2, x2, y, x):
    assert x2 > x1 and y2 > y1
    return x1 < x < x2 and y1 < y < y2


def valid_edge(n1, n2, walls, benches):
    if n1.dist(n2) > 50 or n1 is n2:
        return False

    # get edge discretization
    p1, p2 = n1.pos, n2.pos
    disc = [p1*t + p2*(1-t) for t in np.linspace(0, 1, 6)]

    for w in walls:
        (y1, x1), (y2, x2) = w.get_pos()
        for y, x in disc:
            if point_in_rect(y1, x1, y2, x2, y, x):
                return False

    return True


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
        self.nodes = [Node(y*50, x*50) for y in range(1, (self.h-1)//50)
                      for x in range(1, (self.w-1)//50)]
        for i, n in enumerate(self.nodes):
            n.ind = i
            G.add_node(i, pos=n.pos[::-1])

        for b in benches:
            b.find_nodes(self.nodes)
            y, x = b.pos
            dy, dx = b.size
            for r in range(y, y+dy):
                for c in range(x, x+dx):
                    self.im[r, c] = 2
        self.make_edges(walls, benches)

        self.benches = benches
        self.make_paths()
        self.people = people

    def make_edges(self, walls, benches):
        for n1 in self.nodes:
            for n2 in self.nodes:
                # if n1.dist(n2) < 60 and n1 is not n2:
                if valid_edge(n1, n2, walls, benches):
                    n1.edges.append(n2)
                    G.add_edge(n1.ind, n2.ind)

    def make_paths(self):
        for b1 in self.benches:
            b1.p_dict = {}
            n1 = b1.node
            for b2 in self.benches:
                if b1 is b2:
                    continue
                n2 = b2.node
                b1.p_dict[b2] = dykstra(n1, n2, self.nodes)

    def plot(self, ims):
        im = self.im.copy()
        for p in self.people:
            if p.pos is not None:
                p.draw(im)
        for n in self.nodes:
            n.draw(im)

        ims.append(im)


class Bench(Rectangle):
    def __init__(self, pos1, pos2):
        (y1, x1), (y2, x2) = pos1, pos2
        h = y2 - y1
        w = x2 - x1
        super().__init__(pos1, (h, w), 2)
        if pos1[0] < pos2[0]:
            self.y1, self.x1 = pos1
            self.y2, self.x2 = pos2
        else:
            self.y1, self.x1 = pos2
            self.y2, self.x2 = pos1

    def find_nodes(self, nodes):
        # self.node = [n for n in nodes if self.x1 < n.x < self.x2 and
        #              self.y1 < n.y < self.y2][0]
        nodes = [n for n in nodes if point_in_rect(self.y1, self.x1, self.y2,
                                                   self.x2, n.y, n.x)]
        for n in nodes:
            n.bench = self
            n.v = 6
        self.node = random.choice(nodes)
        # self.node = nodes[1]
        self.node.v = 7


def bench(y1, x1, y2, x2):
    return Bench((y1, x1), (y2, x2))


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
    walls.append(wall(25, 525, 475, 550))
    walls.append(wall(25, 750, 475, 775))

    return walls


def make_benches():
    benches = []

    # upper left benches
    benches.append(bench(25, 425, 225, 525))
    benches.append(bench(250, 425, 450, 525))
    benches.append(bench(25, 550, 125, 750))

    # center room benches
    benches.append(bench(475, 275, 575, 475))
    benches.append(bench(675, 25, 875, 125))
    benches.append(bench(975, 25, 1175, 125))
    benches.append(bench(775, 325, 875, 525))
    benches.append(bench(925, 325, 1025, 525))
    benches.append(bench(1175, 325, 1275, 525))
    benches.append(bench(675, 775, 875, 875))
    benches.append(bench(1000, 775, 1200, 875))

    # upper right benches
    benches.append(bench(25, 1175, 125, 1375))
    benches.append(bench(250, 775, 450, 875))
    benches.append(bench(350, 1075, 550, 1175))

    return benches


if __name__ == '__main__':
    lab_h = 1300
    lab_w = 1400
    walls = make_walls()
    benches = make_benches()

    p = Person(4)

    L = Lab(lab_h, lab_w, walls, benches, [p])

    b1, b2 = random.sample(benches, 2)
    p.pos = b1.node.pos
    p.path = b1.p_dict[b2]

    ims = []
    for i in range(60):
        L.plot(ims)
        # p.move(dpos)
        p.update()

    L.plot(ims)
    L.plot(ims)

    frames = []
    '''
    fig = plt.figure()
    pos = nx.get_node_attributes(G, 'pos')
    frames = [[plt.imshow(i, cmap='BuGn', animated=True), nx.draw(G, pos)]
              for i in ims]
    # frames = [[plt.imshow(i, cmap='BuGn', animated=True)]
    #           for i in ims]

    ani = ArtistAnimation(fig, frames, interval=100, blit=True,
                          repeat_delay=0)
    '''

    fig = plt.figure()
    pos = nx.get_node_attributes(G, 'pos')
    frames = [[plt.imshow(i, cmap='BuGn', animated=True)]
              for i in ims]
    ani = ArtistAnimation(fig, frames, interval=100, blit=True,
                          repeat_delay=0)

    plt.show()
