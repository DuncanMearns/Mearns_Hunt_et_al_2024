import h5py
import numpy as np


class HDF5Attr:

    def __init__(self, key=None):
        self.name = key

    def __set_name__(self, owner, name):
        if not self.name:
            self.name = name
        self.private_name = "_" + self.name

    def __get__(self, instance, owner):
        try:
            return getattr(instance, self.private_name)
        except AttributeError:
            with h5py.File(instance.path, "r") as f:
                val = f[self.name][:]
            setattr(instance, self.private_name, val)
            return val


class TrackingInterface:

    node_names = HDF5Attr()
    data = HDF5Attr("tracks")
    point_scores = HDF5Attr()
    tail_point_names = (b'tail_base', b'tail_1', b'tail_2', b'tail_middle', b'tail_3', b'tail_4', b'tail_5', b'tail_tip')
    heading_node_names = (b'swimbladder', b'nasal')
    centroid_idx = 2

    def __init__(self, path):
        self.path = path

    @property
    def tail_point_idxs(self):
        return [list(self.node_names).index(name) for name in self.tail_point_names]

    @property
    def heading_node_idxs(self):
        return [list(self.node_names).index(name) for name in self.heading_node_names]

    def read(self):
        data = self.data
        data = data.squeeze()
        data = np.swapaxes(data, 0, 2)
        return data

    def tail_points(self):
        data = self.read()
        return data[:, self.tail_point_idxs, :]

    def heading_vectors(self):
        data = self.read()
        heading_vectors = np.diff(data[:, self.heading_node_idxs, :], axis=1).squeeze()
        heading_vectors = heading_vectors / np.linalg.norm(heading_vectors, axis=-1, keepdims=True)
        return heading_vectors

    def headings(self):
        vs = self.heading_vectors()
        return np.arctan2(vs[:, 1], vs[:, 0])

    def tip_vectors(self):
        points = self.tail_points()
        tip_v = points[:, -1] - points[:, 0]
        tip_v = tip_v / np.linalg.norm(tip_v, axis=-1, keepdims=True)
        return tip_v

    def tail_angles(self):
        heading_vectors = self.heading_vectors()
        tip_vectors = self.tip_vectors()
        return np.arcsin(np.cross(-heading_vectors, tip_vectors))

    def velocity(self):
        data = self.read()
        centroids = data[:, self.centroid_idx]
        return np.linalg.norm(np.diff(centroids, axis=0), axis=1)

    def is_tracked(self):
        scores = self.point_scores.squeeze()
        scores = np.swapaxes(scores, 0, 1)
        tracked, = np.where(np.all(scores > 0, axis=-1))
        return tracked

    @property
    def shape(self):
        return self.read().shape
