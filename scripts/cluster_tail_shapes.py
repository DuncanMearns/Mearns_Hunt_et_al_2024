import h5py
import pandas as pd

from experiment import *

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


def trajectory_to_tail(x, components, mean, std):
    y = np.dot(x, components)
    angles = (y * std) + mean
    # convert into dx, dy vectors
    ps = np.concatenate([np.cos(angles)[..., np.newaxis], np.sin(angles)[..., np.newaxis]], axis=-1)
    ps = np.cumsum(ps, axis=1)  # sum vectors along tail
    ps = np.pad(ps, ((0, 0), (1, 0), (0, 0)), constant_values=0)
    return ps


if __name__ == "__main__":

    np.random.seed(202306)

    # for species in ("a_burtoni", "l_attenuatus", "l_ocellatus", "o_latipes"):
    for species in ("a_burtoni",):
        # Paths
        pca_path = expt.directory["postural_decomposition", f"{species}.h5"]
        bouts_path = expt.directory["bouts", species, "bouts_clustered.csv"]
        bout_trajectories_path = expt.directory["bouts", species, "flipped_bouts.npy"]
        # Load data
        with h5py.File(pca_path, "r") as f:
            mean = f["mean"][:]
            std = f["std"][:]
            components = f["eigenfish"][:3]
        bouts = np.load(bout_trajectories_path)
        bouts_info = pd.read_csv(bouts_path)
        # Make plots
        t = np.arange(bouts.shape[1])
        norm = mpl.colors.Normalize(t[10], t[-10])
        time_cm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
        time_colors = time_cm.to_rgba(t, alpha=0.5)

        for l, cluster_bouts in bouts_info.groupby("cluster"):
            idxs = np.random.choice(cluster_bouts.index, 25, False)
            idxs = np.sort(idxs)
            fig, axes = plt.subplots(5, 5)
            fig.suptitle(f"{l}")
            for i, ax in enumerate(axes.ravel()):
                bout_idx = idxs[i]
                points = trajectory_to_tail(bouts[bout_idx], components, mean, std)
                for j, ps in enumerate(points):
                    l, = ax.plot(*ps.T, lw=2, c=time_colors[j])
                    l.set_clip_on(False)
                ax.set_xlim(-1, 8)
                ax.set_ylim(-4, 4)
                ax.set_aspect("equal")
                ax.axis("off")
            plt.show()
