from experiment import *

import numpy as np
import h5py
from sklearn.decomposition import PCA


if __name__ == "__main__":

    # Paths
    pca_directory = expt.directory.new_subdir("postural_decomposition")
    pca_path = pca_directory.joinpath(f"{species}.h5")

    # Open tail curvature data
    data = []
    for trial in trials:
        tracked = trial.tracking.is_tracked()
        ks = trial.tail_curvature[tracked]
        ks = ks[np.abs(ks[:, 0]) < 1.]  # remove frames with kinked tail
        data.append(ks)
    data = np.concatenate(data, axis=0)

    # Whiten
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    whitened = (data - data_mean) / data_std

    # PCA
    pca = PCA()
    transformed = pca.fit_transform(whitened)

    # Save
    with h5py.File(pca_path, "w") as f:
        f.create_dataset("trial", data=data)
        f.create_dataset("whitened", data=whitened)
        f.create_dataset("transformed", data=transformed)
        f.create_dataset("mean", data=data_mean)
        f.create_dataset("std", data=data_std)
        f.create_dataset("explained_variance", data=pca.explained_variance_ratio_)
        f.create_dataset("eigenfish", data=pca.components_)
