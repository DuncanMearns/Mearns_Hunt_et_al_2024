import numpy as np
from sklearn.manifold import TSNE, Isomap, MDS

from experiment import *

from matplotlib import pyplot as plt

from ethomap import pdist_dtw, IsomapPrecomputed, affinity_propagation
from scipy.spatial.distance import squareform
from scipy.interpolate import CubicSpline
from zigzeg import find_runs
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix

def interpolate(x, gap_size=5):
    # get transformed
    x = np.array(x)
    # find gaps to interpolate
    x_nan = np.any(np.isnan(x), axis=1)
    gaps, = np.where(x_nan)
    gaps = find_runs(gaps)
    gaps = [idxs for idxs in gaps if len(idxs) <= gap_size]
    if len(gaps):
        # get indices to interpolate
        interp_t = [idxs for idxs in gaps if
                    (~np.any(x_nan[idxs[0] - 3:idxs[0]]) & ~np.any(x_nan[idxs[-1] + 1:idxs[-1] + 4]))]
        if not len(interp_t):
            return x
        interp_t = np.hstack(interp_t)
        # interpolate
        t, = np.where(~x_nan)
        cs = CubicSpline(t, x[t])
        interp_x = cs(interp_t)
        x[interp_t] = interp_x
    return x


if __name__ == "__main__":

    np.random.seed(2023)

    strikes = []
    sp = []
    for i, species_id in enumerate(expt.species_names):
        print(species_id)
        species_strikes = np.load(expt.directory["analysis", "combined_analysis", "strikes", species_id + ".npy"])
        strikes.append(species_strikes)
        sp.append(np.ones(len(species_strikes)) * i)
    strikes = np.concatenate(strikes, axis=0)
    sp = np.concatenate(sp)

    is_null = np.all(strikes == 0, axis=-1)
    strikes[np.where(is_null)] = np.nan
    strikes = np.array([interpolate(x, 5) for x in strikes])
    strikes = strikes[:, 50:200]

    is_null = np.all(np.isnan(strikes), axis=-1)
    is_null = np.sum(is_null, axis=1)
    keep, = np.where(is_null == 0)

    strikes = strikes[keep]
    sp = sp[keep].astype("i4")
    print(len(strikes))

    n_clusters = 5

    strikes_ = strikes.reshape((len(strikes), -1))
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(strikes_)

    fig, axes = plt.subplots(n_clusters, 3, sharex=True)
    for l, strike in zip(labels, strikes):
        for j in range(3):
            axes[l, j].plot(strike[:, j], lw=0.5, c=f"C{l}", alpha=0.2)
    # plt.show()

    M = confusion_matrix(sp, labels)
    M = M / np.sum(M, keepdims=True, axis=-1)
    print(M)
    plt.matshow(M)
    plt.show()

    strikes_by_cluster = np.array([strikes[labels == l].mean(axis=0) for l in range(n_clusters)])

    # fig, axes = plt.subplots(n_clusters, 3, sharex=True)
    for i, ax in enumerate(axes):
        for j in range(3):
            axes[i, j].plot(strikes_by_cluster[i, :, j], c="k")
    plt.show()
