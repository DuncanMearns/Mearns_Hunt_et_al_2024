from experiment import *

import numpy as np
from sklearn.cluster import AgglomerativeClustering


if __name__ == "__main__":

    for species in expt.species_names:

        print(species)

        # Params
        params = get_params(species)
        n_align_clusters = params["n_align_clusters"]

        # Paths
        flipped_bouts_path = expt.directory["analysis", "bouts", species].new_file("flipped_bouts", "npy")
        bouts_path = expt.directory["analysis", "bouts", species, "aligned_bouts.npy"]

        # Open aligned bouts
        bouts = np.load(bouts_path)
        n_bouts = len(bouts)

        # Cluster bouts with flipped bouts
        doubled_bouts = np.concatenate([bouts, -bouts], axis=0)
        clusterer = AgglomerativeClustering(n_clusters=n_align_clusters, metric="euclidean", linkage="ward")
        reshaped = doubled_bouts.reshape(len(doubled_bouts), -1)
        labels = clusterer.fit_predict(reshaped)

        # Compute sign of each cluster
        cluster_means = np.array([doubled_bouts[labels == l].mean(axis=0) for l in np.arange(n_align_clusters)])
        cluster_signs = np.zeros(n_align_clusters)
        for l in range(n_align_clusters):
            cluster_mean = cluster_means[l]
            peak = np.argmax(np.abs(cluster_mean[:, 0]))
            cluster_signs[l] = np.sign(cluster_mean[peak, 0])

        # Flip bouts
        direction = cluster_signs[labels[:n_bouts]]
        flipped = bouts * direction[:, np.newaxis, np.newaxis]
        np.save(flipped_bouts_path, flipped)
