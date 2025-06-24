from experiment import *
import numpy as np
from sklearn.cluster import AgglomerativeClustering


if __name__ == "__main__":

    # Params
    n_align_clusters = 60

    # Path
    flipped_bouts_path = expt.directory["analysis", "combined_analysis"].new_file("flipped_bouts", "npy")

    bouts = []
    counts = []
    for species, path in expt.directory["analysis", "combined_analysis", "bouts"].items():
        species_bouts = np.load(path)
        bouts.append(species_bouts)
        counts.append(len(species_bouts))
    counts = np.array(counts)
    bouts = np.concatenate(bouts, axis=0)
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
    labels_path = expt.directory["analysis", "combined_analysis"].new_file("initial_labels", "npy")
    np.save(labels_path, labels)
    direction_path = expt.directory["analysis", "combined_analysis"].new_file("bout_directions", "npy")
    np.save(direction_path, direction[:n_bouts])
    counts_path = expt.directory["analysis", "combined_analysis"].new_file("species_counts", "npy")
    np.save(counts_path, counts)
