from experiment import *

import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AffinityPropagation


if __name__ == "__main__":

    for species in expt.species_names:

        print(species)

        # Paths
        bouts_path = expt.directory["analysis", "bouts", species, "bouts_cleaned.csv"]
        flipped_bouts_path = expt.directory["analysis", "bouts", species, "flipped_bouts.npy"]
        clustered_path = expt.directory["analysis", "bouts", species].new_file("bouts_clustered", "csv")
        cluster_params_path = expt.directory["analysis", "bouts", species].new_file("cluster_params", "json")
        cluster_stats_path = expt.directory["analysis", "bouts", species].new_file("cluster_stats", "npy")
        exemplar_path = expt.directory["analysis", "bouts", species].new_file("exemplar_indices", "npy")

        # Open bouts
        bouts_df = pd.read_csv(bouts_path)
        bouts = np.load(flipped_bouts_path)
        assert len(bouts) == len(bouts_df)
        n_bouts = len(bouts)
        print(n_bouts, "bouts")

        # Compute pairwise distances and affinity matrix
        reshaped = bouts.reshape(n_bouts, -1)
        D = pdist(reshaped)
        A = -(D ** 2)

        # Cluster with affinity propagation
        min_A = A.min()
        A = squareform(A)
        clusterer = AffinityPropagation(affinity="precomputed", preference=min_A)
        labels = clusterer.fit_predict(A)
        exemplars = clusterer.cluster_centers_indices_
        n_clusters = len(np.unique(labels))
        print(n_clusters, "clusters")

        # Compute means
        cluster_means = np.array([bouts[labels == l].mean(axis=0) for l in np.arange(n_clusters)])
        cluster_std = np.array([bouts[labels == l].std(axis=0) for l in np.arange(n_clusters)])
        cluster_stats = np.concatenate([cluster_means[np.newaxis], cluster_std[np.newaxis]], axis=0)

        # Save
        bouts_df["cluster"] = labels
        bouts_df.to_csv(clustered_path, index=False)
        # Save params
        cluster_params = dict(n_bouts=n_bouts, affinity=min_A, n_clusters=n_clusters)
        with open(cluster_params_path, "w") as f:
            json.dump(cluster_params, f)
        # Save stats
        np.save(cluster_stats_path, cluster_stats)
        np.save(exemplar_path, exemplars)
