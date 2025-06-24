import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

from experiment import *


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


np.random.seed(2023)


if __name__ == "__main__":

    strikes = []
    sp = []
    dfs = []
    for i, species_id in enumerate(expt.species_names):
        # if species_id == "l_ocellatus":
        #     continue
        print(species_id)
        species_strikes = np.load(expt.directory["analysis", "combined_analysis", "strikes", species_id + ".npy"])
        df = pd.read_csv(expt.directory["analysis", "combined_analysis", "strikes", species_id + "_annotated.csv"], index_col=0)
        species_strikes = species_strikes[df.index]
        strikes.append(species_strikes)
        sp.append(np.ones(len(species_strikes)) * i)
        dfs.append(df)
        print(len(df))
    strikes = np.concatenate(strikes, axis=0)
    sp = np.concatenate(sp).astype("i4")
    df = pd.concat(dfs, ignore_index=True)

    # n_clusters = 8
    #
    # strikes_ = strikes.reshape((len(strikes), -1))
    # labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(strikes_)
    #
    # cluster_counts = np.array([(labels == l).sum() for l in np.arange(n_clusters)])
    # keep_clusters, = np.where(cluster_counts > 20)
    # to_keep = np.where(np.isin(labels, keep_clusters))
    # strikes = strikes[to_keep]
    # sp = sp[to_keep]
    # df = df.loc[to_keep]

    # RE-CLUSTER
    n_clusters = 5

    # labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(strikes_[to_keep])
    strikes_ = strikes.reshape((len(strikes), -1))
    agg = AgglomerativeClustering(n_clusters=5, linkage="ward")  #, distance_threshold=0)
    labels = agg.fit_predict(strikes_)
    # plot_dendrogram(agg, labels=sp)
    # plt.show()

    # M = confusion_matrix(sp, labels)[:5]
    # M_norm = M / np.sum(M, keepdims=True, axis=-1)

    df["strike_cluster"] = labels
    species_ids = list(expt.species_names.keys())
    df["species"] = [species_ids[i] for i in sp]
    df.to_csv(expt.directory["analysis", "combined_analysis", "strikes"].new_file("clustered", "csv"), index=False)
    np.save(expt.directory["analysis", "combined_analysis", "strikes"].new_file("combined", "npy"), strikes)
