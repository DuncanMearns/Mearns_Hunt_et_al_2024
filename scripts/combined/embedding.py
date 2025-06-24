from behavior_analysis.experiment import PreyCaptureExperiment

import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind, wasserstein_distance



if __name__ == "__main__":
    # Experiment
    directory = r"D:\percomorph_analysis"
    # video_directory = r"E:\CICHLID_PAPER\DATA"
    expt = PreyCaptureExperiment.open(directory, None)

    # # PARAMS
    # n_neighbors = 50
    #
    # neighbors_path = expt.directory["combined_analysis"].new_file("nearest_neighbors", "npy")
    # counts_path = expt.directory["combined_analysis"].new_file("species_counts", "npy")
    # distances_path = expt.directory["combined_analysis"].new_file("pdist", "npy")
    # stats_directory = expt.directory["combined_analysis"].new_subdir("stats")
    #
    # # counts = []
    # # for species, path in expt.directory["combined_analysis", "bouts"].items():
    # #     species_bouts = np.load(path)
    # #     counts.append(len(species_bouts))
    # # c = np.zeros(sum([counts]), dtype="uint8")
    # # counts = np.cumsum(counts)
    # # for n in counts:
    # #     c[n:] += 1
    #
    # flipped_bouts = np.load(expt.directory["combined_analysis", "flipped_bouts.npy"])
    # flipped_bouts = flipped_bouts.reshape(len(flipped_bouts), -1)

    # counts = np.load(counts_path)
    # counts = np.hstack([[0], counts])
    # counts = np.cumsum(counts)
    #
    # subset_bouts = []
    # subset_counts = [0]
    # for n, m in zip(counts, counts[1:]):
    #     species_bouts = flipped_bouts[n:m][::5]
    #     subset_bouts.append(species_bouts)
    #     subset_counts.append(len(species_bouts))
    # flipped_bouts = np.concatenate(subset_bouts, axis=0)
    # subset_counts = np.cumsum(subset_counts)
    #
    # nn = NearestNeighbors(n_neighbors=100)
    # nn.fit(flipped_bouts)
    # graph = nn.kneighbors_graph()
    # graph = graph.toarray().astype("int32")
    # np.save(neighbors_path, graph)
    #
    # graph = np.load(neighbors_path)
    # for n, m in zip(counts, counts[1:]):
    #     species_graph = graph[n:m]
    #     intraspecies = species_graph[:, n:m]
    #     print(species_graph.sum(axis=1).mean())
    #
    # D = pdist(flipped_bouts)
    # np.save(distances_path, D)
    #
    # D = np.load(distances_path)
    # D = squareform(D)
    #
    # fig, axes = plt.subplots(1, 4, sharey=True)
    # S = np.zeros((4, 4))
    # T = np.zeros((4, 4))
    # W = np.zeros((4, 4))
    # P = np.zeros((4, 4))
    # for i in range(4):
    #     n = subset_counts[i:i+2]
    #     dset = []
    #     for j in range(4):
    #         m = subset_counts[j:j+2]
    #         d = D[n[0]:n[1], m[0]:m[1]]
    #         nearest_idxs = np.argsort(d, axis=1)
    #         if i == j:
    #             col_idxs = nearest_idxs[:, 1:n_neighbors + 1]
    #         else:
    #             col_idxs = nearest_idxs[:, :n_neighbors]
    #         row_idxs = np.repeat(np.arange(len(d))[:, np.newaxis], n_neighbors, axis=1)
    #         d = d[row_idxs, col_idxs]
    #         median = np.median(d, axis=1)
    #         dset.append(median)
    #         S[i, j] = np.median(median)
    #         # scatter = (np.random.rand(len(median)) - 0.5) * 0.5
    #         # x = (np.ones(len(median)) * j) + scatter
    #         # axes[i].scatter(x, median, lw=0, s=1)
    #     axes[i].violinplot(dset, showmedians=True)
    #
    #     null = dset[i]
    #     for j, dist in enumerate(dset):
    #         t, p = ttest_ind(null, dist)
    #         T[i, j] = t
    #         P[i, j] = p
    #         w = wasserstein_distance(null, dist)
    #         W[i, j] = w
    #     # print()
    # # plt.show()
    #
    # fig, axes = plt.subplots(1, 3)
    # axes[0].matshow(T, cmap="bwr", vmin=-15, vmax=15)
    # axes[1].matshow(W, cmap="inferno")
    # axes[2].matshow(P, cmap="inferno_r")
    # plt.show()

    # tsne = TSNE().fit_transform(flipped_bouts)

    output_path = expt.directory["combined_analysis"].new_file("tsne", "npy")
    # np.save(output_path, tsne)

    tsne = np.load(output_path)


    plt.scatter(*tsne.T, lw=0, s=10, c="k", alpha=0.5)
    plt.show()
