from behavior_analysis.experiment import PreyCaptureExperiment

import numpy as np
import h5py
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # Experiment
    directory = r"D:\percomorph_analysis"
    video_directory = None
    expt = PreyCaptureExperiment.open(directory, video_directory)
    species_ids = expt.video_data["species_id"].unique()

    # Canonical PCs
    pca_path = expt.directory["postural_decomposition", "canonical_pca.h5"]
    with h5py.File(pca_path, "r") as f:
        canonical_pcs = f["eigenfish"][:]

    # Species PCs
    species_pcs = {}
    for species in species_ids:
        pca_path = expt.directory["postural_decomposition", f"{species}.h5"]
        with h5py.File(pca_path, "r") as f:
            eigenfish = f["eigenfish"][:]
        species_pcs[species] = eigenfish

    # Pairwise
    fig, axes = plt.subplots(1, len(species_ids), figsize=(7, 2), sharey="row", dpi=150)
    i = 0
    for species, pcs in species_pcs.items():
        dot = np.dot(canonical_pcs, pcs.T)
        ax = axes[i]
        ax.matshow(np.abs(dot), vmin=0, vmax=1, cmap="inferno")
        ax.set_title(species)
        i += 1
    # plt.show()
    plt.savefig(expt.directory["figures", "figure3"].new_file(f"compare_pcs", "jpg"))
