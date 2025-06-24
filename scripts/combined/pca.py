from behavior_analysis.experiment import PreyCaptureExperiment, TrialData

import numpy as np
import h5py
from sklearn.decomposition import PCA
from collections import defaultdict


if __name__ == "__main__":

    # Experiment
    directory = r"D:\percomorph_analysis"
    video_directory = None
    expt = PreyCaptureExperiment.open(directory, video_directory)
    TrialData.directory = expt.directory["tracking"]

    # Load data
    trials = defaultdict(list)
    for idx, trial_info in expt.video_data.iterrows():
        try:
            trial_data = TrialData(**trial_info.to_dict())
            assert trial_data.tracking_path.exists()
            trials[trial_info.species_id].append(trial_data)
        except ValueError:
            print(f"Missing data: {trial_info.video_id}")
            pass

    # Paths
    pca_directory = expt.directory.new_subdir("postural_decomposition")
    pca_path = pca_directory.joinpath("canonical_pca.h5")

    # Open tail curvature data
    species_data = {}
    for species, species_trials in trials.items():
        data = []
        for trial in species_trials:
            tracked = trial.tracking.is_tracked()
            ks = trial.tail_curvature[tracked]
            ks = ks[np.abs(ks[:, 0]) < 1.]  # remove frames with kinked tail
            data.append(ks)
        species_data[species] = np.concatenate(data, axis=0)
    data = np.concatenate(list(species_data.values()), axis=0)
    print("Data shape:", data.shape)

    # Whiten
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    whitened = (data - data_mean) / data_std

    # PCA
    pca = PCA()
    transformed = pca.fit_transform(whitened)

    # Split by species
    split_idxs = np.cumsum([len(data) for data in species_data.values()])
    whitened = np.split(whitened, split_idxs)
    transformed = np.split(transformed, split_idxs)

    # Save
    with h5py.File(pca_path, "w") as f:
        f.create_dataset("mean", data=data_mean)
        f.create_dataset("std", data=data_std)
        f.create_dataset("explained_variance", data=pca.explained_variance_ratio_)
        f.create_dataset("eigenfish", data=pca.components_)
        for i, (species, data) in enumerate(species_data.items()):
            w = whitened[i]
            t = transformed[i]
            assert len(data) == len(w) == len(t)
            group = f.create_group(species)
            group.create_dataset("trial", data=data)
            group.create_dataset("whitened", data=w)
            group.create_dataset("transformed", data=t)
