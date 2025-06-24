import h5py
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from experiment import *


if __name__ == "__main__":

    for species in expt.species_names:

        print(species)

        # Params
        params = get_params(species)
        n_gauss_mix = params["n_gauss_mix"]
        gap_size = params["gap_size"]

        # Paths
        pca_path = expt.directory["analysis", "postural_decomposition", f"{species}.h5"]
        bouts_path = expt.directory["analysis", "bouts", species, "bouts.csv"]
        params_path = expt.directory["analysis", "bouts", species, "params.json"]
        clean_bouts_path = expt.directory["analysis", "bouts", species].new_file("bouts_cleaned", "csv")
        clean_idxs_path = expt.directory["analysis", "bouts", species].new_file("clean_idxs", "npy")
        bout_std_path = expt.directory["analysis", "bouts", species].new_file("bout_stds", "npy")

        # Import PCA
        n_components = params["n_components"]
        with h5py.File(pca_path, "r") as f:
            mean = f["mean"][:]
            std = f["std"][:]
            components = f["eigenfish"][:n_components]

        # Import trials
        trials = get_trials(species)

        # Import bouts
        bouts_df = pd.read_csv(bouts_path)
        bouts_by_trial = bouts_df.groupby("trial")
        bouts = []
        for trial in trials:
            trial.transform_tail(components, mean, std)
            trial.interpolate(gap_size)
            try:
                trial_bouts = bouts_by_trial.get_group(trial.video_id)
            except KeyError:
                print(f"No bouts in {species} - {trial.fish_id}: {trial.video_id}")
                continue
            xs = trial.from_bouts(trial.transformed, trial_bouts)
            bouts.extend(xs)

        # Fit gaussian mixture model to bout variances
        bout_stds = np.array([bout.std(axis=0) for bout in bouts])
        mix = GaussianMixture(n_components=n_gauss_mix).fit_predict(bout_stds)

        # Remove bouts
        group_means = np.array([bout_stds[mix == l].mean(axis=0) for l in range(n_gauss_mix)])
        smallest = np.argmin(np.linalg.norm(group_means, axis=-1))
        keep_idxs, = np.where(mix != smallest)
        keep_bouts = bouts_df.loc[keep_idxs]

        # Save
        keep_bouts.to_csv(clean_bouts_path, index=False)
        np.save(clean_idxs_path, keep_idxs)
        # Save params
        with open(params_path, "r") as f:
            params = json.load(f)
        params["n_gauss_mix"] = n_gauss_mix
        with open(params_path, "w") as f:
            json.dump(params, f)
        # Save mixture model
        np.save(bout_std_path, bout_stds)
