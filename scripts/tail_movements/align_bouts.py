import h5py
import json
import numpy as np
import pandas as pd

from experiment import *


if __name__ == "__main__":

    for species in expt.species_names:

        print(species)

        # Params
        params = get_params(species)
        percentile = params["alignment_percentile"]
        gap_size = params["gap_size"]

        # Paths
        aligned_bouts_path = expt.directory["analysis", "bouts", species].new_file("aligned_bouts", "npy")
        bouts_path = expt.directory["analysis", "bouts", species, "bouts_cleaned.csv"]
        params_path = expt.directory["analysis", "bouts", species, "params.json"]
        pca_path = expt.directory["analysis", "postural_decomposition", f"{species}.h5"]

        # Import bouts
        bouts_df = pd.read_csv(bouts_path)
        start_to_peak = (bouts_df["peak"] - bouts_df["start"]).values
        peak_to_end = (bouts_df["end"] - bouts_df["peak"]).values

        # Compute number of frames before and after peaks
        pre_peak = int(np.percentile(start_to_peak, percentile))
        post_peak = int(np.percentile(peak_to_end, percentile))

        # Import PCA
        n_components = params["n_components"]
        with h5py.File(pca_path, "r") as f:
            mean = f["mean"][:]
            std = f["std"][:]
            components = f["eigenfish"][:n_components]

        # Import trials
        trials = get_trials(species)

        # Align bouts by trial
        aligned = []
        bouts_by_trial = bouts_df.groupby("trial")
        for trial in trials:
            try:
                trial_bouts = bouts_by_trial.get_group(trial.video_id)
            except KeyError:
                continue
            trial.transform_tail(components, mean, std)
            trial.interpolate(gap_size)
            trial_aligned = trial.align(trial_bouts, pre_peak, post_peak)
            aligned.extend(trial_aligned)
        aligned = np.array(aligned)

        # Save
        np.save(aligned_bouts_path, aligned)
        # Save params
        with open(params_path, "r") as f:
            params = json.load(f)
        params["pre_peak"] = pre_peak
        params["post_peak"] = post_peak
        with open(params_path, "w") as f:
            json.dump(params, f)
