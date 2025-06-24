from behavior_analysis.kinematics import find_threshold
import h5py
import json
import numpy as np
import pandas as pd

from experiment import *


if __name__ == "__main__":

    for species in expt.species_names:

        print(species)
        params = get_params(species)

        # Get params
        n_components = params["n_components"]
        winsize = params["winsize"]
        bandwidth = params["bandwidth"]
        minsize = params["minsize"]
        dim_2_thresh = params["dim_2_thresh"]
        gap_size = params["gap_size"]

        # Paths
        bouts_directory = expt.directory.new_subdir("analysis", "bouts", species)
        bouts_path = bouts_directory.new_file("bouts", "csv")
        kde_path = bouts_directory.new_file("kde", "npy")
        params_path = bouts_directory.new_file("params", "json")
        pca_path = expt.directory["analysis", "postural_decomposition", f"{species}.h5"]

        # Load pcs
        with h5py.File(pca_path, "r") as f:
            mean = f["mean"][:]
            std = f["std"][:]
            components = f["eigenfish"][:n_components]

        # Get trials
        trials = get_trials(species)

        # Filter transformed tail
        print("Filtering tail data...", end=" ")
        data = []
        for trial in trials:
            trial.transform_tail(components, mean, std)
            trial.interpolate(gap_size)
            filtered = trial.filter_tail(trial.seconds_to_frames(winsize))
            data.append(filtered)
        print("done.")

        # Find threshold
        print("Finding threshold...", end=" ")
        data = np.concatenate(data, axis=0)
        data = data[~np.isnan(data)]
        thresh, (bins, counts) = find_threshold(data, bandwidth=bandwidth)
        print("done.")
        kde = np.vstack([bins, counts]).T
        np.save(kde_path, kde)

        # Find bouts
        print("Finding bouts...", end=" ")
        all_bouts = []
        for trial in trials:
            bouts_info = trial.find_bouts(thresh,
                                          minsize=trial.seconds_to_frames(minsize),
                                          winsize=trial.seconds_to_frames(winsize))
            bouts = trial.from_bouts(trial.transformed, bouts_info)
            if (trial.transformed.shape[1] > 2) and dim_2_thresh:
                keep_idxs, = np.where([np.all(np.abs(bout[:, 2]) < dim_2_thresh) for bout in bouts])
                bouts_info = bouts_info.loc[keep_idxs]
                bouts = [bouts[i] for i in keep_idxs]
            # Re-compute peaks
            peaks = [np.linalg.norm(bout, axis=-1).argmax() for bout in bouts]
            bouts_info["peak"] = peaks
            bouts_info["peak"] += bouts_info["start"]
            all_bouts.append(bouts_info)
        print("done.")

        # Save bouts
        bouts_df = pd.concat(all_bouts, axis=0, ignore_index=True)
        bouts_df.to_csv(bouts_path, index=False)

        # Save params
        bout_params = {
            "n_components": n_components,
            "winsize": winsize,
            "bandwidth": bandwidth,
            "minsize": minsize,
            "dim_2_thresh": dim_2_thresh,
            "gap_size": gap_size,
            "thresh": thresh
        }
        with open(params_path, "w") as f:
            json.dump(bout_params, f)
