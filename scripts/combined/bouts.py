from experiment import *
import numpy as np
import pandas as pd
import h5py
from zigzeg import align_peaks


if __name__ == "__main__":
    # Params
    n_components = 3
    fps = 400.
    gap_size = 4
    pre_peak = 0.3
    post_peak = 0.7

    pre_peak_idx = int(pre_peak * fps)
    post_peak_idx = int(post_peak * fps)

    # Experiment
    directory = r"D:\percomorph_analysis"

    output_directory = expt.directory.new_subdir("analysis", "combined_analysis", "bouts")

    # Canonical PCs
    pca_path = expt.directory["analysis", "postural_decomposition", "canonical_pca.h5"]
    with h5py.File(pca_path, "r") as f:
        mean = f["mean"][:]
        std = f["std"][:]
        components = f["eigenfish"][:n_components]

    for species, species_trials in expt.video_data.groupby("species_id"):

        bouts_path = expt.directory["analysis", "bouts", species, "bouts_cleaned.csv"]
        bouts_df = pd.read_csv(bouts_path)
        bouts_df_trials = bouts_df.groupby("trial")

        peak_times = []
        bouts = []

        trials = get_trials(species)
        for trial in trials:
            trial.transform_tail(components, mean, std)
            trial.interpolate(gap_size)
            t, transformed = trial.interpolate_frame_rate(fps)
            try:
                trial_bouts = bouts_df_trials.get_group(trial.video_id)
            except KeyError:
                continue

            for jdx, bout_info in trial_bouts.iterrows():
                t0 = bout_info.start / trial.frame_rate
                t1 = bout_info.end / trial.frame_rate
                bout_frames = (t >= t0) & (t <= t1)
                bout = transformed[bout_frames]
                new_peak = fps * (bout_info.peak - bout_info.start) / trial.frame_rate
                bouts.append(bout)
                peak_times.append(int(new_peak))

        aligned = align_peaks(bouts, peak_times, pre_peak_idx, post_peak_idx)
        output_path = output_directory.new_file(species, "npy")
        np.save(output_path, aligned)
