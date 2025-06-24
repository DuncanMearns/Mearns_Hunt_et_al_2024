import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from zigzeg import find_runs

from experiment import *


def interpolate(x, gap_size=5):
    # get transformed
    x = np.array(x)
    # find gaps to interpolate
    x_nan = np.any(np.isnan(x), axis=1)
    gaps, = np.where(x_nan)
    gaps = find_runs(gaps)
    gaps = [idxs for idxs in gaps if len(idxs) <= gap_size]
    if len(gaps):
        # get indices to interpolate
        interp_t = [idxs for idxs in gaps if
                    (~np.any(x_nan[idxs[0] - 3:idxs[0]]) & ~np.any(x_nan[idxs[-1] + 1:idxs[-1] + 4]))]
        if not len(interp_t):
            return x
        interp_t = np.hstack(interp_t)
        # interpolate
        t, = np.where(~x_nan)
        cs = CubicSpline(t, x[t])
        interp_x = cs(interp_t)
        x[interp_t] = interp_x
    return x


if __name__ == "__main__":

    bouts = np.load(expt.directory["analysis", "combined_analysis", "flipped_bouts.npy"])
    counts = np.load(expt.directory["analysis", "combined_analysis", "species_counts.npy"])
    counts = np.hstack([[0], np.cumsum(counts)])
    bouts_by_species = [bouts[i:j] for (i, j) in zip(counts, counts[1:])]
    n_bouts = [len(bouts) for bouts in bouts_by_species]

    output_directory = expt.directory["analysis", "combined_analysis"].new_subdir("strikes")

    for species_id in expt.species_names:
        # Open data
        bouts_df = pd.read_csv(expt.directory["analysis", "bouts", species_id, "bouts_clustered.csv"])
        species_idx = n_bouts.index(len(bouts_df))
        species_bouts = bouts_by_species[species_idx]
        if species_id == "o_latipes":
            strike_cluster = 5
            strike_idxs = bouts_df[bouts_df["cluster"] == strike_cluster].index
        else:
            strike_idxs = []
            # Group by trial
            bouts_by_trial = bouts_df.groupby("trial")
            # Find putative strikes
            events = pd.read_csv(expt.directory["analysis", "eye_analysis", species_id, "events.csv"])
            for idx, event_info in events.iterrows():
                trial_bouts = bouts_by_trial.get_group(event_info.video_id)
                peak_to_end = trial_bouts["peak"] - event_info.end
                nearest = np.abs(peak_to_end).argmin()
                t_offset = peak_to_end.iloc[nearest] / 400.
                if -0.1 <= t_offset <= 0.2:
                    bout_idx = trial_bouts.index[nearest]
                    strike_idxs.append(bout_idx)
        # Set strike bouts
        strike_idxs = np.sort(strike_idxs)
        strikes = species_bouts[strike_idxs]
        # Interpolate and crop
        is_null = np.all(strikes == 0, axis=-1)
        strikes[np.where(is_null)] = np.nan
        strikes = np.array([interpolate(x, 5) for x in strikes])
        strikes = strikes[:, 50:200]
        # Remove nan bouts
        is_null = np.all(np.isnan(strikes), axis=-1)
        is_null = np.sum(is_null, axis=1)
        keep, = np.where(is_null == 0)
        # Final strikes
        strike_ids = strike_idxs[keep]
        strikes = strikes[keep]
        strike_df = bouts_df.loc[strike_ids]
        strike_df["bout_index"] = strike_df.index
        t = []
        for idx, bout_info in strike_df.iterrows():
            video_info = expt.video_data[expt.video_data["video_id"] == bout_info.trial].squeeze()
            t.append(bout_info.start / video_info.frame_rate)
        strike_df["time"] = t
        # Save
        npy_path = output_directory.new_file(species_id, "npy")
        np.save(npy_path, np.array(strikes))
        csv_path = output_directory.new_file(species_id, "csv")
        strike_df.to_csv(csv_path, index=False)
