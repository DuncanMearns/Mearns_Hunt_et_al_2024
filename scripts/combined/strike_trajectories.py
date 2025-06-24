from experiment import *
from behavior_analysis.io import TrackingInterface
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress


if __name__ == "__main__":

    strike_clusters = [1, 2, 4]
    time_window = 0.2  # time (seconds) to study pre-strike
    onset_time = 0.025  # time before peak
    frame_rate = 400.

    output_directory = expt.directory["analysis", "combined_analysis"].new_subdir("heading")

    strikes_df = pd.read_csv(expt.directory["analysis", "combined_analysis", "strikes", "clustered.csv"])
    strikes_df = strikes_df[strikes_df["strike_cluster"].isin(strike_clusters)]

    t = np.linspace(0, time_window, int(time_window * frame_rate))

    for species, species_strikes in strikes_df.groupby("species"):
        if species not in ("l_attenuatus", "o_latipes"):
            continue
        thetas = []
        slopes = []
        # ax = next(axes)
        for video_id, video_strikes in species_strikes.groupby("trial"):
            video_info = expt.video_data[expt.video_data["video_id"] == video_id].squeeze()
            tracking_path = expt.directory[
                "tracking", "eyes", species, video_info.fish_id, video_id + ".h5"
            ]
            tracking = TrackingInterface(tracking_path)
            data = tracking.read()
            sb = data[:, 6]
            mp = np.mean(data[:, (4, 5)], axis=1)
            # Frames to take
            n_frames = int(time_window * video_info.frame_rate)
            pre_peak = int(onset_time * video_info.frame_rate)
            # Iter strikes
            for idx, strike_info in video_strikes.iterrows():
                centers, vectors = np.ones((2, n_frames, 2)) * np.nan
                t1 = strike_info.peak
                t0 = max([0, t1 - n_frames])
                frame_diff = t0 - (t1 - n_frames)
                centers[frame_diff:] = sb[t0:t1]
                vectors[frame_diff:] = mp[t0:t1] - sb[t0:t1]
                # Center
                centers -= centers[-pre_peak]
                # Normalize vectors
                vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
                # Rotate
                theta = -np.arctan2(vectors[-pre_peak, 1], vectors[-pre_peak, 0])
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
                centers = np.einsum("ij,nj->ni", R, centers)
                vectors = np.einsum("ij,nj->ni", R, vectors)
                # Heading relative to end
                theta = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
                theta *= np.sign(np.nanmean(theta))
                # Interpolate
                nonan, = np.where(~np.isnan(theta))
                t_native = np.linspace(0, time_window, n_frames)
                interp = interp1d(t_native[nonan], theta[nonan], kind="cubic")
                max_t = t_native[nonan][-1]
                theta = interp(t[t <= max_t])
                theta = np.pad(theta, (0, len(t) - len(theta)), constant_values=np.nan)

                # Compute regression
                nonan, = np.where(~np.isnan(theta))
                t_reg = t[nonan]
                theta_reg = theta[nonan]
                theta_reg = theta_reg[t_reg <= (time_window - onset_time)]
                t_reg = t_reg[t_reg <= (time_window - onset_time)]
                reg = linregress(t_reg, theta_reg)

                slopes.append(reg.slope)
                thetas.append(theta)

        thetas = np.array(thetas)
        slopes = np.array(slopes)

        np.save(output_directory.new_file(species + "_headings", "npy"), thetas)
        np.save(output_directory.new_file(species + "_slopes", "npy"), slopes)

    # print(mannwhitneyu(all_slopes[0], all_slopes[1]))
    # plt.violinplot(all_slopes)
    # plt.show()
