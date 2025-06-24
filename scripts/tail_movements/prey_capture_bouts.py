import numpy as np
import pandas as pd

from experiment import *


if __name__ == "__main__":

    for species_id in expt.species_names:
        if species_id == "o_latipes":
            continue
        print(species_id)

        bouts_df = pd.read_csv(expt.directory["analysis", "bouts", species_id, "bouts_clustered.csv"])

        predictions = {}
        for video_id in bouts_df["trial"].unique():
            prediction = np.load(expt.directory["analysis", "eye_analysis", species_id, "predict", f"{video_id}.npy"])
            predictions[video_id] = prediction

        mean_scores = []
        peak_scores = []
        init_scores = []
        end_scores = []
        for idx, bout_info in bouts_df.iterrows():
            frame_rate = expt.video_data[expt.video_data["video_id"] == bout_info.trial].squeeze().frame_rate
            t = int(0.1 * frame_rate)
            prediction = predictions[bout_info.trial]
            try:
                mean_score = np.nanmean(prediction[bout_info.start:bout_info.end])
                peak_score = prediction[bout_info.peak]
                init_score = prediction[bout_info.start]
                end_score = prediction[bout_info.end]
            except IndexError:
                mean_score, peak_score, init_score, end_score = np.empty(4) + np.nan
            mean_scores.append(mean_score)
            peak_scores.append(peak_score)
            init_scores.append(init_score)
            end_scores.append(end_score)
        mean_scores = np.array(mean_scores)
        peak_scores = np.array(peak_scores)
        init_scores = np.array(init_scores)
        end_scores = np.array(end_scores)

        bouts_df["mean_score"] = mean_scores
        bouts_df["peak_score"] = peak_scores
        bouts_df["init_score"] = init_scores
        bouts_df["end_score"] = end_scores

        output_path = expt.directory["analysis", "bouts", species_id].new_file("bouts_scored", "csv")
        bouts_df.to_csv(output_path, index=False)
