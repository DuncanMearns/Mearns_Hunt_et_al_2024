from experiment import *
from behavior_analysis.frame_annotation import annotate_frames
import pandas as pd
import cv2
import numpy as np


if __name__ == "__main__":

    species_id = "o_latipes"
    side_swing = 1

    output_directory = expt.directory["analysis"].new_subdir("medaka")
    output_path = output_directory.new_file("prey_positions", "csv")

    if output_path.exists():
        strikes = pd.read_csv(output_path)
    else:
        strikes_path = expt.directory["analysis", "combined_analysis", "strikes", "clustered.csv"]
        strikes = pd.read_csv(strikes_path)
        strikes = strikes[strikes["species"] == species_id]
        strikes = strikes[strikes["strike_cluster"] == side_swing]
        strikes["x"] = None
        strikes["y"] = None
        strikes["annotated"] = False
        strikes.to_csv(output_path, index=False)

    to_analyze = strikes[~strikes["annotated"]]

    for video_id, video_strikes in to_analyze.groupby("trial"):
        video_info = expt.video_data[expt.video_data["video_id"] == video_id].squeeze()
        video_path = expt.data_directory[
            species_id, "videos", video_info.date, video_info.fish_name, video_info.timestamp + ".avi"
        ]
        cap = cv2.VideoCapture(str(video_path))
        for idx, strike_info in video_strikes.iterrows():
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, strike_info.start)
            for f in range(strike_info.start, strike_info.peak):
                ret, frame = cap.read()
                frames.append(frame)
            frames = np.array(frames)
            x, y = annotate_frames(frames)
            strikes.loc[idx, "x"] = x
            strikes.loc[idx, "y"] = y
            strikes.loc[idx, "annotated"] = True
            strikes.to_csv(output_path, index=False)
        cap.release()
