import pandas as pd
import numpy as np

from experiment import *
from matplotlib import pyplot as plt


if __name__ == "__main__":

    for species_id in ("l_attenuatus", "l_ocellatus", "n_multifasciatus"):
        frame_info_directory = expt.directory["tracking", "artemia", "frame_info", species_id]
        frame_directory = expt.directory["tracking", "artemia", "frames", species_id, "csvs"]
        all_counts = []
        for fish_id, fish_path in frame_info_directory.items():
            fish_counts = np.ones((21, 100)) * np.nan
            df = pd.read_csv(fish_path)
            for t, t_info in df.iterrows():
                if not pd.isnull(t_info.video_id):
                    frame_id = f"{t_info.video_id}_{int(t_info.frame_number)}-{int(t_info.frame_number) + 100}"
                    labels_path = frame_directory[frame_id + ".csv"]
                    labels = pd.read_csv(labels_path)
                    labels = labels[labels["confidence"] > 0.5]
                    time_counts = []
                    for frame, frame_artemia in labels.groupby("frame"):
                        time_counts.append(len(frame_artemia))
                    fish_counts[t, :len(time_counts)] = time_counts
            # fish_counts = np.pad(fish_counts, (1, 1), mode="edge")
            # fish_counts = np.convolve(fish_counts, np.ones(3) / 3, mode="valid")
            # fish_counts = np.cumsum(-np.diff(fish_counts))
            # fish_counts = np.pad(fish_counts, (1, 0))
            all_counts.append(fish_counts)
        all_counts = np.array(all_counts)
        output_path = expt.directory["analysis", "artemia", species_id].new_file("artemia_counts", "npy")
        np.save(output_path, all_counts)
