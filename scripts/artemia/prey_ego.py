import numpy as np
import pandas as pd

from experiment import *
from behavior_analysis.io import TrackingInterface


if __name__ == "__main__":
    for species_id in expt.species_names:
        try:
            seq_directory = expt.directory["analysis", "artemia", species_id, "sequences"]
        except ValueError:
            continue
        output_directory = expt.directory["analysis", "artemia", species_id].new_subdir("egocentric")
        # Events
        events = pd.read_csv(expt.directory["analysis", "eye_analysis", species_id, "events.csv"])
        events = events[~events["boundary"]]
        for video_id, video_events in events.groupby("video_id"):
            # Tracking
            fish_id, *_ = video_id.split("_")
            tracking_path = expt.directory["tracking", "eyes", species_id, fish_id, video_id + ".h5"]
            tracking = TrackingInterface(tracking_path)
            data = tracking.read()
            sb = data[:, 6]
            mp = np.mean(data[:, (4, 5)], axis=1)
            heading_vectors = mp - sb
            heading_vectors /= np.linalg.norm(heading_vectors, axis=-1, keepdims=True)
            # Video info
            video_info = expt.video_data[expt.video_data["video_id"] == video_id].squeeze()
            xscale = ARENA_SIZE / video_info.width
            yscale = ARENA_SIZE / video_info.height
            # Prey tracks
            for tracks_path in seq_directory.glob(f"{video_id}_*.npy"):
                output_path = output_directory.new_file(tracks_path.name)
                if output_path.exists():
                    continue
                *_, start_end = tracks_path.stem.split("_")
                start, end = start_end.split("-")
                start, end = int(start), int(end)
                # Get xy prey positions
                xy = np.load(tracks_path)
                # Centre and rotate
                xy -= mp[start:end, np.newaxis]
                v1 = heading_vectors[start:end] * np.array([[1, -1]])
                v2 = heading_vectors[start:end, ::-1]
                R = np.concatenate([v1[..., np.newaxis], v2[..., np.newaxis]], axis=-1)
                xyR = np.einsum("nij,npj->npi", R, xy) * (xscale, yscale)
                # Save
                np.save(output_path, xyR)
