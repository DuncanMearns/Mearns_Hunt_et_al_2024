import h5py

from experiment import *
import numpy as np
import pandas as pd


if __name__ == "__main__":
    strike_dist_mm = params["strike_dist_mm"]
    offset_time = params["strike_offset_time"]
    min_mov_dist = params["min_prey_movement"]
    for species_id in expt.species_names:
        try:
            ego_directory = expt.directory["analysis", "artemia", species_id, "egocentric"]
        except ValueError:
            continue
        # Find non-boundary events
        events = pd.read_csv(expt.directory["analysis", "eye_analysis", species_id, "events.csv"])
        events = events[~events["boundary"]]
        data = {}  # all data for species
        for idx, event in events.iterrows():
            # Get video and event info
            video_info = expt.video_data[expt.video_data["video_id"] == event.video_id].squeeze()
            fps = video_info.frame_rate
            start, end = event.start, event.end
            event_id = f"{event.video_id}_{event.start}-{event.end}"
            # Load tracks
            t = int(offset_time * fps)
            tracks = np.load(ego_directory[f"{event_id}.npy"])
            # Invert y-axis (array coordinates to math coordinates)
            tracks[..., 1] *= -1
            # Order tracks by distance to strike zone at event offset
            dists = []
            for track in np.moveaxis(tracks, 1, 0):
                nonan = track[~np.any(np.isnan(track), axis=-1)]
                last = np.mean(track[-t:], axis=0)
                dists.append(np.linalg.norm(last - (strike_dist_mm, 0), axis=-1))
            order = np.argsort(dists)
            tracks = tracks[:, order]
            # Find most likely target
            event_data = {"tracks": tracks}
            for i, track in enumerate(np.moveaxis(tracks, 1, 0)):
                nonan = track[~np.any(np.isnan(track), axis=-1)]
                mov_dist = np.linalg.norm(nonan[0] - nonan[-1])
                theta = np.arccos(nonan[:, 0] / np.linalg.norm(nonan, axis=-1))
                # Filter by movement dist, initial azimuthal position, and average azimuthal position
                if (mov_dist > min_mov_dist) & (np.mean(theta[:t]) < (3 * np.pi / 4)) & (theta.mean() < np.pi / 2):
                    onset = np.mean(nonan[:t], axis=0)
                    mi = len(nonan) // 2
                    middle = np.mean(nonan[mi - t: mi + t], axis=0)
                    offset = np.mean(nonan[-t:], axis=0)
                    event_data["target_index"] = i
                    event_data["onset"] = onset
                    event_data["middle"] = middle
                    event_data["offset"] = offset
                    event_data["frame_rate"] = fps
                    event_data["fish_id"] = video_info.fish_id
                    event_data["start"] = start
                    event_data["end"] = end
                    break
            data[event_id] = event_data
        # Save all tracks together
        output_path = expt.directory["analysis", "artemia", species_id].new_file("events_data.h5")
        with h5py.File(output_path, "w") as h5:
            for event_id, event_data in data.items():
                tracks = event_data.pop("tracks")
                dset = h5.create_dataset(event_id, data=tracks)
                dset.attrs.update(event_data)
