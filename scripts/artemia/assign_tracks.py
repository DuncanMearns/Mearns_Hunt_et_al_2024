import sys
sys.path.append(r"C:\Users\dm3169\PycharmProjects\comparative_prey_capture")
from experiment import *
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class ArtemiaTracker:

    def __init__(self, x_init, err_tol):
        self.x = x_init
        self.err_tol = err_tol

    @property
    def n_items(self):
        return len(self.x)

    @property
    def n_dim(self):
        return self.x.shape[1]

    def update(self, z):
        z = z[~np.any(np.isnan(z), axis=-1)]
        x_idxs, = np.where(~np.any(np.isnan(self.x), axis=-1))
        y = cdist(self.x[x_idxs], z)
        row, col = linear_sum_assignment(y)
        z_assigned = []
        for i, j in zip(row, col):
            if y[i, j] < self.err_tol:
                self.x[x_idxs[i]] = z[j]
                z_assigned.append(j)
        x_unassigned = [i for i in np.arange(self.n_items) if i not in x_idxs]
        z_unassigned = [j for j in np.arange(len(z)) if j not in z_assigned]
        n_unassigned = len(z_unassigned)
        if n_unassigned <= len(x_unassigned):
            self.x[x_unassigned[:n_unassigned]] = z[z_unassigned]
        else:
            self.x = np.vstack([self.x, z[z_unassigned]])
        return self.x


if __name__ == "__main__":
    for species_id in expt.species_names:
        # Src and dst directories
        species_directory = expt.directory["tracking", "artemia", "sequences"].joinpath(species_id)
        if not species_directory.exists():
            continue
        output_directory = expt.directory["analysis"].new_subdir("artemia", species_id, "sequences")
        # Iter events
        events = pd.read_csv(expt.directory["analysis", "eye_analysis", species_id, "events.csv"])
        events = events[~events["boundary"]]
        for idx, event in events.iterrows():
            video_info = expt.video_data[expt.video_data["video_id"] == event.video_id].squeeze()
            csv_path = species_directory.joinpath(f"{event.video_id}_{event.start}-{event.end}.csv")
            output_path = output_directory.new_file(csv_path.stem, "npy")
            if output_path.exists():
                continue
            prey_df = pd.read_csv(csv_path)
            prey_df[["xcenter", "ycenter"]] *= (video_info.width, video_info.height)
            # Track prey
            tracker = None
            tracks = []
            for f in range(event.start, event.end):
                frame_prey = prey_df[prey_df["frame"] == f]
                xy = frame_prey[["xcenter", "ycenter"]].values
                if tracker:
                    tracker.update(xy)
                else:
                    tracker = ArtemiaTracker(xy, 20)
                tracks.append(tracker.x)
            max_len = len(tracks[-1])
            tracks = map(lambda a: np.pad(a, ((0, max_len - len(a)), (0, 0)), constant_values=np.nan), tracks)
            tracks = np.array(list(tracks))  # n_frames, n_prey, xy
            # Save
            np.save(output_path, tracks)
