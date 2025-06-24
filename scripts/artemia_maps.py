from experiment import *
from behavior_analysis.io import TrackingInterface
import numpy as np
import pandas as pd

if __name__ == "__main__":
    for species, species_directory in expt.directory["artemia_tracking"].items():
        if species == "a_burtoni":  # skip AB
            continue
        print(species)
        data = pd.read_csv(species_directory["labels", "frames.csv"])
        # Create column with video_id
        data["video_id"] = data["file_name"].apply(lambda x: str(x).split("-")[0])
        # All points
        artemia = []
        nearest = []
        for trial_id, trial_data in data.groupby("video_id"):
            # Open tracking data
            fish_id, timestamp = str(trial_id).split("_")
            tracking_path = expt.directory["tracking", species, fish_id, f"{trial_id}.h5"]
            tracking = TrackingInterface(tracking_path)
            tracked_frames = tracking.is_tracked()
            tracking_data = tracking.read()
            heading_points = tracking_data[:, tracking.heading_node_idxs]
            # Get video info
            video_info = expt.video_data[expt.video_data["video_id"] == trial_id].squeeze()
            w, h = video_info.width, video_info.height
            # Iterate annotated frames in video
            for frame_id, frame_data in trial_data.groupby("file_name"):
                frame_number = int(str(frame_id).split("-")[1])
                # Check if frame is tracked
                if np.isin(frame_number, tracked_frames):
                    # Get artemia positions
                    xs = frame_data["xcenter"].values  # * w
                    ys = frame_data["ycenter"].values  # * h
                    xy = np.vstack([xs, ys]).T
                    # Get fish position
                    fish = heading_points[frame_number] / (w, h)
                    c = fish[0]
                    xy_centered = xy - c
                    v = (fish[1] - fish[0]) / np.linalg.norm(fish[1] - fish[0])
                    theta = -np.arctan2(v[1], v[0])
                    R = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
                    # Rotate
                    xy_rotated = np.einsum("ij,nj->ni", R, xy_centered)
                    artemia.append(xy_rotated)
                    # Nearest
                    front = xy_rotated[xy_rotated[:, 0] > 0]
                    if len(front):
                        dists = np.linalg.norm(front, axis=-1)
                        art = front[np.argmin(dists)]
                        nearest.append(art)
                    # ==============
                    # CHECK PLOTTING
                    # --------------
                    # img = cv2.imread(str(expt.directory["artemia_tracking", species, "frames", f"{frame_id}.jpg"]))
                    # plt.imshow(img)
                    # plt.plot(*fish.T)
                    # plt.scatter(fish[0, [0]], fish[0, [1]])
                    # plt.scatter(*xy.T)
                    # plt.gca().set_aspect("equal")
                    # plt.show()
                    # ==============
        artemia = np.concatenate(artemia, axis=0)
        np.save(expt.directory["artemia_tracking", species].new_file("positions", "npy"), artemia)
        nearest = np.array(nearest)
        np.save(expt.directory["artemia_tracking", species].new_file("nearest", "npy"), nearest)
        # print(len(artemia))
        # plt.scatter(*artemia.T, lw=0, c="k", s=1, alpha=0.5)
        # plt.plot([0, 0.05], [0, 0], c="r", lw=1)
        # plt.show()
