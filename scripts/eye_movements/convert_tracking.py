import pandas as pd

from experiment import *
import h5py
import numpy as np


if __name__ == "__main__":
    min_angle = params["min_angle"]
    max_angle = params["max_angle"]
    conf = params["confidence_threshold"]
    for species, tracking_directory in expt.directory["tracking", "eyes"].items():
        if species != "n_multifasciatus":
            continue
        print(species)
        angles_directory = expt.directory.new_subdir("analysis", "eye_analysis", species, "angles")
        species_data = expt.video_data.groupby("species_id").get_group(species)
        # ========
        # GET DATA
        # ========
        all_convergence = []
        for idx, video_info in species_data.iterrows():
            output_path = angles_directory.new_file(video_info.video_id, "csv")
            try:
                path = tracking_directory[video_info.fish_id, video_info.video_id + ".h5"]
                with h5py.File(path, "r") as h5:
                    tracking = h5["tracks"][0]
                    left = tracking[:, (0, 2)]
                    right = tracking[:, (1, 3)]
                    sb_mid = tracking[:, (6, 5, 4)]
                    points_scores = h5["point_scores"][0, :4, :].T
            except (ValueError, KeyError, OSError):
                print("Error:", video_info.video_id)
                # angles = np.empty((0, 2))
                angles = pd.DataFrame(columns=["left", "right", "convergence", "tracked"])
                angles.to_csv(output_path, index=False)
                continue
            sb_mid = sb_mid[:, 0] - sb_mid[:, 1:].mean(axis=1)
            left = np.diff(left, axis=1).squeeze()
            right = np.diff(right, axis=1).squeeze()
            # Transpose
            sb_mid = sb_mid.T
            left = left.T
            right = right.T
            # Normalize
            sb_mid = sb_mid / np.linalg.norm(sb_mid, axis=1, keepdims=True)
            left = left / np.linalg.norm(left, axis=1, keepdims=True)
            right = right / np.linalg.norm(right, axis=1, keepdims=True)
            # Angles
            left_sign = np.sign(np.cross(sb_mid, left))
            left_angle = np.arccos(np.einsum("ni,ni->n", sb_mid, left)) * left_sign
            right_sign = np.sign(np.cross(sb_mid, right))
            right_angle = np.arccos(np.einsum("ni,ni->n", sb_mid, right)) * right_sign
            # Convert to degrees
            left_angle = np.degrees(left_angle)
            right_angle = np.degrees(right_angle)
            # Tracked indices
            keep_tracks = (left_angle >= -min_angle) & \
                          (left_angle <= max_angle) & \
                          (right_angle >= -max_angle) & \
                          (right_angle <= min_angle) & \
                          (np.all(points_scores >= conf, axis=1))
            angles_data = dict(left=left_angle,
                               right=right_angle,
                               convergence=left_angle-right_angle,
                               tracked=keep_tracks)
            angles = pd.DataFrame(angles_data, columns=["left", "right", "convergence", "tracked"])
            all_convergence.append(angles[angles["tracked"]]["convergence"].values)
            angles.to_csv(output_path, index=False)
        all_convergence = np.concatenate(all_convergence, axis=0)
        np.save(expt.directory["analysis", "eye_analysis", species].new_file(f"convergence.npy"),
                all_convergence)
