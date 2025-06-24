from experiment import *

import numpy as np
import cv2
from scipy import ndimage


def rotate_and_center(img: np.ndarray, xy: tuple, radians: float):
    h, w = img.shape
    x, y = xy
    cos = np.cos(-radians)
    sin = np.sin(-radians)
    M1 = np.array([[1, 0, -y],
                   [0, 1, -x],
                   [0, 0, 1]])
    M2 = np.array([[cos, sin, 0],
                   [-sin, cos, 0],
                   [0, 0, 1]])
    M3 = np.array([[1, 0, h / 2],
                   [0, 1, w / 2],
                   [0, 0, 1]])
    M = M3 @ M2 @ M1
    img_rot = ndimage.affine_transform(img, np.linalg.inv(M))
    return img_rot


if __name__ == "__main__":

    # # for species in ("a_burtoni", "l_attenuatus", "l_ocellatus", "o_latipes"):
    for species in ("o_latipes",):

        # Paths
        bouts_path = expt.directory["analysis", "bouts", species, "bouts_clustered.csv"]
        params_path = expt.directory["analysis", "bouts", species, "params.json"]
        output_directory = expt.directory["analysis"].new_subdir("cluster_videos", species)

    #     # Bouts
    #     bouts = pd.read_csv(bouts_path)
    #     with open(params_path, "r") as f:
    #         bout_params = json.load(f)
    #
    #     np.random.seed(202306)
    #
    #     for label, cluster_bouts in bouts.groupby("cluster"):
    #         idxs = np.random.choice(cluster_bouts.index, 25, False)
    #         idxs = np.sort(idxs)
    #         example_bouts = cluster_bouts.loc[idxs]
    #         cluster_frames = []
    #         for idx, bout_info in example_bouts.iterrows():
    #             # Bout info
    #             # start = max([0, bout_info.peak - bout_params["pre_peak"]])
    #             start = bout_info.peak - bout_params["pre_peak"]
    #             end = bout_info.peak + bout_params["post_peak"]
    #             # Trial info
    #             trial_info = expt.video_data[expt.video_data["video_id"] == bout_info.trial].iloc[0]
    #             # Video
    #             video_path = expt.data_directory[
    #                 species, "videos", trial_info.date, trial_info.fish_name, trial_info.timestamp + ".avi"]
    #             cap = cv2.VideoCapture(str(video_path))
    #             w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #             h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #             # Tracking
    #             tracking_path = expt.directory["tracking", species, trial_info.fish_id, trial_info.video_id + ".h5"]
    #             tracking = TrackingInterface(tracking_path)
    #             points = tracking.read()
    #             centroids = points[:, tracking.centroid_idx, :]
    #             angles = tracking.headings()
    #             # Bout
    #             cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    #             bout_frames = []
    #             for f in range(start, end):
    #                 ret, frame = cap.read()
    #                 frame = frame[..., 0]
    #                 c = centroids[f]
    #                 angle = angles[f]
    #                 img = rotate_and_center(frame, c, angle)
    #                 img = img[(h // 2) - 150: (h // 2) + 150, (w // 2) - 200: (w // 2) + 100]
    #                 bout_frames.append(img)
    #             cap.release()
    #             cluster_frames.append(bout_frames)
    #         cluster_frames = np.array(cluster_frames)
    #         path = output_directory.new_file(f"cluster_{label}", "npy")
    #         np.save(path, cluster_frames)

        # for f in output_directory.values():
        #     print(f)
        #     cluster = f.stem
        for l in [10]:
            cluster = f"cluster_{l}"
            f = output_directory[cluster+".npy"]
            try:
                frames = np.load(f)
            except ValueError:
                continue
            print(frames.shape)
            frames = frames.reshape(3, 3, frames.shape[1], frames.shape[2], frames.shape[3])
            frames = np.concatenate(frames, axis=2)
            frames = np.concatenate(frames, axis=2)
            frames = frames[:, ::3, ::3]
            cv2.namedWindow(cluster)
            for frame in frames:
                cv2.imshow(cluster, frame)
                cv2.waitKey(10)
            cv2.destroyWindow(cluster)
