import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from behavior_analysis.io import TrackingInterface

from experiment import *

if __name__ == "__main__":

    pre_strike = 0.02
    pixel_size = 20 / 500

    path = expt.directory["analysis", "medaka", "prey_positions.csv"]
    strikes = pd.read_csv(path)
    strikes = strikes[~pd.isnull(strikes["x"])]

    xys = []
    lefts = []
    rights = []

    for idx, strike in strikes.iterrows():
        video_info = expt.video_data[expt.video_data["video_id"] == strike.trial].squeeze()
        tracking_path = expt.directory[
            "tracking", "eyes", "o_latipes", video_info.fish_id, video_info.video_id + ".h5"
        ]
        tracking = TrackingInterface(tracking_path)
        data = tracking.read()
        frame_number = strike.peak - int(pre_strike * video_info.frame_rate)
        data = data[frame_number]
        sb = data[6]
        mp = np.mean(data[[4, 5]], axis=0)
        left = data[[0, 2, 4]].mean(axis=0)
        right = data[[1, 3, 5]].mean(axis=0)
        v = mp - sb
        v /= np.linalg.norm(v)
        theta = (np.pi / 2) - np.arctan2(v[1], v[0])
        xy = np.array([strike.x, strike.y])
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        # Rotate
        xy = np.dot(R, xy - mp) * pixel_size
        xy[0] *= np.sign(xy[0])
        left = np.dot(R, left - mp) * pixel_size
        right = np.dot(R, right - mp) * pixel_size
        xys.append(xy)
        lefts.append(left)
        rights.append(right)
    xys = np.array(xys)
    lefts = np.array(lefts)
    rights = np.array(rights)
    points = np.concatenate([xys[:, np.newaxis], lefts[:, np.newaxis], rights[:, np.newaxis]], axis=1)
    np.save(expt.directory["analysis", "medaka"].new_file("points", "npy"), points)

    # fig, ax = plt.subplots()
    # ax.scatter(*xys.T)
    # ax.scatter(*lefts.T)
    # ax.scatter(*rights.T)
    # ax.set_aspect("equal")
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # plt.show()
