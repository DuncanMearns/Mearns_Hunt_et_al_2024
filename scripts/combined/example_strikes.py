import pandas as pd
import numpy as np
import cv2

from behavior_analysis.io import TrackingInterface
from experiment import *
from skimage.transform import AffineTransform, warp
from skimage.exposure import rescale_intensity
from matplotlib import pyplot as plt


if __name__ == "__main__":

    examples = {"o_latipes": 4, "l_attenuatus": 1}
    sizes = {"o_latipes": 20, "l_attenuatus": 30}
    rescale = {"o_latipes": (50, 200), "l_attenuatus": (20, 230)}
    final_pixel_size = 0.05

    strikes_path = expt.directory["analysis", "combined_analysis", "strikes", "clustered.csv"]
    strikes = pd.read_csv(strikes_path)
    for l, species in enumerate(("o_latipes", "l_attenuatus"), start=1):
        species_strikes = strikes[strikes["species"] == species]
        species_strikes = species_strikes[species_strikes["strike_cluster"] == l]
        example_idx = examples[species]

        strike = species_strikes.iloc[example_idx]

        video_info = expt.video_data[expt.video_data["video_id"] == strike.trial].squeeze()
        video_path = expt.data_directory[
            species, "videos", video_info.date, video_info.fish_name, video_info.timestamp + ".avi"
        ]
        cap = cv2.VideoCapture(str(video_path))

        tracking_path = expt.directory[
            "tracking", "eyes", species, video_info.fish_id, video_info.video_id + ".h5"
        ]
        tracking = TrackingInterface(tracking_path)
        data = tracking.read()
        frame_number = strike.peak
        data = data[frame_number]
        sb = data[6]
        mp = np.mean(data[[4, 5]], axis=0)
        v = mp - sb
        v /= np.linalg.norm(v)
        theta = (np.pi / 2) - np.arctan2(v[1], v[0])
        height, width = video_info.height, video_info.width

        scale = final_pixel_size / (sizes[species] / width)

        A1 = AffineTransform(translation=-sb).params
        A2 = AffineTransform(rotation=theta + np.pi).params
        A3 = AffineTransform(translation=(width // 2, height // 2), scale=scale).params
        A = A3 @ A2 @ A1
        affine = AffineTransform(matrix=A)

        cap.set(cv2.CAP_PROP_POS_FRAMES, strike.peak)
        ret, frame = cap.read()

        centered = warp(frame[..., 0], affine.inverse)

        w = int(width * scale * 0.15)
        h = int(height * scale * 0.1)
        centered = centered[height // 2 - h:height // 2 + (2 * h), width // 2 - w:width // 2 + w]
        centered = (centered * 255).astype("uint8")

        centered = rescale_intensity(centered, rescale[species])

        np.save(expt.directory["analysis", "combined_analysis"].new_file(species, "npy"), centered)
        # cv2.imshow(species, centered)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
