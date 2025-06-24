from experiment import *
from datetime import datetime
import numpy as np
import pandas as pd


if __name__ == "__main__":

    dt = 30
    t_end = 600

    for species_id, species_data in expt.video_data.groupby("species_id"):
        if species_id in ("a_burtoni", "o_latipes"):
            continue
        output_directory = expt.directory.new_subdir("tracking", "artemia", "frame_info", species_id)
        for fish_id, fish_data in species_data.groupby("fish_id"):
            output_path = output_directory.new_file(fish_id + ".csv")
            vid0 = fish_data.iloc[0]
            t0 = datetime.strptime(f"{vid0.date} {vid0.timestamp}", "%Y_%m_%d %H-%M-%S")
            t0 = t0.timestamp()
            frame_ids = []
            t = 0
            for idx, video_info in fish_data.iterrows():
                vt = datetime.strptime(f"{video_info.date} {video_info.timestamp}", "%Y_%m_%d %H-%M-%S")
                vt_start = vt.timestamp() - t0
                duration = (video_info.n_frames - 1) / video_info.frame_rate
                vt_times = np.linspace(vt_start, vt_start + duration, video_info.n_frames)
                while t < vt_times[-1]:
                    positive, = np.where((vt_times - t) >= 0)
                    if len(positive):
                        frame_number = positive[0]
                        if (vt_times[frame_number] - t) > 5:
                            frame_ids.append([None, None, None])
                        else:
                            frame_ids.append((video_info.video_id,
                                              frame_number,
                                              vt_times[frame_number]))
                        t += dt
                        if t > t_end:
                            break
                if t > t_end:
                    break
            frame_ids = pd.DataFrame(frame_ids, columns=["video_id", "frame_number", "time"])
            frame_ids.to_csv(output_path, index=False)
