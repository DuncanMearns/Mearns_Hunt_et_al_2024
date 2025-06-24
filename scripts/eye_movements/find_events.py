from experiment import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zigzeg import find_runs


def get_datetime(date, timestamp):
    Y, M, D = date.split("_")
    h, m, s = timestamp.split("-")
    return datetime(int(Y), int(M), int(D), int(h), int(m), int(s))


if __name__ == "__main__":

    thresh = params["convergence_threshold"]
    window = params["event_window"]

    for species, species_data in expt.video_data.groupby("species_id"):
        if species == "o_latipes":
            continue
        print(species)
        all_durations = []
        fish_medians = []
        fish_proportions = []
        fish_rates = []
        all_events = []
        for fish_id, fish_data in species_data.groupby("fish_id"):
            video_info = fish_data.iloc[0]
            start_time = get_datetime(video_info.date, video_info.timestamp)
            # Get data within time window
            video_times = fish_data["n_frames"] / fish_data["frame_rate"]
            # Fish parameters
            fish_events = []
            fish_early_time = 1
            for idx, video_info in fish_data.iterrows():
                video_time = get_datetime(video_info.date, video_info.timestamp)
                time_delta = video_time - start_time
                pc = np.load(expt.directory["analysis", "eye_analysis", species, "predict", f"{video_info.video_id}.npy"])
                if not len(pc):
                    continue
                not_nan, = np.where(~np.isnan(pc))
                section_idxs = find_runs(not_nan)
                for idxs in section_idxs:
                    first = idxs[0]
                    section = pc[idxs]
                    # Add section to total time
                    dt = (time_delta + timedelta(seconds=first / video_info.frame_rate)).total_seconds()
                    is_early = (dt <= window)
                    if is_early:
                        fish_early_time += len(section) / video_info.frame_rate
                    # Find events
                    above_thresh, = np.where(section > 0.5)
                    events = find_runs(above_thresh)
                    events = [event for event in events if
                              (np.sum(section[event] > thresh) / len(event) > 0.5)
                              ]
                    # Add total time and time in prey capture
                    if not len(events):
                        continue
                    # Add events
                    for x in events:
                        f0 = first + x[0]
                        f1 = first + x[-1]
                        is_boundary = (x[0] == 0) | (x[-1] == len(section) - 1)
                        fish_events.append((video_info.fish_id,
                                            video_info.video_id,
                                            f0,
                                            f1,
                                            is_boundary,
                                            is_early,
                                            video_info.frame_rate))
            if not len(fish_events):
                continue
            fish_events = pd.DataFrame(fish_events, columns=["fish_id",
                                                             "video_id",
                                                             "start",
                                                             "end",
                                                             "boundary",
                                                             "early",
                                                             "frame_rate"])
            early_events = fish_events[fish_events["early"]]
            durations = (early_events["end"] - early_events["start"]) / early_events["frame_rate"]
            # Total time in prey capture
            pc_time = durations.sum()
            # PC rate
            pc_rate = len(early_events) / fish_early_time
            # Median hunting episode (excluding boundary episodes)
            durations = durations[~early_events["boundary"]]
            if len(durations):
                fish_med = np.median(durations)
                if fish_med > 5:  # remove outlier (NM)
                    continue
                fish_medians.append(fish_med)
                all_durations.extend(durations.values)
            # Add to species data
            fish_proportions.append(pc_time / fish_early_time)
            fish_rates.append(pc_rate)
            all_events.append(fish_events)

        np.save(expt.directory["analysis", "eye_analysis", species].new_file("fish_medians", "npy"),
                np.array(fish_medians))

        np.save(expt.directory["analysis", "eye_analysis", species].new_file("durations", "npy"),
                np.array(all_durations))

        np.save(expt.directory["analysis", "eye_analysis", species].new_file("proportions", "npy"),
                np.array(fish_proportions))

        np.save(expt.directory["analysis", "eye_analysis", species].new_file("rates", "npy"),
                np.array(fish_rates))

        df = pd.concat(all_events, axis=0)
        df.to_csv(expt.directory["analysis", "eye_analysis", species].new_file("events", "csv"), index=False)
