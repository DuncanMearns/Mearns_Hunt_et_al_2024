from experiment import *
import numpy as np

if __name__ == "__main__":
    # Remove bad trials
    bad_trials_path = expt.directory["bad_videos.txt"]
    bad_trials = []
    with open(bad_trials_path, "r") as f:
        for trial in f.readlines():
            bad_trials.append(trial.replace("\n", ""))
    metadata = expt.video_data
    metadata = metadata[~np.isin(metadata["video_id"], bad_trials)]
    metadata.to_csv(expt.directory["video_data.csv"], index=False)
