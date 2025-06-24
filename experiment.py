from behavior_analysis.experiment import PreyCaptureExperiment, TrialData

__all__ = ("expt", "ARENA_SIZE", "get_trials", "get_params")

# =====================
# EXPERIMENT PARAMETERS
# =====================

directory = r"PATH_TO_EXPERIMENT_DIRECTORY"
video_directory = r"PATH_TO_VIDEO_DIRECTORY"

ARENA_SIZE = 30

_default_params = {

    # PCA
    "n_components": 3,

    # Find bouts
    "winsize": 0.1,
    "bandwidth": 0.005,
    "minsize": 0.05,

    # Bout filtering
    "dim_2_thresh": 10.,
    "n_gauss_mix": 3,

    # Interpolation
    "gap_size": 4,

    # Bout alignment
    "alignment_percentile": 80,
    "n_align_clusters": 30,

    # Eye angle filtering
    "min_angle": 10,
    "max_angle": 60,
    "confidence_threshold": 0.8,
    "convergence_ksize": 0.2,
    "event_window": 360,
    "convergence_threshold": 0.8,

    # Artemia
    "strike_dist_mm": 2,
    "strike_offset_time": 0.1,
    "min_prey_movement": 0.1

}


def get_params(species_id):

    params = dict(_default_params)

    # Species-specific parameters
    if species_id == "a_burtoni":
        params["n_gauss_mix"] = 2

    if species_id == "l_attenuatus":
        params["bandwidth"] = 0.01
        params["dim_2_thresh"] = 5.

    if species_id == "l_ocellatus":
        params["bandwidth"] = 0.02
        params["n_gauss_mix"] = 2

    if species_id == "o_latipes":
        params["n_gauss_mix"] = 10

    return params


# ===============
# OPEN EXPERIMENT
# ===============

expt = PreyCaptureExperiment.open(directory, data_directory=video_directory)
TrialData.directory = expt.directory["tracking", "tail"]
TrialData.eye_directory = expt.directory["tracking", "eyes"]


def get_trials(species_id):
    species_metadata = expt.video_data.groupby("species_id").get_group(species_id)
    trials = []
    for idx, trial_info in species_metadata.iterrows():
        try:
            trial_data = TrialData(**trial_info.to_dict())
            assert trial_data.tracking_path.exists()
            trials.append(trial_data)
        except ValueError:
            print(f"Missing data: {trial_info.video_id}")
    return trials


if __name__ == "__main__":
    # create metadata when script is run as __main__
    expt.create_metadata()
