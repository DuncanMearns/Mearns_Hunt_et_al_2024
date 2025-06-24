import pandas as pd

from experiment import *
import numpy as np
import json
from sklearn.mixture import GaussianMixture


if __name__ == "__main__":

    np.random.seed(2023)

    winsize = params["convergence_ksize"]
    min_angle = params["min_angle"]
    max_angle = params["max_angle"]
    # ====================
    # PREDICT PREY CAPTURE
    # ====================
    for species, species_data in expt.video_data.groupby("species_id"):
        if species == "o_latipes":
            continue
        print(species)
        # Set mixture components
        with open(expt.directory["analysis", "eye_analysis", species, "gmm_params.json"], "r") as f:
            gmm = json.load(f)
            model = gmm["n=2"]
        means = model["means"]
        covs = model["covariances"]
        weights = model["weights"]
        precision = model["precision"]
        # Create GMM
        gmm = GaussianMixture(n_components=2, covariance_type="spherical")
        gmm.means_ = np.array(means)[:, np.newaxis]
        gmm.covariances_ = np.array(covs)
        gmm.weights_ = np.array(weights)
        gmm.precisions_cholesky_ = np.array(precision)
        # Get converged index
        converged_index = np.argmax(means)
        # Iter through videos
        angles_directory = expt.directory["analysis", "eye_analysis", species, "angles"]
        predict_directory = expt.directory["analysis", "eye_analysis", species].new_subdir("predict")
        for idx, video_info in species_data.iterrows():
            output_path = predict_directory.new_file(video_info.video_id, "npy")
            # Kernel for filtering
            kernel = np.ones(int(winsize * video_info.frame_rate))
            kernel = kernel / len(kernel)
            # Load angles
            path = angles_directory[video_info.video_id + ".csv"]
            angles = pd.read_csv(path)
            if not len(angles):
                np.save(output_path, np.empty(0))
                continue
            # Find tracked
            is_tracked = angles["tracked"]
            conv = angles[is_tracked]["convergence"].values
            # Score convergence probability
            scored = gmm.predict_proba(conv[:, np.newaxis])[:, converged_index]
            # Convolve
            converged = np.zeros(len(angles)) + np.nan
            converged[np.where(is_tracked)] = scored
            converged = np.convolve(converged, kernel, mode="same")
            # Save
            np.save(output_path, converged)
