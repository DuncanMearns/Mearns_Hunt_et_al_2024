from experiment import *
import numpy as np
from sklearn.mixture import GaussianMixture
import json


if __name__ == "__main__":
    # ======================
    # GAUSSIAN MIXTURE MODEL
    # ======================
    np.random.seed(2023)
    for species in expt.species_names:
        print(species)
        analysis_dir = expt.directory["analysis", "eye_analysis", species]
        conv = np.load(analysis_dir["convergence.npy"])
        gmm_params = {}
        for n_components in (1, 2):
            mix = GaussianMixture(n_components=n_components, covariance_type="spherical")
            mix.fit(conv[:, np.newaxis])
            means = mix.means_[:, 0]
            covs = mix.covariances_
            weights = mix.weights_
            precision = mix.precisions_cholesky_
            bic = mix.bic(conv[:, np.newaxis])
            gmm_params[f"n={n_components}"] = {
                "means": list(means),
                "covariances": list(covs),
                "weights": list(weights),
                "precision": list(precision),
                "bic": int(bic)
            }
        print(gmm_params)
        with open(analysis_dir.new_file("gmm_params", "json"), "w") as f:
            json.dump(gmm_params, f)
