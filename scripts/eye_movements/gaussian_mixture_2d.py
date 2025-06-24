import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import json

from experiment import *
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # ======================
    # GAUSSIAN MIXTURE MODEL
    # ======================
    for species in expt.species_names:
        if species != "l_ocellatus":
            continue
        print(species)
        analysis_dir = expt.directory["analysis", "eye_analysis", species]
        angles = []
        for f in analysis_dir["angles"].values():
            df = pd.read_csv(f)
            df = df[df["tracked"]]
            if len(df):
                angles.append(df[["left", "right"]].values)
        angles = np.concatenate(angles, axis=0)
        R = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                      [np.sin(np.pi/4), np.cos(np.pi/4)]])
        angles = np.einsum("ij,nj->ni", R, angles)
        angles = np.concatenate([angles, angles * (1, -1)], axis=0)
        gmm_params = {}
        for n_components in (1, 2, 3, 4):
            print("n_components =", n_components)
            mix = GaussianMixture(n_components=n_components, covariance_type="diag")
            mix.fit(angles)
            means = mix.means_
            covs = mix.covariances_
            weights = mix.weights_
            precision = mix.precisions_cholesky_
            bic = mix.bic(angles)
            gmm_params[f"n={n_components}"] = {
                "means": means.tolist(),
                "covariances": covs.tolist(),
                "weights": weights.tolist(),
                "precision": precision.tolist(),
                "bic": int(bic)
            }
        print()
        with open(analysis_dir.new_file("gmm_params_2d", "json"), "w") as f:
            json.dump(gmm_params, f)
