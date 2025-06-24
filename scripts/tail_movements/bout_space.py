from experiment import *
import numpy as np
from sklearn.manifold import TSNE


if __name__ == "__main__":

    for species in expt.species_names:

        print(species)

        # Paths
        tsne_path = expt.directory["analysis", "bouts", species].new_file("bout_space", "npy")
        flipped_bouts_path = expt.directory["analysis", "bouts", species, "flipped_bouts.npy"]
        # Import data
        bouts = np.load(flipped_bouts_path)
        reshaped = bouts.reshape(len(bouts), -1)
        # Generate bout space
        tsne = TSNE()
        bout_space = tsne.fit_transform(reshaped)
        # Save
        np.save(tsne_path, bout_space)
