import colorsys
import numpy as np
import matplotlib as mpl

species_names = ["L. ocellatus", "N. multifasciatus", "L. attenuatus", "A. burtoni", "O. latipes"]
species_ids = list(map(lambda x: "_".join([x[0].lower(), x.split(" ")[1]]), species_names))
species_labels = list(map(lambda x: x[0].upper() + x.split(" ")[1][0].upper(), species_names))

n_species = len(species_names)
species_colors = np.array([colorsys.hsv_to_rgb(h, 0.5, 0.8) for h in
                           np.linspace(1.7, 0.7, n_species + 1)[1:]])

FONT_SIZE = 8
FIG_WIDTH = 1.7
FIG_HEIGHT = 1.7

mpl.rcParams["lines.linewidth"] = 1
mpl.rcParams["lines.dash_capstyle"] = "round"
mpl.rcParams["lines.solid_capstyle"] = "round"

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.size"] = FONT_SIZE
mpl.rcParams["font.sans-serif"] = "Arial"

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.labelsize"] = FONT_SIZE
mpl.rcParams["axes.unicode_minus"] = False

mpl.rcParams["xtick.labelsize"] = FONT_SIZE
mpl.rcParams["ytick.labelsize"] = FONT_SIZE

mpl.rcParams["xtick.major.size"] = 2
mpl.rcParams["xtick.minor.size"] = 1.5
mpl.rcParams["ytick.major.size"] = 2
mpl.rcParams["ytick.minor.size"] = 1.5

mpl.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_HEIGHT)
mpl.rcParams["figure.dpi"] = 300
