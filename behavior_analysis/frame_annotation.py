from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseButton
import numpy as np


class Annotator:

    def __init__(self, frames):
        self.frames = frames
        self.frame_number = 0
        self.im = self.frames[self.frame_number]
        self.xy = (None, None)
        # Defined on start
        self.ax = None
        self.point = None

    @property
    def n_frames(self):
        return len(self.frames)

    def slider_changed(self, val):
        self.frame_number = val
        self.update()

    def on_click(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:
                self.xy = (event.xdata, event.ydata)
            elif event.button == 3:
                self.xy = (None, None)
            self.update()

    def update(self):
        self.im.set_data(self.frames[self.frame_number])
        x, y = self.xy
        if (x is not None) and (y is not None):
            self.point.set_offsets([x, y])
        else:
            self.point.set_offsets([np.nan, np.nan])
        plt.draw()

    def start(self):
        fig, self.ax = plt.subplots(figsize=(3, 3), dpi=300)
        self.im = self.ax.imshow(self.frames[self.frame_number])
        self.point = self.ax.scatter([], [], c="k", lw=0, s=15)
        ax_t = fig.add_axes([0.2, 0.05, 0.5, 0.03])
        slider = Slider(ax_t, "Time", 0, self.n_frames - 1, 0, valstep=1)
        slider.on_changed(self.slider_changed)
        plt.connect("button_press_event", self.on_click)
        plt.show()
        return self.xy


def annotate_frames(frames):
    ann = Annotator(frames)
    return ann.start()
