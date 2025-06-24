from experiment import *
import pandas as pd
import cv2
import numpy as np


if __name__ == "__main__":

    species = "l_attenuatus"

    events_path = expt.directory["analysis", "eye_analysis", species, "events.csv"]
    videos_directory = expt.directory["tracking", "artemia", species, "videos"]
    labels_directory = expt.directory["tracking", "artemia", species, "csvs"]

    events = pd.read_csv(events_path)

    for i, event in events.iterrows():

        event_id = f"{event.video_id}-{event.start}-{event.end}"

        video_path = videos_directory[event_id + ".avi"]
        labels_path = labels_directory[event_id + ".csv"]

        labels = pd.read_csv(labels_path).groupby("frame")

        cap = cv2.VideoCapture(str(video_path))
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        for fn in range(event.start + 1, event.end + 1):

            frame = np.zeros((int(h), int(w)), dtype="uint8")

            pos_info = labels.get_group(fn)
            xs = pos_info["xcenter"].values * w
            ys = pos_info["ycenter"].values * h

            for (x, y) in zip(xs, ys):
                cv2.circle(frame, (int(x), int(y)), 3, 255, -1)

            cv2.imshow("window", frame)
            cv2.waitKey(1)

        cv2.destroyWindow(event_id)
        cap.release()

        break
