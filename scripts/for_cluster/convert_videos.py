import sys
from pathlib import Path
import cv2
import pandas as pd
import time
from joblib import Parallel, delayed


def convert_video(src, dst):
    # Check output
    if Path(dst).exists():
        return
    # Open video
    cap = cv2.VideoCapture(str(src))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Create output
    writer = cv2.VideoWriter(str(dst),
                             cv2.VideoWriter_fourcc(*"MPEG"),
                             fps,
                             (w, h), True)
    # Save frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()
    cap.release()


if __name__ == "__main__":

    expt_dir = sys.argv[1]
    expt_dir = Path(expt_dir)

    video_data = pd.read_csv(expt_dir.joinpath("video_data.csv"))
    video_data = video_data.groupby("species_id").get_group("a_burtoni")

    src_dir = expt_dir.joinpath("videos", "a_burtoni")
    dst_dir = expt_dir.joinpath("videos_mp4")
    if not dst_dir.exists():
        dst_dir.mkdir()

    func_calls = []
    for idx, video_info in video_data.iterrows():
        path = src_dir.joinpath(video_info.date, video_info.fish_name, video_info.timestamp + ".avi")
        output = dst_dir.joinpath(video_info.video_id + ".mp4")
        delayed_call = delayed(convert_video)(path, output)
        func_calls.append(delayed_call)

    t = time.time()
    result = Parallel(n_jobs=4)(func_calls)
    print((time.time() - t) / 60, "minutes")
