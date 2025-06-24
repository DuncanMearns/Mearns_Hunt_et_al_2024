import sys
from pathlib import Path
import pandas as pd
import cv2
import time
from joblib import Parallel, delayed
import os
import shutil
import numpy as np


def save_frames_as_images(src, dst, starts, ends):
    dst = Path(dst)
    dst.mkdir(exist_ok=True, parents=True)
    cap = cv2.VideoCapture(str(src))
    for start, end in zip(starts, ends):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for f in range(start, end):
            ret, frame = cap.read()
            if ret:
                output_path = dst.joinpath(f"{dst.name}_{f}.jpg")
                cv2.imwrite(str(output_path), frame)
    cap.release()
    return dst


def run_yolo_predict(script_path, weights_path, data_path, img_size, output_directory, name):
    yolo_predict_command = [f'python {script_path}',
                            f'--weights "{weights_path}"',
                            f'--source "{data_path}"',
                            f'--imgsz {img_size}',
                            f'--save-txt',
                            f'--save-conf',
                            f'--project "{output_directory}"',
                            f'--name {name}',
                            f'--exist-ok',
                            f'--nosave']
    yolo_predict_command = ' '.join(yolo_predict_command)
    return_code = os.system(yolo_predict_command)
    return return_code


def yolo_cleanup(src, dst, name, starts, ends):
    src = Path(src)
    dst = Path(dst)
    for start, end in zip(starts, ends):
        output_path = dst.joinpath(f"{name}_{start}-{end}.csv")
        data = []
        to_remove = []
        for frame_number in range(start, end):
            txt_file = src.joinpath(f"{name}_{frame_number}.txt")
            if not txt_file.exists():
                continue
            with open(txt_file, "r") as txt:
                lines = txt.readlines()
            lines = list(map(lambda line: [frame_number] + line.rstrip().split(' ')[1:], lines))
            data += lines
            to_remove.append(txt_file)
        data = pd.DataFrame(data, columns=['frame',
                                           'xcenter',
                                           'ycenter',
                                           'width',
                                           'height',
                                           'confidence'])
        data.to_csv(output_path, index=False)
        for txt_file in to_remove:
            os.remove(txt_file)


def find_artemia(video_id, video_path, img_directory, txt_directory, csv_directory,
                 yolo_path, model_path, starts, ends, imgsize):
    img_directory = Path(img_directory).joinpath(video_id)
    txt_directory = Path(txt_directory)
    save_frames_as_images(video_path, img_directory, starts, ends)
    run_yolo_predict(yolo_path, model_path, img_directory, imgsize, txt_directory, video_id)
    labels_directory = txt_directory.joinpath(video_id, "labels")
    yolo_cleanup(labels_directory, csv_directory, video_id, starts, ends)
    if labels_directory.exists():
        os.rmdir(labels_directory)
        os.rmdir(labels_directory.parent)
    shutil.rmtree(img_directory)


if __name__ == "__main__":

    EXPT_DIR, SPECIES, VIDEO_DIR, YOLO_DETECT, N_CORES = sys.argv[1:]
    EXPT_DIR = Path(EXPT_DIR)
    VIDEO_DIR = Path(VIDEO_DIR)
    N_CORES = int(N_CORES)

    output_dir = EXPT_DIR.joinpath("tracking", "artemia", "frames", SPECIES)
    output_dir.mkdir(exist_ok=True, parents=True)

    img_dir = output_dir.joinpath("imgs")
    img_dir.mkdir(exist_ok=True, parents=True)
    txt_dir = output_dir.joinpath("txts")
    txt_dir.mkdir(exist_ok=True, parents=True)
    csv_dir = output_dir.joinpath("csvs")
    csv_dir.mkdir(exist_ok=True, parents=True)

    model = EXPT_DIR.joinpath("tracking", "artemia", "model", "weights", "best.pt")

    video_data = pd.read_csv(EXPT_DIR.joinpath("video_data.csv"))
    species_data = video_data[video_data["species_id"] == SPECIES]

    frame_info_dir = EXPT_DIR.joinpath("tracking", "artemia", "frame_info", SPECIES)

    func_calls = []

    for fish_path in frame_info_dir.glob("*.csv"):
        df = pd.read_csv(fish_path)
        for video_id, video_events in df.groupby("video_id"):
            video_info = video_data[video_data["video_id"] == video_id].squeeze()
            video_path = VIDEO_DIR.joinpath(video_info.date, video_info.fish_name, video_info.timestamp + ".avi")
            start_frames = video_events["frame_number"].values
            start_frames = start_frames[~np.isnan(start_frames)].astype("i4")
            end_frames = start_frames + 100
            final_output = csv_dir.joinpath(f"{video_id}_{start_frames[-1]}-{end_frames[-1]}.csv")
            if not final_output.exists():
                func = delayed(find_artemia)(video_id, video_path, img_dir, txt_dir, csv_dir,
                                             YOLO_DETECT, model, start_frames, end_frames, video_info.width)
                func_calls.append(func)

    t = time.time()
    ret = Parallel(N_CORES)(func_calls)
    print((time.time() - t) / 60, "minutes")
