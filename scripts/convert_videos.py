import sys
from pathlib import Path
import cv2
import pandas as pd
from joblib import Parallel, delayed


def convert_video(src, dst):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MPEG"), fps, (w, h), True)
    while True:
        ret, frame = cap.read()
        writer.write(frame)
        if not ret:
            break
    writer.release()
    cap.release()


if __name__ == "__main__":

    expt_dir, species = sys.argv[1:]
    expt_dir = Path(expt_dir)

    output_dir = expt_dir.joinpath("videos_mp4", species)
    output_dir.mkdir(exist_ok=True, parents=True)

    video_data = pd.read_csv(expt_dir.joinpath("video_data.csv"))
    video_data = video_data[video_data["species_id"] == species]

    func_calls = []
    for idx, video_info in video_data.iterrows():
        video_path = expt_dir.joinpath("videos", species, video_info.date, video_info.fish_name,
                                       video_info.timestamp + ".avi")
        output_path = output_dir.joinpath(video_info.video_id + ".mp4")
        if not output_path.exists():
            f = delayed(convert_video)(video_path, output_path)
            func_calls.append(f)

    ret = Parallel(n_jobs=8)(func_calls)

    # fish_ids = video_data["fish_id"].unique()
    # fish_ids = list(filter(lambda x: str(x).endswith(("2", "4", "6")), fish_ids))

    # by_fish = video_data.groupby("fish_id")
    # for fish_id in fish_ids:
    #     fish_data = by_fish.get_group(fish_id)
    #     video_info = fish_data.iloc[0]
    #     video_path = expt_dir.joinpath("videos", species, video_info.date, video_info.fish_name,
    #                                    video_info.timestamp + ".avi")
    #     output_path = output_dir.joinpath(video_info.video_id + ".mp4")
    #     convert_video(video_path, output_path)
