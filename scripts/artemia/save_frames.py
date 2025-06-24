from experiment import *
import pandas as pd
import cv2

if __name__ == "__main__":

    for species_id, species_data in expt.video_data.groupby("species_id"):
        if species_id in ("a_burtoni", "o_latipes"):
            continue
        output_directory = expt.directory.new_subdir("tracking", "artemia", "frames", species_id)
        for fish_id, fish_data in species_data.groupby("fish_id"):
            path = expt.directory["tracking", "artemia", "frame_info", species_id, fish_id + ".csv"]
            frame_ids = pd.read_csv(path)
            frame_ids = frame_ids[~pd.isnull(frame_ids["frame_number"])]
            for video_id, video_frames in frame_ids.groupby("video_id"):
                vi = fish_data[fish_data["video_id"] == video_id].squeeze()
                video_path = expt.data_directory[species_id, "videos", vi.date, vi.fish_name, vi.timestamp + ".avi"]
                cap = cv2.VideoCapture(str(video_path))
                for frame_number in video_frames["frame_number"].values.astype("int"):
                    output_path = output_directory.new_file(f"{video_id}_{frame_number}", "jpg")
                    if output_path.exists():
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                    cv2.imwrite(str(output_path), frame)
                cap.release()
