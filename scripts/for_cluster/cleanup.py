import sys
import pandas as pd
from pathlib import Path


def transform_yolo_line_2_table_row(file_name, frame_number, line):
    cls, xc, yc, w, h, conf = line.rstrip().split(' ')
    return [file_name, frame_number, xc, yc, w, h, conf]


if __name__ == "__main__":

    dst, src, *_ = sys.argv[1:]

    dst = Path(dst)
    src = Path(src)

    labels = dst.joinpath("result", "labels")
    names = list(map(lambda x: x.stem, src.glob("*.avi")))

    output_dir = dst.joinpath("csvs")
    output_dir.mkdir(exist_ok=True)

    for vid in names:
        txt_files = labels.glob(f"{vid}*.txt")
        data = []
        for txt in txt_files:
            frame = txt.stem.split("_")[-1]
            with open(txt, "r") as f:
                lines = f.readlines()
            lines = list(map(lambda x: transform_yolo_line_2_table_row(vid, frame, x), lines))
            data += lines
        df = pd.DataFrame(data, columns=['file_name',
                                         'frame',
                                         'xcenter',
                                         'ycenter',
                                         'width',
                                         'height',
                                         'confidence'])
        output_path = output_dir.joinpath(vid + ".csv")
        df.to_csv(output_path, index=False)
        df = pd.read_csv(output_path)
        df.sort_values("frame", inplace=True)
        df.to_csv(output_path, index=False)
