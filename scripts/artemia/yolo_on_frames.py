from run_yolo import run_yolo_predict
from pathlib import Path


if __name__ == "__main__":

    yolo_script = r"C:\Users\dm3169\PycharmProjects\YOLO-Gooey\yolov5\detect.py"

    expt_directory = Path(r"D:\percomorph_analysis")
    weights_path = expt_directory.joinpath("tracking", "artemia", "model", "weights", "best.pt")
    img_size = 800

    for species_id in ("l_attenuatus", "l_ocellatus", "n_multifasciatus"):
        frames_directory = expt_directory.joinpath("tracking", "artemia", "frames", species_id)
        run_yolo_predict(yolo_script, weights_path, frames_directory, 800, frames_directory, "positions")
