import sys
import os
import yaml
from src.yolo_wrapper import YOLOWrapper


if __name__ == "__main__":

    dest_dir, src_dir, model_dir, *_ = sys.argv[1:]

    weights_path = os.path.join(model_dir, "weights", "best.pt")
    with open(os.path.join(model_dir, 'opt.yaml'), 'rb') as config_file:
        image_size = yaml.safe_load(config_file)["imgsz"]

    yolo_wrapper = YOLOWrapper()
    yolo_wrapper.predict(dest_dir, src_dir, weights_path, image_size, False, False, False, 0)
