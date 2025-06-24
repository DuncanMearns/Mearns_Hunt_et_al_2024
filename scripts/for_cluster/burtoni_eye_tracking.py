import sys
import time
from joblib import Parallel, delayed
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance as ssd
from pathlib import Path


def background_normalization(img, background):
    sub = background - img
    norm = (sub + 1) / (background + 1)
    img = np.clip(norm * 255, 0, 255).astype("uint8")
    return img


def find_contours(image, threshold, n=-1, invert=False):
    """Finds all the contours in an image after binarizing with the threshold.
    Parameters
    ----------
    image : array like
        Unsigned 8-bit integer array.
    threshold : int
        Threshold applied to images to find contours.
    n : int (default = -1)
        Number of contours to be extracted (-1 for all contours identified with a given threshold).
    invert : bool (default = False)
        Whether to invert the binarization.
    Returns
    -------
    contours : list
        A list of arrays representing all the contours found in the image sorted by contour area (largest first)
        after applying the threshold.
    """
    # apply threshold
    if invert:
        ret, threshed = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        ret, threshed = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    # find contours
    try:
        img, contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        contours, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort in descending size order
    contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)
    if n > -1:
        return contours[:n]
    return contours


def get_eye_points(contours):
    centroids = np.array([contour.mean(axis=0) for contour in contours])
    distances = ssd.pdist(centroids)
    sb_index = 2 - distances.argmin()
    eye_idxs = [i for i in range(3) if i != sb_index]
    eyes = [contours[i] for i in eye_idxs]
    eye_centroids = centroids[eye_idxs]
    sb_centroid = centroids[sb_index]
    eye_vectors = eye_centroids - sb_centroid
    cross_product = np.cross(*eye_vectors)
    cross_sign = int(np.sign(cross_product))
    eyes = eyes[::cross_sign]  # left, right
    eye_centroids = eye_centroids[::cross_sign]
    heading_vector = eye_centroids.mean(axis=0) - sb_centroid
    heading_vector /= np.linalg.norm(heading_vector)
    points = [sb_centroid]
    is_left = True
    for c, eye in zip(eye_centroids, eyes):
        c, wh, theta = cv2.fitEllipse(eye)
        v = np.array([-np.sin(np.radians(theta)), np.cos(np.radians(theta))])
        v = v * np.sign(np.dot(v, heading_vector))
        c = np.array(c)
        p1 = c + (v * wh[1] / 2.)  # nasal
        p2 = c - (v * wh[1] / 2.)  # temporal
        v_orth = v[::-1] * (-1, 1)
        if is_left:
            p3 = c + (v_orth * wh[0] / 2.)  # middle
        else:
            p3 = c - (v_orth * wh[0] / 2.)  # middle
        points.extend([p1, p2, p3])
        is_left = not is_left
    points = np.array(points)
    return points


def track_eyes(path, bg, thresh1, thresh2):
    cap = cv2.VideoCapture(str(path))
    bg = bg.astype("float64")
    all_points = []
    nan = np.zeros((7, 2)) + np.nan
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[..., 0].astype('float64')
        img = background_normalization(frame, bg)
        fish_cntr = find_contours(img, thresh1, 1)
        masked = np.zeros_like(img)
        cv2.drawContours(masked, fish_cntr, 0, 1, -1)
        masked[masked == 1] = img[masked == 1]
        masked = cv2.equalizeHist(masked)
        cntrs = find_contours(masked, thresh2, 3)
        if len(cntrs) < 3:
            all_points.append(nan)
            continue
        cntrs = [cntr[:, 0, :] for cntr in cntrs]
        if any([cntr.shape[0] < 5 for cntr in cntrs]):  # need at least five points for ellipse function to work
            all_points.append(nan)
            continue
        points = get_eye_points(cntrs)  # sb, left nasal, left temp, left mid, right nasal, right temp, right mid
        all_points.append(points)
    cap.release()
    return np.array(all_points)


def run_eye_tracking(input_path, output_path, background, thresh1, thresh2):
    points = track_eyes(input_path, background, thresh1, thresh2)
    np.save(output_path, points)


if __name__ == "__main__":

    expt_path, *_ = sys.argv[1:]

    species = "a_burtoni"

    expt_path = Path(expt_path)
    video_data = pd.read_csv(expt_path.joinpath("video_data.csv"))
    video_data = video_data.groupby("species_id").get_group(species)

    fish_data = pd.read_csv(expt_path.joinpath("burtoni_data.csv")).groupby("name")

    output_directory = expt_path.joinpath("eye_tracking_new", species)
    output_directory.mkdir(exist_ok=True, parents=True)

    video_args = []

    for idx, video_info in video_data.iterrows():
        # Paths and tracking params
        video_path = expt_path.joinpath(
            "videos", video_info.species_id, video_info.date, video_info.fish_name, video_info.timestamp + ".avi"
        )
        fish_info = fish_data.get_group(video_info.fish_name).squeeze()
        thresh1, thresh2 = int(fish_info.thresh1), int(fish_info.thresh2)
        bg_path = expt_path.joinpath("backgrounds", species, str(fish_info.ID) + ".tiff")
        bg = cv2.imread(str(bg_path), 0)
        output_path = output_directory.joinpath(video_info.video_id + ".npy")
        if output_path.exists():
            continue
        print(video_path)
        video_args.append(delayed(run_eye_tracking)(video_path, output_path, bg.copy(), thresh1, thresh2))

    t = time.time()
    Parallel(n_jobs=4)(video_args)
    print((time.time() - t) / 60, "minutes")
