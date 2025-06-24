Mearns Duncan S, Hunt Sydney A, Schneider Martin W, Parker Ash V, Stemmer Manuel, Baier Herwig (2024) Diverse prey capture strategies in teleost larvae eLife 13:RP98347
https://doi.org/10.7554/eLife.98347.2

# Metadata
Metadata, tracking files and analysis are organized using Experiment object from exptlib: https://github.com/DuncanMearns/exptlib

1. experiment.py

# Tail tracking

1. pca.py
2. find_bouts.py
3. clean_bouts.py
4. align_bouts.py
5. flip_bouts.py
6. cluster_bouts.py
7. bout_space.py

## Parameters

### PCA
- _n_components_ : Number of principal components to keep.

### Find bouts
- _winsize_ : Scaling parameter for the gaussian window used to filter tail angles.
- _bandwidth_ : Bandwidth of the kernel used for the kernel density estimation used to set the movement threshold.
- _minsize_ : Minimum bout duration.

### Bout filtering
- _dim_2_thresh_ : Max absolute limit of the 2nd principal component, used to exclude tracking errors.
- _n_gauss_mix_ : Number of mixture components to use in the gaussian mixture model to exclude non bouts.

### Interpolation
- _gap_size_ : Maximum number of consecutive frames over which to interpolate missing data.

### Bout alignment
- _alignment_percentile_ : Percentage of bouts whose beginning and end should fall within the alignment window.
- _n_align_clusters_ : Number of clusters for left/right alignment.

# Eye tracking

1. convert_tracking.py
2. gaussian_mixture.py
3. predict_convergence.py
4. find_events.py

## Parameters

- _min_angle_ : Minimum allowed divergence angle for a single eye (degrees).
- _max_angle_ : Maximum allowed convergence angle for a single eye (degrees).
- _confidence_threshold_ : Minimum allowed confidence for point scores to be included in analysis.
- _convergence_ksize_ : Kernel size (seconds) for filtering eye convergence traces.
- _event_window_ : Time window considered when finding prey capture sequences for each fish (seconds).
- _convergence_threshold_ : Threshold that must be exceeded for an event to be considered prey capture.

# Description of analysis

## Tail tracking
The tail is tracked using a 12-point sleap model.

### Dimensionality reduction
Dimensionality reduction of tail pose using PCA.

Code: _tail_movements/pca.py_

Directory: _{directory}/analysis/postural_decomposition_

Files:
  - _{species}.h5_

### Segmentation
Tail traces segmented into individual bouts.

Code: _tail_movements/find_bouts.py_

Directory: _{directory}/analysis/bouts/{species}_

Files:
  - _bouts.csv_
  - _kde.npy_
  - _params.json_

### Filtering
Filter out bad bouts using a GMM.

Code: _tail_movements/clean_bouts.py_

Directory: _{directory}/analysis/bouts/{species}_

Files:
  - _bouts_cleaned.csv_
  - _clean_idxs.npy_
  - _bouts_stds.npy_

### Alignment
Align bouts to peak.

Code: _tail_movements/align_bouts.py_

Directory: _{directory}/analysis/bouts/{species}_

Files:
  - _aligned_bouts.npy_

### Flip bouts
Flip bouts so maximum tail angle always in same direction.

Code: _tail_movements/flip_bouts.py_

Directory: _{directory}/analysis/bouts/{species}_

Files:
  - _flipped_bouts.npy_

### Clustering
Cluster bouts.

Code: _tail_movements/cluster_bouts.py_

Directory: _{directory}/analysis/bouts/{species}_

Files:
  - _bouts_clustered.csv_
  - _cluster_params.json_
  - _cluster_stats.npy_
  - _exemplar_indices.npy_

### Embedding
Create non-linear embedding of bouts.

Code: _tail_movements/bout_space.py_

Directory: _{directory}/analysis/bouts/{species}_

Files:
  - _bout_space.npy_

### Prey capture
Score bouts as prey capture or spontaneous (except medaka, _O. latipes_).

Code: _tail_movements/prey_capture_bouts.py_

Directory: _{directory}/analysis/bouts/{species}_

Files:
  - _bouts_scored.csv_

## Eye tracking
The eyes are tracked using a 7-point sleap model.

### Kinematics
Extract angles from tracking data.

Code: _eye_movements/convert_tracking.py_

Directory: _{directory}/analysis/eye_analysis/{species}_

Files:
  - _/angles/{trial_id}.csv_
  - _convergence.npy_

### Convergence states
Detect eye convergence using GMM.

Code: _eye_movements/gaussian_mixture.py_

Directory: _{directory}/analysis/eye_analysis/{species}_

Files:
  - _gmm_params.json_

### Convergence time series
Detect continuous periods of eye convergence.

Code: _eye_movements/predict_convergence.py_

Directory: _{directory}/analysis/eye_analysis/{species}_

Files:
  - _/predict/{trial_id}.npy_

### Hunting episodes
Identify hunting episodes from convergence time series.

Code: _eye_movements/find_events.py_

Directory: _{directory}/analysis/eye_analysis/{species}_

Files:
  - _events.csv_
  - _fish_medians.npy_
  - _durations.npy_
  - _proportions.npy_
  - _rates.npy_

## Artemia tracking
Artemia are tracked using YOLO.

### Tracking files
  - _run_yolo.py_
  - _run_yolo_frames.py_
  - _yolo_on_frames.py_
  - _prey_over_time.py_

### Hunting rate
Count paramecia over time.

Code: _artemia/artemia_counts.py_

Directory: _{directory}/analysis/artemia/{species}_

Files:
  - _artemia_counts.npy_

### Trajectory analysis

  - assign_tracks.py -> sequences
  - prey_ego.py -> egocentric
  - order_tracks.py -> events_data.h5
