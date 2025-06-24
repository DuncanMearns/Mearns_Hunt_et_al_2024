import numpy as np
from scipy.signal.windows import gaussian
from scipy.interpolate import CubicSpline
from zigzeg import event_finder, find_runs
from zigzeg.events import find_events_threshold
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity


def find_events_peaks(x, thresh, distance=100):
    events = []
    peaks, props = find_peaks(x, height=thresh, prominence=thresh, distance=distance)
    bases = set(list(props["left_bases"]) + list(props["right_bases"]))
    bases = np.sort(list(bases))
    for start, end in zip(bases[:-1], bases[1:]):
        peak_idx, = np.where((peaks > start) & (peaks < end))
        if not len(peak_idx):
            continue
        if len(peak_idx) > 1:
            raise ValueError("Multiple peaks in between bases: this isn't supposed to be possible.")
        peak_idx = peak_idx[0]
        events.append((start, peaks[peak_idx], end))
    return events


def find_bouts(x, thresh, minsize=25):
    events = []
    above_thresh = find_events_threshold(x, thresh, minsize=minsize)
    for (first, last) in above_thresh:
        x_segment = x[first:last+1]
        peaks, props = find_peaks(x_segment, prominence=thresh)
        if len(peaks):
            bases = set(list(props["left_bases"]) + list(props["right_bases"]) + [0, len(x_segment)])
            bases = np.sort(list(bases))
            for left_base, right_base in zip(bases[:-1], bases[1:]):
                if right_base - left_base < minsize:
                    continue
                peak_idx, = np.where((peaks > left_base) & (peaks < right_base))
                if len(peak_idx) > 1:
                    raise ValueError("Multiple peaks in between bases: this isn't supposed to be possible.")
                try:
                    peak_idx = peak_idx[0]
                    peak = peaks[peak_idx]
                except IndexError:
                    peak = np.argmax(x_segment[left_base:right_base])
                events.append((first + left_base, first + peak, first + right_base))
        else:
            peak = np.argmax(x_segment)
            events.append((first, first + peak, last))
    return events


def compute_tail_curvature(points, headings):
    # Compute the vectors for each tail segment
    vs = np.empty(points.shape)
    vs[:, 1:] = np.diff(points, axis=1)
    vs[:, 0] = np.array([np.cos(headings + np.pi), np.sin(headings + np.pi)]).T
    # Tail segment lengths
    ls = np.linalg.norm(vs, axis=2)
    # Compute angles between successive tail segments from the arcsin of cross products
    crosses = np.cross(vs[:, :-1], vs[:, 1:])
    crosses /= (ls[:, :-1] * ls[:, 1:])
    dks = np.arcsin(crosses)
    # Cumulative sum angle differences between segments
    ks = np.cumsum(dks, axis=1)
    # Sum tail segments lengths
    tail_lengths = np.sum(ls[:, 1:], axis=1)
    return ks, tail_lengths


def tail_angle_filter(x, winsize=30):
    # Generate kernel
    kernel = gaussian(winsize * 2, (winsize * 2) / 5.)
    kernel /= np.sum(kernel)
    # Filter
    diffed = np.pad(np.diff(x), (1, 0), constant_values=0)
    mod_derivative = np.abs(diffed)
    filtered = np.convolve(mod_derivative, kernel, mode='same')
    return filtered


def find_threshold(*xs, bandwidth: float = 0.1):
    x = np.concatenate(xs, axis=0) if (len(xs) > 1) else xs[0]
    # Compute kde
    bins = np.linspace(0, x.max() / 2, int(1. / bandwidth))
    kde = KernelDensity(bandwidth=bandwidth).fit(x[:, np.newaxis])
    counts = kde.score_samples(bins[:, np.newaxis])
    counts = np.exp(counts)
    peaks, props = find_peaks(counts, height=counts.max() / 2., prominence=(None, None))
    if len(peaks) > 1:  # antimode
        bases = set(list(props["left_bases"]) + list(props["right_bases"]))
        bases = np.sort(list(bases))
        thresh_idx = np.where((bases > peaks[0]) & (bases < peaks[1]))
        thresh_idx, = bases[thresh_idx]
        thresh = bins[thresh_idx]
    else:  # first inflection after peak
        peak = peaks[0]
        inflections = np.diff(counts[:-1]) * np.diff(counts[1:])
        idx = inflections[peak:].argmax()
        thresh = bins[peak + idx + 1]
    return thresh, (bins, counts)


BoutDetector = event_finder(f_threshold=find_threshold, f_events=find_events_peaks, f_filter=tail_angle_filter)


def interpolate_frame_rate(x, fps_old, fps_new):
    if x.ndim > 2:
        raise ValueError("x.ndim must be <= 2")
    t_old = np.arange(len(x)) / fps_old
    t_last = t_old[-1]
    t_new = np.arange(0, t_last, 1. / fps_new)
    t_interpable = np.zeros(len(t_new), dtype="bool")
    tracked = np.all(~np.isnan(x), axis=1)
    interp_idxs, = np.where(tracked)
    interp_runs = find_runs(interp_idxs)
    for idxs in interp_runs:
        t0, t1 = t_old[[idxs[0], idxs[-1]]]
        interp = (t_new >= t0) & (t_new <= t1)
        t_interpable[interp] = True
    cs = CubicSpline(t_old[tracked], x[tracked, :])
    if x.ndim == 1:
        x_new = np.ones(len(t_new)) * np.nan
    else:
        x_new = np.ones((len(t_new), x.shape[1])) * np.nan
    x_new[t_interpable] = cs(t_new[t_interpable])
    return t_new, x_new
