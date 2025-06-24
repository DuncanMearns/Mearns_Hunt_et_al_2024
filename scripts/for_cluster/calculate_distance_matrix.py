import sys
import time
import numpy as np
from ethomap import pdist_dtw


def scaling_test(input_path):
    from matplotlib import pyplot as plt
    bouts = np.load(input_path)
    print(len(bouts))
    pdist_dtw(bouts[:10], n_processors=8, fs=1, bw=5)
    ns = [10, 50, 100, 150, 200, 300]
    ts = []
    for n in ns:
        t0 = time.time()
        pdist_dtw(bouts[:n], n_processors=8, fs=1, bw=5)
        t = time.time() - t0
        print(f"{n}: {t:.2f}")
        ts.append(t)
    plt.plot(ns, ts)
    plt.show()
    estimated_time = ts[-1] * ((len(bouts) / ns[-1]) ** 2)
    print("Estimated time:", estimated_time / 60., "minutes")


def main(input_path, output_path):
    t0 = time.time()
    bouts = np.load(input_path)
    _ = pdist_dtw(bouts[:100], n_processors=12, fs=1, bw=5)
    d = pdist_dtw(bouts, n_processors=12, fs=1, bw=5)
    np.save(output_path, d)
    print(time.time() - t0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path, output_path = sys.argv[1:3]
    else:
        input_path = r"D:\comparative_paper\bouts\aligned_bouts.npy"
        output_path = r"D:\comparative_paper\bouts\distances.npy"

    print(input_path, output_path)
    main(input_path, output_path)
    # scaling_test(input_path)
