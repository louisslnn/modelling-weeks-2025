"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import lru_cache
from optimize_azimuthal_bins import find_optimal_scan_bins

# Constants
angle_min = -45
angle_max = 45
frequency = 1.2e9  # 1.2 GHz
c = 3e8
window_type = "uniform"  # Could be 'uniform','chebyshev','hamming'
ripple_db = 30.0  # Used if Chebyshev
angle_step = 0.5  # Increased step size to reduce computation

# LRU cache to avoid redundant calculations
@lru_cache(maxsize=None)
def cached_find_optimal_scan_bins(N):
    return len(find_optimal_scan_bins(
        angle_min, angle_max, N, frequency, c=c,
        window_type=window_type, ripple_db=ripple_db, angle_step=angle_step
    ))

# Parallelized computation using multiprocessing
def compute_bin_count(N):
    return cached_find_optimal_scan_bins(N)

if __name__ == "__main__":
    num_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 CPU cores
    with multiprocessing.Pool(processes=num_workers) as pool:
        arr = pool.map(compute_bin_count, range(1, 41))

    print(arr)
"""

import numpy as np
import matplotlib.pyplot as plt
from optimize_azimuthal_bins import *

angle_min = -45
angle_max = 45
frequency = 1.2e9  # 1.2 GHz
c = 3e8
window_type = "uniform"  # Could be 'uniform','chebyshev','hamming'
ripple_db = 30.0  # Used if Chebyshev
angle_step = 0.1

arr = []

for N in range(25, 41):
        arr.append(len(find_optimal_scan_bins(
    angle_min, angle_max,
    N, frequency, c=c,
    window_type=window_type,
    ripple_db=ripple_db,
    angle_step=angle_step))
)

print(arr)