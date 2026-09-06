import time
from decay import simulate, simulate_loop

# Parameters for a large simulation run
N0 = 200000
lam = 0.4

# 1. Time the pure-Python loop version
t0 = time.perf_counter()
simulate_loop(N0, lam)
t1 = time.perf_counter()
time_loop = t1 - t0

# 2. Time the vectorized NumPy version
t0 = time.perf_counter()
simulate(N0, lam)
t1 = time.perf_counter()
time_numpy = t1 - t0

# 3. Calculate speedup factor
speedup = time_loop / time_numpy

# 4. Print results
print(f"Pure Python loop time: {time_loop:.4f} seconds")
print(f"NumPy vectorized time: {time_numpy:.4f} seconds")
print(f"NumPy is {speedup:.2f}x faster than pure Python")