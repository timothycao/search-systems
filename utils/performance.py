"""
Performance tracker for search systems.
Use track='time' or 'memory' (default=None for no tracking).
"""

import time
import tracemalloc

def track_performance(func, *args, track: str | None = None, **kwargs):
    """Track runtime or peak memory usage for any callable."""
    if track is None: return func(*args, **kwargs)

    if track == "time":
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"[Performance] Time={end_time - start_time:.3f}s")
        return result

    if track == "memory":
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"[Performance] Peak Memory={peak / (1024 ** 2):.2f}MB")
        return result