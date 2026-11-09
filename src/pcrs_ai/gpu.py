"""Wait for free GPU memory before starting a job (the GPU is shared with other jobs)."""
from __future__ import annotations

import time

import torch


def wait_for_memory(min_free_gb: float, poll_s: int = 60, max_wait_h: float = 8.0) -> float:
    """Block until at least `min_free_gb` GB are free on the current device. Returns the waited time in seconds."""
    t0 = time.time()
    while True:
        free, _ = torch.cuda.mem_get_info()
        if free / 1e9 >= min_free_gb:
            return time.time() - t0
        if time.time() - t0 > max_wait_h * 3600:
            raise RuntimeError(f"GPU memory did not free up: {free / 1e9:.1f} GB free < {min_free_gb} GB")
        print(f"[gpu] {free / 1e9:.1f} GB free, waiting for {min_free_gb} GB", flush=True)
        time.sleep(poll_s)
