import argparse
import gc
import os
import time

import psutil
import ray
import torch

from ray_zerocopy import TaskWrapper
from ray_zerocopy.benchmark import monitor_memory_context


def create_large_model():
    """Create a model large enough to see memory differences (~500MB)."""
    return torch.nn.Sequential(
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 100),
    )


def get_memory_mb(pid=None):
    if pid is None:
        pid = os.getpid()
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / 1024 / 1024
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


@ray.remote
def worker_task_normal(model_copy, sleep_time):
    """Worker task that loads model normally (creates a copy)."""
    pid = os.getpid()
    print(f"Worker (Normal) {pid} started. Allocating memory...")

    # Access model to ensure it's loaded
    with torch.no_grad():
        _ = model_copy(torch.randn(1, 5000))

    # Force GC to stabilize memory reading
    gc.collect()
    process = psutil.Process(pid)
    try:
        mem_uss = process.memory_full_info().uss / 1024 / 1024
        mem_rss = process.memory_info().rss / 1024 / 1024
        print(
            f"Worker (Normal) {pid} ready. USS: {mem_uss:.1f} MB, RSS: {mem_rss:.1f} MB. Sleeping {sleep_time}s..."
        )
    except:
        mem = process.memory_info().rss / 1024 / 1024
        print(
            f"Worker (Normal) {pid} ready. Mem: {mem:.1f} MB. Sleeping {sleep_time}s..."
        )

    time.sleep(sleep_time)
    return 0


@ray.remote
def worker_task_zerocopy(wrapped_pipeline, sleep_time):
    """Worker task that uses the zero-copy wrapped pipeline."""
    pid = os.getpid()
    print(f"Worker (ZeroCopy) {pid} started. Mapping memory...")

    # Ensure we force garbage collection BEFORE measuring or doing anything else
    gc.collect()

    # Use the wrapped pipeline - model loading happens automatically with zero-copy
    with torch.no_grad():
        _ = wrapped_pipeline(torch.randn(1, 5000))

    # Force GC to stabilize memory reading
    gc.collect()

    process = psutil.Process(pid)
    try:
        mem_uss = process.memory_full_info().uss / 1024 / 1024
        mem_rss = process.memory_info().rss / 1024 / 1024
        print(
            f"Worker (ZeroCopy) {pid} ready. USS: {mem_uss:.1f} MB, RSS: {mem_rss:.1f} MB. Sleeping {sleep_time}s..."
        )
    except:
        mem = process.memory_info().rss / 1024 / 1024
        print(
            f"Worker (ZeroCopy) {pid} ready. Mem: {mem:.1f} MB. Sleeping {sleep_time}s..."
        )

    time.sleep(sleep_time)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Memory usage benchmark for Ray ZeroCopy"
    )
    parser.add_argument(
        "--zerocopy", action="store_true", help="Enable zero-copy optimization"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration to keep workers alive (seconds)",
    )
    args = parser.parse_args()

    # Initialize Ray
    if not ray.is_initialized():
        # Try to connect to existing cluster first
        try:
            ray.init(ignore_reinit_error=True)
        except:
            # Fallback to local cluster with large object store
            ray.init(object_store_memory=4 * 1024 * 1024 * 1024)

    print("Creating large model (~500MB)...")
    model = create_large_model()

    print(f"Starting benchmark with {args.workers} workers for {args.duration} seconds")
    print(f"Mode: {'ZERO-COPY' if args.zerocopy else 'NORMAL (Copy)'}")
    print("-" * 50)

    # Use memory monitor context manager
    with monitor_memory_context(interval=1.0, show_parent=True) as memory_stats:
        start_time = time.time()

        if args.zerocopy:
            # Zero-copy path using TaskWrapper
            class Pipeline:
                def __init__(self, model):
                    self.model = model

                def __call__(self, x):
                    return self.model(x)

            pipeline = Pipeline(model)
            wrapped = TaskWrapper(pipeline)

            futures = [
                worker_task_zerocopy.remote(wrapped, args.duration)
                for _ in range(args.workers)
            ]
        else:
            # Normal path
            model_ref = ray.put(model)
            futures = [
                worker_task_normal.remote(model_ref, args.duration)
                for _ in range(args.workers)
            ]

        ray.get(futures)

        end_time = time.time()
        print("-" * 50)
        print(f"Benchmark completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
