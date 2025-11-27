import argparse
import time
import ray
import torch
import psutil
import os
import gc
import threading
from ray_zerocopy import (
    rewrite_pipeline,
    ZeroCopyModel,
)


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


def monitor_memory(stop_event, interval=1.0):
    """Monitor memory usage of all python processes related to this script."""
    current_pid = os.getpid()
    print(f"\nStarting memory monitor (Parent PID: {current_pid})...")

    while not stop_event.is_set():
        try:
            ray_workers_rss = []
            ray_workers_uss = []

            # Iterate over all processes to find Ray workers
            for proc in psutil.process_iter(
                ["pid", "name", "cmdline", "memory_info", "memory_full_info"]
            ):
                try:
                    name = proc.info["name"] or ""
                    cmdline = proc.info["cmdline"] or []
                    cmdline_str = " ".join(cmdline)

                    is_worker = (
                        "ray::worker_task" in name or "ray::worker_task" in cmdline_str
                    )

                    if is_worker:
                        # RSS includes shared memory
                        rss = proc.info["memory_info"].rss / 1024 / 1024
                        ray_workers_rss.append(rss)

                        # USS is unique private memory (excludes shared)
                        # Note: memory_full_info can be slower/require privileges
                        try:
                            uss = proc.memory_full_info().uss / 1024 / 1024
                            ray_workers_uss.append(uss)
                        except:
                            pass

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Calculate totals
            total_rss = sum(ray_workers_rss)
            total_uss = sum(ray_workers_uss) if ray_workers_uss else 0

            # Format output
            timestamp = time.strftime("%H:%M:%S")
            workers_info = f"Workers: {len(ray_workers_rss)}"
            if ray_workers_rss:
                workers_info += (
                    f" | Total RSS: {total_rss:.1f} MB | Total USS: {total_uss:.1f} MB"
                )

            print(
                f"[{timestamp}] Parent: {get_memory_mb(current_pid):.1f} MB | {workers_info}"
            )

        except Exception as e:
            pass

        time.sleep(interval)


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
def worker_task_zerocopy(model_ref, sleep_time):
    """Worker task that loads and uses the model with zero-copy."""
    pid = os.getpid()
    print(f"Worker (ZeroCopy) {pid} started. Mapping memory...")

    # Ensure we force garbage collection BEFORE measuring or doing anything else
    gc.collect()

    model = ZeroCopyModel.from_object_ref(model_ref)

    with torch.no_grad():
        _ = model(torch.randn(1, 5000))

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
        default=60,
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

    print(f"Creating large model (~500MB)...")
    model = create_large_model()

    print(f"Starting benchmark with {args.workers} workers for {args.duration} seconds")
    print(f"Mode: {'ZERO-COPY' if args.zerocopy else 'NORMAL (Copy)'}")
    print("-" * 50)

    # Start memory monitor thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=monitor_memory, args=(stop_monitor,))
    monitor_thread.daemon = True
    monitor_thread.start()

    start_time = time.time()

    if args.zerocopy:
        # Zero-copy path
        class Pipeline:
            def __init__(self, model):
                self.model = model

        pipeline = Pipeline(model)
        rewritten = ZeroCopyModel.rewrite(pipeline)
        model_ref = ZeroCopyModel.to_object_ref(rewritten.model)  # type: ignore

        futures = [
            worker_task_zerocopy.remote(model_ref, args.duration)
            for _ in range(args.workers)
        ]
    else:
        # Normal path
        model_ref = ray.put(model)
        futures = [
            worker_task_normal.remote(model_ref, args.duration)
            for _ in range(args.workers)
        ]

    # Wait for all tasks to complete
    ray.get(futures)

    # Stop monitor
    stop_monitor.set()
    monitor_thread.join()

    end_time = time.time()
    print("-" * 50)
    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
