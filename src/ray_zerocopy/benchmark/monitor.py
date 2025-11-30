"""Memory monitoring utilities for benchmarking Ray workers."""

import os
import threading
import time
from contextlib import contextmanager

import psutil
import torch


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in MB."""
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    total_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (total_params + total_buffers) / 1024 / 1024


USS_UNAVAILABLE_MESSAGE = """
Note: USS (Unique Set Size) cross-process monitoring is not available on macOS
due to system security restrictions. Individual workers can still measure
their own USS, which is reported in the worker startup messages above.
""".strip()


def get_memory_mb(pid=None):
    if pid is None:
        pid = os.getpid()
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / 1024 / 1024
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def monitor_memory(stop_event, stats, interval=1.0, show_parent=False):
    """Monitor memory usage of Ray workers and track stats.

    Args:
        stop_event: Threading event to signal when to stop monitoring
        stats: Dictionary to track peak memory statistics
        interval: Monitoring interval in seconds
        show_parent: Whether to show parent process memory in output
    """
    current_pid = os.getpid()
    print(f"\nMemory Monitor Started (Parent PID: {current_pid})...")
    print("=" * 80)

    while not stop_event.is_set():
        try:
            ray_workers_rss = []
            ray_workers_uss = []
            parent_rss = 0

            # Get parent process memory if requested
            if show_parent:
                try:
                    parent_process = psutil.Process(current_pid)
                    parent_rss = parent_process.memory_info().rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
                try:
                    name = proc.info["name"] or ""
                    cmdline = proc.info["cmdline"] or []
                    cmdline_str = " ".join(cmdline)

                    # Detect Ray workers - look for various patterns
                    is_worker = (
                        "ray::worker" in cmdline_str
                        or "ray::worker_task" in name
                        or "ray_worker" in name
                        or "raylet" in cmdline_str
                        or ("ray" in cmdline_str and "worker" in cmdline_str)
                        or "MapWorker" in cmdline_str
                    )

                    if is_worker:
                        rss = proc.info["memory_info"].rss / 1024 / 1024
                        ray_workers_rss.append(rss)

                        try:
                            uss = proc.memory_full_info().uss / 1024 / 1024
                            ray_workers_uss.append(uss)
                            # Track USS availability
                            if "uss_available" not in stats:
                                stats["uss_available"] = True
                        except (AttributeError, psutil.AccessDenied):
                            if "uss_available" not in stats:
                                stats["uss_available"] = False

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            total_rss = sum(ray_workers_rss)
            total_uss = sum(ray_workers_uss) if ray_workers_uss else 0

            # Track peak values
            if total_rss > stats["peak_total_rss"]:
                stats["peak_total_rss"] = total_rss
            if total_uss > stats["peak_total_uss"]:
                stats["peak_total_uss"] = total_uss
            if ray_workers_rss:
                avg_rss = total_rss / len(ray_workers_rss)
                if avg_rss > stats["peak_avg_rss"]:
                    stats["peak_avg_rss"] = avg_rss

            timestamp = time.strftime("%H:%M:%S")

            # Format output
            if show_parent:
                output = f"[{timestamp}] Parent: {parent_rss:7.1f} MB | "
            else:
                output = f"[{timestamp}] "

            if ray_workers_rss:
                avg_rss = total_rss / len(ray_workers_rss)
                avg_uss = total_uss / len(ray_workers_uss) if ray_workers_uss else 0
                uss_str = f"{total_uss:7.1f}" if ray_workers_uss else "    N/A"
                output += (
                    f"Workers: {len(ray_workers_rss):2d} | "
                    f"Total RSS: {total_rss:7.1f} MB | Avg RSS: {avg_rss:6.1f} MB | "
                    f"Total USS: {uss_str} MB"
                )
                if ray_workers_uss:
                    output += f" | Avg USS: {avg_uss:6.1f} MB"
            else:
                output += "Workers: 0"

            print(output)

        except Exception:
            pass

        time.sleep(interval)


@contextmanager
def monitor_memory_context(interval=1.0, show_parent=False):
    """Context manager for memory monitoring.

    Args:
        interval: Monitoring interval in seconds
        show_parent: Whether to show parent process memory in output

    Yields:
        Dictionary containing memory statistics (peak values and USS availability)
    """
    # Start memory monitor with stats tracking
    memory_stats = {
        "peak_total_rss": 0.0,
        "peak_total_uss": 0.0,
        "peak_avg_rss": 0.0,
    }
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_memory, args=(stop_monitor, memory_stats, interval, show_parent)
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        yield memory_stats
    finally:
        stop_monitor.set()
        monitor_thread.join()

        # Print USS availability note if needed
        if not memory_stats.get("uss_available", False):
            print(f"\n{USS_UNAVAILABLE_MESSAGE}")
