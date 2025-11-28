import os
import threading
import time
from contextlib import contextmanager

import psutil


def monitor_memory(stop_event, stats, interval=1.0):
    """Monitor memory usage of Ray workers and track stats."""
    current_pid = os.getpid()
    print(f"\nMemory Monitor Started (Parent PID: {current_pid})...")
    print("=" * 80)

    while not stop_event.is_set():
        try:
            ray_workers_rss = []
            ray_workers_uss = []

            for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
                try:
                    name = proc.info["name"] or ""
                    cmdline = proc.info["cmdline"] or []
                    cmdline_str = " ".join(cmdline)

                    # Detect Ray workers - look for various patterns
                    is_worker = (
                        "ray::worker" in cmdline_str
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
                        except:
                            pass

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
            if ray_workers_rss:
                avg_rss = total_rss / len(ray_workers_rss)
                avg_uss = total_uss / len(ray_workers_uss) if ray_workers_uss else 0
                print(
                    f"[{timestamp}] Workers: {len(ray_workers_rss):2d} | "
                    f"Total RSS: {total_rss:7.1f} MB | Avg RSS: {avg_rss:6.1f} MB | "
                    f"Total USS: {total_uss:7.1f} MB | Avg USS: {avg_uss:6.1f} MB"
                )

        except Exception:
            pass

        time.sleep(interval)


@contextmanager
def monitor_memory_context():
    # Start memory monitor with stats tracking
    memory_stats = {
        "peak_total_rss": 0.0,
        "peak_total_uss": 0.0,
        "peak_avg_rss": 0.0,
    }
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_memory, args=(stop_monitor, memory_stats)
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        yield memory_stats
    finally:
        stop_monitor.set()
        monitor_thread.join()
