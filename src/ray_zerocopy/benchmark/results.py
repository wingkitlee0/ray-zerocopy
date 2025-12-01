"""Result formatting and aggregation functions for benchmarks."""

import json
from typing import Any, Dict

import numpy as np

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def aggregate_memory_stats_from_results(
    results_df, memory_stats, workers, use_monitor_peak=False
):
    """Aggregate memory statistics from Ray Data results DataFrame.

    Args:
        results_df: Pandas DataFrame with memory stats per sample
        memory_stats: Dict with peak memory statistics from monitor
        workers: Number of workers (fallback if no results)
        use_monitor_peak: If True, use monitor peak values instead of aggregating from DataFrame.
                          Use this for task-based execution where tasks are short-lived.

    Returns:
        Dict with total_rss, total_uss, avg_rss, avg_uss, num_workers
    """
    # For task-based execution, use monitor peak values since tasks are short-lived
    # and DataFrame aggregation would sum memory from many different PIDs incorrectly
    if use_monitor_peak:
        total_rss = memory_stats["peak_total_rss"]
        total_uss = memory_stats["peak_total_uss"]
        avg_rss = memory_stats["peak_avg_rss"] if "peak_avg_rss" in memory_stats else 0
        # Estimate avg USS from total USS and number of workers seen
        if total_uss > 0 and workers > 0:
            avg_uss = total_uss / workers
        else:
            avg_uss = 0
        num_workers = workers
    elif not results_df.empty and "memory_rss_mb" in results_df.columns:
        grouped = results_df.groupby("actor_pid")
        stats = grouped.agg(
            max_rss_mb=("memory_rss_mb", "max"),
            max_uss_mb=("memory_uss_mb", "max"),
        )
        stats["samples"] = grouped.size()

        if not stats.empty:
            total_rss = stats["max_rss_mb"].sum()
            total_uss = stats["max_uss_mb"].sum()
            avg_rss = stats["max_rss_mb"].mean()
            avg_uss = stats["max_uss_mb"].mean()
            num_workers = len(stats)
        else:
            total_rss = total_uss = avg_rss = avg_uss = num_workers = 0
    else:
        total_rss = memory_stats["peak_total_rss"]
        total_uss = memory_stats["peak_total_uss"]
        avg_rss = memory_stats["peak_avg_rss"] if "peak_avg_rss" in memory_stats else 0
        avg_uss = 0
        num_workers = workers

    # For Ray Data:
    # - When use_monitor_peak=False: worker_uss_total = sum from DataFrame (actors)
    # - When use_monitor_peak=True: worker_uss_total = 0 (no per-worker data available)
    # monitor_peak_uss always uses the monitor's peak value
    worker_uss_total = total_uss if not use_monitor_peak else 0
    monitor_peak_uss = memory_stats["peak_total_uss"]

    return {
        "total_rss": total_rss,
        "total_uss": total_uss,
        "worker_uss_total": worker_uss_total,  # USS from worker/actor results
        "monitor_peak_uss": monitor_peak_uss,  # USS from monitor (includes all detected workers)
        "avg_rss": avg_rss,
        "avg_uss": avg_uss,
        "num_workers": num_workers,
    }


def format_ray_core_results(
    results, memory_stats, runtime, workers, batches, batch_size
):
    """Format results from Ray Core execution into standard benchmark result dict.

    Args:
        results: List of result dicts from workers with "rss_mb" and "uss_mb" keys
        memory_stats: Dict with peak memory statistics from monitor
        runtime: Runtime in seconds
        workers: Number of workers
        batches: Number of batches processed
        batch_size: Size of each batch

    Returns:
        Dict with benchmark results including both worker_uss_total and monitor_peak_uss
    """
    worker_rss = [r["rss_mb"] for r in results]
    worker_uss = [r["uss_mb"] for r in results]

    # Return both metrics:
    # - worker_uss_total: Sum of USS from main task/actor workers (measured in workers)
    # - monitor_peak_uss: Peak total USS from monitor (includes all detected Ray workers)
    worker_uss_total = sum(worker_uss) if worker_uss else 0
    monitor_peak_uss = memory_stats["peak_total_uss"]

    return {
        "peak_total_rss_mb": memory_stats["peak_total_rss"],
        "peak_total_uss_mb": monitor_peak_uss,  # Keep for backward compatibility
        "worker_uss_total_mb": worker_uss_total,  # USS from worker results
        "monitor_peak_uss_mb": monitor_peak_uss,  # USS from monitor (includes nested tasks)
        "avg_rss_per_worker_mb": np.mean(worker_rss) if worker_rss else 0,
        "avg_uss_per_worker_mb": np.mean(worker_uss) if worker_uss else 0,
        "runtime_s": runtime,
        "throughput_rows_per_s": (workers * batches * batch_size) / runtime
        if runtime > 0
        else 0,  # Total rows processed per second
        "num_workers": len(results),
    }


USS_MEASUREMENT_NOTES = """
USS Measurement Notes:
  USS-Worker: Sum of USS from main task/actor workers (measured in worker code).
             For task-based, this excludes nested inference tasks.
  USS-Monitor: Peak total USS from monitor (includes all detected Ray workers).
               For task-based, this includes nested inference tasks spawned by
               wrapped_pipeline().

  With zero-copy, model tensors are in shared memory (not counted in USS).
  Per-worker USS should be much lower than model size (~50-200 MB overhead).
""".strip()


def print_comparison_table(results: Dict[str, Dict[str, Any]], model_size_mb: float):
    """Print a comparison table of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    approaches = ["normal", "task_based", "actor_based"]
    approach_names = {
        "normal": "Normal Actor",
        "task_based": "Task-based",
        "actor_based": "Actor-based",
    }

    # Prepare table data - ensure all approaches are included
    table_data = []
    headers = [
        "Approach",
        "RSS (MB)",
        "USS-Worker (MB)",
        "USS-Monitor (MB)",
        "Runtime (s)",
        "Throughput (rows/sec)",
    ]

    for approach in approaches:
        if approach not in results:
            continue
        r = results[approach]
        name = approach_names[approach]

        # Get worker USS (if available) or use monitor USS as fallback
        worker_uss = r.get("worker_uss_total_mb", 0)
        if worker_uss == 0:
            worker_uss = r.get("peak_total_uss_mb", 0)  # Fallback to peak_total_uss
        monitor_uss = r.get("monitor_peak_uss_mb", r.get("peak_total_uss_mb", 0))

        rss = r.get("peak_total_rss_mb", 0)
        runtime = r.get("runtime_s", 0)
        throughput = r.get("throughput_rows_per_s", 0)

        # Convert to float - let errors propagate if data is invalid
        rss = float(rss) if rss is not None else 0.0
        worker_uss = (
            float(worker_uss) if worker_uss is not None and worker_uss > 0 else None
        )
        monitor_uss = float(monitor_uss) if monitor_uss is not None else 0.0
        runtime = float(runtime) if runtime is not None else 0.0
        throughput = float(throughput) if throughput is not None else 0.0

        table_data.append([name, rss, worker_uss, monitor_uss, runtime, throughput])

    # Print table using rich if available, otherwise fall back to manual formatting
    if HAS_RICH:
        console = Console()
        table = Table(show_header=True, header_style="bold", box=None)

        # Add columns
        table.add_column("Approach", justify="left", style="cyan", no_wrap=True)
        table.add_column("RSS (MB)", justify="right", style="magenta")
        table.add_column("USS-Worker (MB)", justify="right", style="yellow")
        table.add_column("USS-Monitor (MB)", justify="right", style="yellow")
        table.add_column("Runtime (s)", justify="right", style="green")
        table.add_column("Throughput (rows/sec)", justify="right", style="blue")

        # Add rows - all rows in table_data should be added
        for row in table_data:
            name, rss, worker_uss, monitor_uss, runtime, throughput = row
            worker_uss_str = (
                f"{worker_uss:.1f}"
                if worker_uss is not None and worker_uss > 0
                else "N/A"
            )
            table.add_row(
                str(name),
                f"{rss:.1f}",
                worker_uss_str,
                f"{monitor_uss:.1f}",
                f"{runtime:.2f}",
                f"{throughput:.2f}",
            )

        print()
        console.print(table)
    else:
        # Fallback to manual formatting with proper alignment
        col_widths = [20, 13, 18, 19, 13, 20]  # Widths for each column
        # Print header
        header_row = " | ".join(
            f"{h:<{w}}" if i == 0 else f"{h:>{w}}"
            for i, (h, w) in enumerate(zip(headers, col_widths))
        )
        print("\n" + header_row)
        # Calculate separator length: sum of widths + separators (3 chars each)
        separator_len = sum(col_widths) + (len(col_widths) - 1) * 3
        print("-" * separator_len)
        # Print data rows
        for row in table_data:
            name, rss, worker_uss, monitor_uss, runtime, throughput = row
            worker_uss_str = f"{worker_uss:.1f}" if worker_uss is not None else "N/A"
            data_row = " | ".join(
                [
                    f"{name:<{col_widths[0]}}",
                    f"{rss:>{col_widths[1]}.1f}",
                    f"{worker_uss_str:>{col_widths[2]}}",
                    f"{monitor_uss:>{col_widths[3]}.1f}",
                    f"{runtime:>{col_widths[4]}.2f}",
                    f"{throughput:>{col_widths[5]}.2f}",
                ]
            )
            print(data_row)

    # Calculate savings
    if "normal" in results and "actor_based" in results:
        normal_uss = results["normal"]["peak_total_uss_mb"]
        actor_uss = results["actor_based"]["peak_total_uss_mb"]
        if normal_uss > 0:
            savings_pct = (1 - actor_uss / normal_uss) * 100
            print("\n" + "-" * 80)
            print(f"Memory Savings (Actor-based vs Normal): {savings_pct:.1f}%")
            print(
                f"Expected savings: ~{(1 - 1 / results['normal']['num_workers']) * 100:.0f}%"
            )

        # Show per-worker metrics for better understanding
        print("\n" + "-" * 80)
        print("Per-Worker Memory Breakdown:")
        for approach in approaches:
            if approach in results:
                r = results[approach]
                name = approach_names[approach]
                num_workers = r.get("num_workers", 1)
                if num_workers > 0:
                    avg_uss = r.get("avg_uss_per_worker_mb", 0)
                    if avg_uss == 0 and "peak_total_uss_mb" in r:
                        # Fallback: calculate from total
                        avg_uss = r["peak_total_uss_mb"] / num_workers
                    print(f"  {name}: {avg_uss:.1f} MB USS per worker (avg)")

        print(f"\nModel size: {model_size_mb:.1f} MB")
        print(
            f"Expected normal mode memory: ~{results['normal']['num_workers'] * model_size_mb:.0f} MB"
        )
        print(f"Expected zero-copy memory: ~{model_size_mb:.0f} MB")
        print("\n" + "-" * 100)
        print(USS_MEASUREMENT_NOTES)

    print("=" * 80)


def save_results_json(
    results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    execution_mode: str,
    output_file: str,
    use_jit: bool = False,
):
    """Save results to JSON file."""
    output = {
        "execution_mode": execution_mode,
        "use_jit": use_jit,
        "config": config,
        "approaches": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")
