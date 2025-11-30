"""
Comprehensive memory benchmark comparing three approaches:
1. Normal Actor - Baseline with full model copies per actor
2. Task-based - Using ModelWrapper.for_tasks()
3. Actor-based - Using ModelWrapper.from_model(..., mode="actor")

Supports both Ray Core and Ray Data execution modes.

Usage:
    # Ray Core mode
    python benchmark_comprehensive.py --mode ray_core --workers 4 --batches 20

    # Ray Data mode
    python benchmark_comprehensive.py --mode ray_data --workers 4 --batches 20

Note:
    This benchmark script is under development. In particular, the memory
    measurements are not yet fully accurate. There are several reasons:
    - Currently, the script only measures the memory usage at a specific
      time interval. This causes inaccuracies for task-based execution,
      which can be very short-live.
    - For task-based execution, each call spawns a new Ray task. As a result,
      the memory measurement at each top-level Ray task does not include the
      nested Ray task calls.
"""

import argparse
import gc
import json
import os
import time
from typing import Any, Dict

import numpy as np
import psutil
import ray
import torch
from ray.data import ActorPoolStrategy, TaskPoolStrategy

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from ray_zerocopy import ModelWrapper
from ray_zerocopy.benchmark import (
    create_large_model,
    estimate_model_size_mb,
    get_memory_mb,
    log_memory,
    monitor_memory_context,
)

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

# ============================================================================
# Ray Core Mode Implementations
# ============================================================================


@ray.remote
def normal_actor_worker(model_copy, batches, batch_size):
    """Normal actor worker that loads model (Ray automatically deserializes ObjectRef)."""
    pid = os.getpid()
    print(f"[Normal Actor] Worker {pid} started. Loading model...")

    # Ray automatically deserializes ObjectRefs passed as arguments
    # So model_copy is already the deserialized model (creates a copy per worker)
    model = model_copy

    # Run inference on batches
    with torch.no_grad():
        for _ in range(batches):
            inputs = torch.randn(batch_size, 5000)
            _ = model(inputs)

    # Measure memory
    gc.collect()
    rss_mb = get_memory_mb(pid)
    try:
        process = psutil.Process(pid)
        uss_mb = process.memory_full_info().uss / 1024 / 1024
    except:
        uss_mb = 0

    print(
        f"[Normal Actor] Worker {pid} ready. RSS: {rss_mb:.1f} MB, USS: {uss_mb:.1f} MB"
    )

    return {"pid": pid, "rss_mb": rss_mb, "uss_mb": uss_mb}


@ray.remote
def task_based_worker(wrapped_pipeline, batches, batch_size):
    """Task-based worker using ModelWrapper.for_tasks()."""
    pid = os.getpid()
    print(f"[Task-based] Worker {pid} started. Using zero-copy pipeline...")

    # Run inference using wrapped pipeline (zero-copy)
    with torch.no_grad():
        for _ in range(batches):
            inputs = torch.randn(batch_size, 5000)
            _ = wrapped_pipeline(inputs)

    # Measure memory
    gc.collect()
    rss_mb = get_memory_mb(pid)
    try:
        process = psutil.Process(pid)
        uss_mb = process.memory_full_info().uss / 1024 / 1024
    except:
        uss_mb = 0

    print(
        f"[Task-based] Worker {pid} ready. RSS: {rss_mb:.1f} MB, USS: {uss_mb:.1f} MB"
    )

    return {"pid": pid, "rss_mb": rss_mb, "uss_mb": uss_mb}


@ray.remote
class ActorBasedWorker:
    """Actor-based worker using ModelWrapper.from_model(..., mode="actor")."""

    def __init__(self, model_wrapper: ModelWrapper):
        self.pid = os.getpid()
        with log_memory(f"Actor-based Worker {self.pid}") as get_mem:
            # Load model using zero-copy
            self.pipeline = model_wrapper.load()

        # Get memory from context manager
        mem = get_mem()
        self.rss_mb = mem["rss"]
        self.uss_mb = mem["uss"]

    def process_batches(self, batches: int, batch_size: int):
        """Process batches of data."""
        with torch.no_grad():
            for _ in range(batches):
                inputs = torch.randn(batch_size, 5000)
                _ = self.pipeline(inputs)

        return {"pid": self.pid, "rss_mb": self.rss_mb, "uss_mb": self.uss_mb}


# ============================================================================
# Ray Data Mode Implementations
# ============================================================================


class NormalActorRayData:
    """Normal actor for Ray Data that loads model via ray.get (full copy)."""

    def __init__(self, model_ref):
        self.pid = os.getpid()
        with log_memory(f"Normal Actor {self.pid}") as get_mem:
            # Load model (creates full copy)
            self.model = ray.get(model_ref)

        # Get memory from context manager (measured after gc.collect())
        mem = get_mem()
        self.rss_mb = mem["rss"]
        self.uss_mb = mem["uss"]

    def __call__(self, batch):
        """Process a batch of data."""
        batch_size = (
            int(batch["size"][0])
            if hasattr(batch["size"], "__len__")
            else int(batch["size"])
        )

        with torch.no_grad():
            inputs = torch.randn(batch_size, 5000)
            outputs = self.model(inputs)

        return {
            "result": [outputs.shape[0]],
            "memory_rss_mb": [self.rss_mb],
            "memory_uss_mb": [self.uss_mb],
            "actor_pid": [self.pid],
        }


def task_based_function(batch, wrapped_pipeline):
    """Task-based function for Ray Data using ModelWrapper.for_tasks()."""

    # wrapped_pipeline handles zero-copy loading of the model
    # so we do not need to call ray.get on it directly.

    with log_memory(f"Task-based function {os.getpid()}") as get_mem:
        batch_size = int(batch["size"])

        with torch.no_grad():
            inputs = torch.randn(batch_size, 5000)
            outputs = wrapped_pipeline(inputs)

        mem = get_mem()
        rss_mb = mem["rss"]
        uss_mb = mem["uss"]

    return {
        "result": outputs.shape[0],
        "memory_rss_mb": rss_mb,
        "memory_uss_mb": uss_mb,
        "actor_pid": os.getpid(),
    }


class ActorBasedRayData:
    """Actor-based worker for Ray Data using ModelWrapper.from_model(..., mode="actor")."""

    def __init__(self, model_wrapper: ModelWrapper):
        self.pid = os.getpid()
        with log_memory(f"Actor-based Worker {self.pid}") as get_mem:
            # Load model using zero-copy
            self.pipeline = model_wrapper.load()

        # Get memory from context manager
        mem = get_mem()
        self.rss_mb = mem["rss"]
        self.uss_mb = mem["uss"]

    def __call__(self, batch):
        """Process a batch of data."""
        batch_size = (
            int(batch["size"][0])
            if hasattr(batch["size"], "__len__")
            else int(batch["size"])
        )

        with torch.no_grad():
            inputs = torch.randn(batch_size, 5000)
            outputs = self.pipeline(inputs)

        return {
            "result": [outputs.shape[0]],
            "memory_rss_mb": [self.rss_mb],
            "memory_uss_mb": [self.uss_mb],
            "actor_pid": [self.pid],
        }


# ============================================================================
# Helper Functions
# ============================================================================


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


# ============================================================================
# Benchmark Execution Functions
# ============================================================================


def run_ray_core_normal(model, workers, batches, batch_size):
    """Run normal actor approach with Ray Core."""
    print("\n" + "=" * 80)
    print("RAY CORE - NORMAL ACTOR (Baseline)")
    print("=" * 80)

    # Put model in object store - Ray will automatically deserialize it in workers
    # This still creates a copy per worker (baseline behavior)
    model_ref = ray.put(model)

    with monitor_memory_context(interval=1.0) as memory_stats:
        start_time = time.time()

        # Pass model_ref - Ray will deserialize it in each worker (creating copies)
        futures = [
            normal_actor_worker.remote(model_ref, batches, batch_size)
            for _ in range(workers)
        ]
        results = ray.get(futures)

        end_time = time.time()
        runtime = end_time - start_time

    return format_ray_core_results(
        results, memory_stats, runtime, workers, batches, batch_size
    )


def run_ray_core_task_based(model, workers, batches, batch_size):
    """Run task-based approach with Ray Core."""
    print("\n" + "=" * 80)
    print("RAY CORE - TASK-BASED (ModelWrapper.for_tasks())")
    print("=" * 80)

    # Create pipeline wrapper
    class Pipeline:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    pipeline = Pipeline(model)
    wrapped_pipeline = ModelWrapper.for_tasks(pipeline)

    with monitor_memory_context(interval=1.0) as memory_stats:
        start_time = time.time()

        futures = [
            task_based_worker.remote(wrapped_pipeline, batches, batch_size)
            for _ in range(workers)
        ]
        results = ray.get(futures)

        end_time = time.time()
        runtime = end_time - start_time

    return format_ray_core_results(
        results, memory_stats, runtime, workers, batches, batch_size
    )


def run_ray_core_actor_based(model, workers, batches, batch_size):
    """Run actor-based approach with Ray Core."""
    print("\n" + "=" * 80)
    print("RAY CORE - ACTOR-BASED (ModelWrapper.from_model(..., mode='actor'))")
    print("=" * 80)

    # Create pipeline wrapper
    class Pipeline:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    pipeline = Pipeline(model)
    model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

    with monitor_memory_context(interval=1.0) as memory_stats:
        start_time = time.time()

        # Create actors
        actors = [ActorBasedWorker.remote(model_wrapper) for _ in range(workers)]

        # Process batches
        futures = [
            actor.process_batches.remote(batches, batch_size) for actor in actors
        ]
        results = ray.get(futures)

        end_time = time.time()
        runtime = end_time - start_time

    return format_ray_core_results(
        results, memory_stats, runtime, workers, batches, batch_size
    )


def run_ray_data_normal(model, workers, batches, batch_size):
    """Run normal actor approach with Ray Data."""
    print("\n" + "=" * 80)
    print("RAY DATA - NORMAL ACTOR (Baseline)")
    print("=" * 80)

    model_ref = ray.put(model)

    # Create dataset
    ds = ray.data.range(batches)
    ds.set_name("normal_actor")
    ds = ds.map(lambda x: {"size": batch_size})

    # Disable progress bars
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False

    with monitor_memory_context(interval=1.0) as memory_stats:
        start_time = time.time()

        results = ds.map_batches(
            NormalActorRayData,
            fn_constructor_kwargs={"model_ref": model_ref},
            batch_size=1,
            compute=ActorPoolStrategy(size=workers),
            ray_remote_args_fn=lambda: {
                "max_restarts": 0,
            },
        )

        # Materialize results
        results_df = results.to_pandas()

        end_time = time.time()
        runtime = end_time - start_time

    # Aggregate memory stats from results
    mem_stats = aggregate_memory_stats_from_results(results_df, memory_stats, workers)

    return {
        "peak_total_rss_mb": mem_stats["total_rss"],
        "peak_total_uss_mb": mem_stats["total_uss"],  # Keep for backward compatibility
        "worker_uss_total_mb": mem_stats.get(
            "worker_uss_total", mem_stats["total_uss"]
        ),
        "monitor_peak_uss_mb": mem_stats.get(
            "monitor_peak_uss", mem_stats["total_uss"]
        ),
        "avg_rss_per_worker_mb": mem_stats["avg_rss"],
        "avg_uss_per_worker_mb": mem_stats["avg_uss"],
        "runtime_s": runtime,
        "throughput_rows_per_s": (batches * batch_size) / runtime
        if runtime > 0
        else 0,  # Total rows processed per second
        "num_workers": mem_stats["num_workers"],
    }


def run_ray_data_task_based(model, workers, batches, batch_size):
    """Run task-based approach with Ray Data."""
    print("\n" + "=" * 80)
    print("RAY DATA - TASK-BASED (ModelWrapper.for_tasks())")
    print("=" * 80)

    # Create pipeline wrapper
    class Pipeline:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    pipeline = Pipeline(model)
    wrapped_pipeline = ModelWrapper.for_tasks(pipeline)

    # Create dataset
    ds = ray.data.range(batches)
    ds.set_name("task_based")
    ds = ds.map(lambda x: {"size": batch_size})

    # Disable progress bars
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False

    with monitor_memory_context(interval=0.2) as memory_stats:
        start_time = time.time()

        # Use map_batches with a function (not actor) for task-based execution
        # The wrapped_pipeline will spawn Ray tasks for each inference call
        # Put wrapped pipeline in object store for serialization
        # wrapped_pipeline_ref = ray.put(wrapped_pipeline)
        results = ds.map(
            task_based_function,
            fn_kwargs={"wrapped_pipeline": wrapped_pipeline},
            # compute=TaskPoolStrategy(size=workers - 1),
        )

        # Materialize results
        results_df = results.to_pandas()

        end_time = time.time()
        runtime = end_time - start_time

    # Aggregate memory stats
    # For task-based execution, use monitor peak values since tasks are short-lived
    # and DataFrame aggregation would incorrectly sum memory from many different PIDs
    mem_stats = aggregate_memory_stats_from_results(
        results_df, memory_stats, workers, use_monitor_peak=True
    )

    return {
        "peak_total_rss_mb": mem_stats["total_rss"],
        "peak_total_uss_mb": mem_stats["total_uss"],  # Keep for backward compatibility
        "worker_uss_total_mb": mem_stats.get(
            "worker_uss_total", mem_stats["total_uss"]
        ),
        "monitor_peak_uss_mb": mem_stats.get(
            "monitor_peak_uss", mem_stats["total_uss"]
        ),
        "avg_rss_per_worker_mb": mem_stats["avg_rss"],
        "avg_uss_per_worker_mb": mem_stats["avg_uss"],
        "runtime_s": runtime,
        "throughput_rows_per_s": (batches * batch_size) / runtime
        if runtime > 0
        else 0,  # Total rows processed per second
        "num_workers": mem_stats["num_workers"],
    }


def run_ray_data_actor_based(model, workers, batches, batch_size):
    """Run actor-based approach with Ray Data."""
    print("\n" + "=" * 80)
    print("RAY DATA - ACTOR-BASED (ModelWrapper.from_model(..., mode='actor'))")
    print("=" * 80)

    # Create pipeline wrapper
    class Pipeline:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    pipeline = Pipeline(model)
    model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

    # Create dataset
    ds = ray.data.range(batches)
    ds.set_name("actor_based")
    ds = ds.map(lambda x: {"size": batch_size})

    # Disable progress bars
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False

    with monitor_memory_context(interval=1.0) as memory_stats:
        start_time = time.time()

        results = ds.map_batches(
            ActorBasedRayData,
            fn_constructor_kwargs={"model_wrapper": model_wrapper},
            batch_size=1,
            compute=ActorPoolStrategy(size=workers),
        )

        # Materialize results
        results_df = results.to_pandas()

        end_time = time.time()
        runtime = end_time - start_time

    # Aggregate memory stats
    # For actor-based, use monitor peak values for consistency with Ray Core
    # The monitor sums USS across all detected Ray worker processes
    mem_stats = aggregate_memory_stats_from_results(
        results_df, memory_stats, workers, use_monitor_peak=True
    )

    return {
        "peak_total_rss_mb": mem_stats["total_rss"],
        "peak_total_uss_mb": mem_stats["total_uss"],  # Keep for backward compatibility
        "worker_uss_total_mb": mem_stats.get(
            "worker_uss_total", mem_stats["total_uss"]
        ),
        "monitor_peak_uss_mb": mem_stats.get(
            "monitor_peak_uss", mem_stats["total_uss"]
        ),
        "avg_rss_per_worker_mb": mem_stats["avg_rss"],
        "avg_uss_per_worker_mb": mem_stats["avg_uss"],
        "runtime_s": runtime,
        "throughput_rows_per_s": (batches * batch_size) / runtime
        if runtime > 0
        else 0,  # Total rows processed per second
        "num_workers": mem_stats["num_workers"],
    }


# ============================================================================
# Results Output
# ============================================================================


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
):
    """Save results to JSON file."""
    output = {
        "execution_mode": execution_mode,
        "config": config,
        "approaches": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive memory benchmark for Ray ZeroCopy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["ray_core", "ray_data"],
        required=True,
        help="Execution mode: ray_core or ray_data",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--batches", type=int, default=20, help="Number of batches to process"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--ray-reset",
        default=False,
        help="Reset Ray cluster after each approach",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MEMORY BENCHMARK")
    print("=" * 80)
    print(f"Execution Mode: {args.mode.upper()}")
    print(f"Workers: {args.workers}")
    print(f"Batches: {args.batches}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 80)

    # Initialize Ray
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True)
        except Exception as e:
            print(
                f"Warning: Ray init failed with {e}, trying with explicit object store memory..."
            )
            try:
                ray.init(
                    object_store_memory=4 * 1024 * 1024 * 1024, ignore_reinit_error=True
                )
            except Exception as e2:
                print(f"Error: Failed to initialize Ray: {e2}")
                raise

    # Create model
    print("\nCreating large model (~500MB)...")
    model = create_large_model()
    model_size_mb = estimate_model_size_mb(model)
    print(f"Model size: {model_size_mb:.1f} MB")

    config = {
        "workers": args.workers,
        "batches": args.batches,
        "batch_size": args.batch_size,
    }

    results = {}

    try:
        if args.mode == "ray_core":
            # Run all three approaches with Ray Core
            approaches = {
                "normal": run_ray_core_normal,
                "task_based": run_ray_core_task_based,
                "actor_based": run_ray_core_actor_based,
            }

            for approach_name, run_func in approaches.items():
                results[approach_name] = run_func(
                    model, args.workers, args.batches, args.batch_size
                )
                print("\nWaiting 5 seconds after job completion...")
                time.sleep(5)

        elif args.mode == "ray_data":
            # Run all three approaches with Ray Data

            approaches = {
                "normal": run_ray_data_normal,
                "task_based": run_ray_data_task_based,
                "actor_based": run_ray_data_actor_based,
            }

            for approach_name, run_func in approaches.items():
                results[approach_name] = run_func(
                    model, args.workers, args.batches, args.batch_size
                )
                print("\nWaiting 5 seconds after job completion...")
                time.sleep(5)

                if args.ray_reset and ray.is_initialized():
                    ray.shutdown()

        # Print comparison table
        print_comparison_table(results, model_size_mb)

        # Save results to JSON
        save_results_json(results, config, args.mode, args.output)

        print("\n✅ Benchmark completed successfully!")

    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # Cleanup - only shutdown if we initialized Ray
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception as e:
            print(f"Warning: Error during Ray shutdown: {e}")


if __name__ == "__main__":
    main()
