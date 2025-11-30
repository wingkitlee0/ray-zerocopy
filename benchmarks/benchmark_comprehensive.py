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
from ray.data import ActorPoolStrategy

from ray_zerocopy import ModelWrapper
from ray_zerocopy.benchmark import (
    create_large_model,
    estimate_model_size_mb,
    get_memory_mb,
    monitor_memory_context,
)


# ============================================================================
# Ray Core Mode Implementations
# ============================================================================


@ray.remote
def normal_actor_worker(model_copy, batches, batch_size, sleep_time):
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

    print(f"[Normal Actor] Worker {pid} ready. RSS: {rss_mb:.1f} MB, USS: {uss_mb:.1f} MB")

    time.sleep(sleep_time)
    return {"pid": pid, "rss_mb": rss_mb, "uss_mb": uss_mb}


@ray.remote
def task_based_worker(wrapped_pipeline, batches, batch_size, sleep_time):
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

    print(f"[Task-based] Worker {pid} ready. RSS: {rss_mb:.1f} MB, USS: {uss_mb:.1f} MB")

    time.sleep(sleep_time)
    return {"pid": pid, "rss_mb": rss_mb, "uss_mb": uss_mb}


@ray.remote
class ActorBasedWorker:
    """Actor-based worker using ModelWrapper.from_model(..., mode="actor")."""

    def __init__(self, model_wrapper: ModelWrapper, sleep_time: float):
        self.pid = os.getpid()
        print(f"[Actor-based] Worker {self.pid} started. Loading model...")

        # Load model using zero-copy
        self.pipeline = model_wrapper.load()

        # Measure memory after loading
        gc.collect()
        self.rss_mb = get_memory_mb(self.pid)
        try:
            process = psutil.Process(self.pid)
            self.uss_mb = process.memory_full_info().uss / 1024 / 1024
        except:
            self.uss_mb = 0

        print(
            f"[Actor-based] Worker {self.pid} ready. RSS: {self.rss_mb:.1f} MB, USS: {self.uss_mb:.1f} MB"
        )

        time.sleep(sleep_time)

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
        print(f"[Normal Actor] Worker {self.pid} started. Loading model...")

        # Load model (creates full copy)
        self.model = ray.get(model_ref)

        # Measure memory
        gc.collect()
        self.rss_mb = get_memory_mb(self.pid)
        try:
            process = psutil.Process(self.pid)
            self.uss_mb = process.memory_full_info().uss / 1024 / 1024
        except:
            self.uss_mb = 0

        print(
            f"[Normal Actor] Worker {self.pid} ready. RSS: {self.rss_mb:.1f} MB, USS: {self.uss_mb:.1f} MB"
        )

    def __call__(self, batch):
        """Process a batch of data."""
        batch_size = int(batch["size"][0]) if hasattr(batch["size"], "__len__") else int(batch["size"])

        with torch.no_grad():
            inputs = torch.randn(batch_size, 5000)
            outputs = self.model(inputs)

        return {
            "result": [outputs.shape[0]],
            "memory_rss_mb": [self.rss_mb],
            "memory_uss_mb": [self.uss_mb],
            "actor_pid": [self.pid],
        }


def task_based_function(batch, wrapped_pipeline_ref):
    """Task-based function for Ray Data using ModelWrapper.for_tasks()."""
    # Get wrapped pipeline from object store
    wrapped_pipeline = ray.get(wrapped_pipeline_ref)
    
    batch_size = int(batch["size"][0]) if hasattr(batch["size"], "__len__") else int(batch["size"])

    with torch.no_grad():
        inputs = torch.randn(batch_size, 5000)
        outputs = wrapped_pipeline(inputs)

    pid = os.getpid()
    gc.collect()
    rss_mb = get_memory_mb(pid)
    try:
        process = psutil.Process(pid)
        uss_mb = process.memory_full_info().uss / 1024 / 1024
    except:
        uss_mb = 0

    return {
        "result": [outputs.shape[0]],
        "memory_rss_mb": [rss_mb],
        "memory_uss_mb": [uss_mb],
        "actor_pid": [pid],
    }


class ActorBasedRayData:
    """Actor-based worker for Ray Data using ModelWrapper.from_model(..., mode="actor")."""

    def __init__(self, model_wrapper: ModelWrapper):
        self.pid = os.getpid()
        print(f"[Actor-based] Worker {self.pid} started. Loading model...")

        # Load model using zero-copy
        self.pipeline = model_wrapper.load()

        # Measure memory after loading
        gc.collect()
        self.rss_mb = get_memory_mb(self.pid)
        try:
            process = psutil.Process(self.pid)
            self.uss_mb = process.memory_full_info().uss / 1024 / 1024
        except:
            self.uss_mb = 0

        print(
            f"[Actor-based] Worker {self.pid} ready. RSS: {self.rss_mb:.1f} MB, USS: {self.uss_mb:.1f} MB"
        )

    def __call__(self, batch):
        """Process a batch of data."""
        batch_size = int(batch["size"][0]) if hasattr(batch["size"], "__len__") else int(batch["size"])

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
# Benchmark Execution Functions
# ============================================================================


def run_ray_core_normal(model, workers, batches, batch_size, duration):
    """Run normal actor approach with Ray Core."""
    print("\n" + "=" * 80)
    print("RAY CORE - NORMAL ACTOR (Baseline)")
    print("=" * 80)

    # Put model in object store - Ray will automatically deserialize it in workers
    # This still creates a copy per worker (baseline behavior)
    model_ref = ray.put(model)
    sleep_time = max(0, duration - 1)  # Leave 1 second for processing

    with monitor_memory_context(interval=1.0) as memory_stats:
        start_time = time.time()

        # Pass model_ref - Ray will deserialize it in each worker (creating copies)
        futures = [
            normal_actor_worker.remote(model_ref, batches, batch_size, sleep_time)
            for _ in range(workers)
        ]
        results = ray.get(futures)

        end_time = time.time()
        runtime = end_time - start_time

    # Aggregate memory stats
    worker_rss = [r["rss_mb"] for r in results]
    worker_uss = [r["uss_mb"] for r in results]

    return {
        "peak_total_rss_mb": memory_stats["peak_total_rss"],
        "peak_total_uss_mb": memory_stats["peak_total_uss"],
        "avg_rss_per_worker_mb": np.mean(worker_rss) if worker_rss else 0,
        "avg_uss_per_worker_mb": np.mean(worker_uss) if worker_uss else 0,
        "runtime_s": runtime,
        "throughput_batches_per_s": (workers * batches) / runtime if runtime > 0 else 0,  # Total batch operations across all workers
        "num_workers": len(results),
    }


def run_ray_core_task_based(model, workers, batches, batch_size, duration):
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
    sleep_time = max(0, duration - 1)

    with monitor_memory_context(interval=1.0) as memory_stats:
        start_time = time.time()

        futures = [
            task_based_worker.remote(wrapped_pipeline, batches, batch_size, sleep_time)
            for _ in range(workers)
        ]
        results = ray.get(futures)

        end_time = time.time()
        runtime = end_time - start_time

    # Aggregate memory stats
    worker_rss = [r["rss_mb"] for r in results]
    worker_uss = [r["uss_mb"] for r in results]

    return {
        "peak_total_rss_mb": memory_stats["peak_total_rss"],
        "peak_total_uss_mb": memory_stats["peak_total_uss"],
        "avg_rss_per_worker_mb": np.mean(worker_rss) if worker_rss else 0,
        "avg_uss_per_worker_mb": np.mean(worker_uss) if worker_uss else 0,
        "runtime_s": runtime,
        "throughput_batches_per_s": (workers * batches) / runtime if runtime > 0 else 0,  # Total batch operations across all workers
        "num_workers": len(results),
    }


def run_ray_core_actor_based(model, workers, batches, batch_size, duration):
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
    sleep_time = max(0, duration - 1)

    with monitor_memory_context(interval=1.0) as memory_stats:
        start_time = time.time()

        # Create actors
        actors = [ActorBasedWorker.remote(model_wrapper, sleep_time) for _ in range(workers)]

        # Process batches
        futures = [actor.process_batches.remote(batches, batch_size) for actor in actors]
        results = ray.get(futures)

        end_time = time.time()
        runtime = end_time - start_time

    # Aggregate memory stats
    worker_rss = [r["rss_mb"] for r in results]
    worker_uss = [r["uss_mb"] for r in results]

    return {
        "peak_total_rss_mb": memory_stats["peak_total_rss"],
        "peak_total_uss_mb": memory_stats["peak_total_uss"],
        "avg_rss_per_worker_mb": np.mean(worker_rss) if worker_rss else 0,
        "avg_uss_per_worker_mb": np.mean(worker_uss) if worker_uss else 0,
        "runtime_s": runtime,
        "throughput_batches_per_s": (workers * batches) / runtime if runtime > 0 else 0,  # Total batch operations across all workers
        "num_workers": len(results),
    }


def run_ray_data_normal(model, workers, batches, batch_size, duration):
    """Run normal actor approach with Ray Data."""
    print("\n" + "=" * 80)
    print("RAY DATA - NORMAL ACTOR (Baseline)")
    print("=" * 80)

    model_ref = ray.put(model)

    # Create dataset
    ds = ray.data.range(batches).map(lambda x: {"size": batch_size})

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
        )

        # Materialize results
        results_df = results.to_pandas()

        end_time = time.time()
        runtime = end_time - start_time

        # Keep running for duration
        elapsed = runtime
        remaining = max(0, duration - elapsed)
        if remaining > 0:
            print(f"\nKeeping actors alive for {remaining:.1f} more seconds...")
            time.sleep(remaining)
            runtime = time.time() - start_time

    # Aggregate memory stats from results
    if not results_df.empty and "memory_rss_mb" in results_df.columns:
        from ray.data.aggregate import Count, Max

        actor_stats = (
            results.groupby("actor_pid")
            .aggregate(
                Max("memory_rss_mb", alias_name="max_rss_mb"),
                Max("memory_uss_mb", alias_name="max_uss_mb"),
                Count(alias_name="samples"),
            )
            .to_pandas()
        )

        if not actor_stats.empty:
            total_rss = actor_stats["max_rss_mb"].sum()
            total_uss = actor_stats["max_uss_mb"].sum()
            avg_rss = actor_stats["max_rss_mb"].mean()
            avg_uss = actor_stats["max_uss_mb"].mean()
            num_workers = len(actor_stats)
        else:
            total_rss = total_uss = avg_rss = avg_uss = num_workers = 0
    else:
        total_rss = memory_stats["peak_total_rss"]
        total_uss = memory_stats["peak_total_uss"]
        avg_rss = memory_stats["peak_avg_rss"] if "peak_avg_rss" in memory_stats else 0
        avg_uss = 0
        num_workers = workers

    return {
        "peak_total_rss_mb": total_rss,
        "peak_total_uss_mb": total_uss,
        "avg_rss_per_worker_mb": avg_rss,
        "avg_uss_per_worker_mb": avg_uss,
        "runtime_s": runtime,
        "throughput_batches_per_s": batches / runtime if runtime > 0 else 0,  # Total batches from dataset
        "num_workers": num_workers,
    }


def run_ray_data_task_based(model, workers, batches, batch_size, duration):
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
    ds = ray.data.range(batches).map(lambda x: {"size": batch_size})

    # Disable progress bars
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False

    with monitor_memory_context(interval=1.0) as memory_stats:
        start_time = time.time()

        # Use map_batches with a function (not actor) for task-based execution
        # The wrapped_pipeline will spawn Ray tasks for each inference call
        # Put wrapped pipeline in object store for serialization
        wrapped_pipeline_ref = ray.put(wrapped_pipeline)
        results = ds.map_batches(
            task_based_function,
            batch_size=1,
            fn_kwargs={"wrapped_pipeline_ref": wrapped_pipeline_ref},
        )

        # Materialize results
        results_df = results.to_pandas()

        end_time = time.time()
        runtime = end_time - start_time

        # Keep running for duration
        elapsed = runtime
        remaining = max(0, duration - elapsed)
        if remaining > 0:
            print(f"\nKeeping workers alive for {remaining:.1f} more seconds...")
            time.sleep(remaining)
            runtime = time.time() - start_time

    # Aggregate memory stats
    if not results_df.empty and "memory_rss_mb" in results_df.columns:
        from ray.data.aggregate import Count, Max

        worker_stats = (
            results.groupby("actor_pid")
            .aggregate(
                Max("memory_rss_mb", alias_name="max_rss_mb"),
                Max("memory_uss_mb", alias_name="max_uss_mb"),
                Count(alias_name="samples"),
            )
            .to_pandas()
        )

        if not worker_stats.empty:
            total_rss = worker_stats["max_rss_mb"].sum()
            total_uss = worker_stats["max_uss_mb"].sum()
            avg_rss = worker_stats["max_rss_mb"].mean()
            avg_uss = worker_stats["max_uss_mb"].mean()
            num_workers = len(worker_stats)
        else:
            total_rss = total_uss = avg_rss = avg_uss = num_workers = 0
    else:
        total_rss = memory_stats["peak_total_rss"]
        total_uss = memory_stats["peak_total_uss"]
        avg_rss = memory_stats["peak_avg_rss"] if "peak_avg_rss" in memory_stats else 0
        avg_uss = 0
        num_workers = workers

    return {
        "peak_total_rss_mb": total_rss,
        "peak_total_uss_mb": total_uss,
        "avg_rss_per_worker_mb": avg_rss,
        "avg_uss_per_worker_mb": avg_uss,
        "runtime_s": runtime,
        "throughput_batches_per_s": batches / runtime if runtime > 0 else 0,  # Total batches from dataset
        "num_workers": num_workers,
    }


def run_ray_data_actor_based(model, workers, batches, batch_size, duration):
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
    ds = ray.data.range(batches).map(lambda x: {"size": batch_size})

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

        # Keep running for duration
        elapsed = runtime
        remaining = max(0, duration - elapsed)
        if remaining > 0:
            print(f"\nKeeping actors alive for {remaining:.1f} more seconds...")
            time.sleep(remaining)
            runtime = time.time() - start_time

    # Aggregate memory stats
    if not results_df.empty and "memory_rss_mb" in results_df.columns:
        from ray.data.aggregate import Count, Max

        actor_stats = (
            results.groupby("actor_pid")
            .aggregate(
                Max("memory_rss_mb", alias_name="max_rss_mb"),
                Max("memory_uss_mb", alias_name="max_uss_mb"),
                Count(alias_name="samples"),
            )
            .to_pandas()
        )

        if not actor_stats.empty:
            total_rss = actor_stats["max_rss_mb"].sum()
            total_uss = actor_stats["max_uss_mb"].sum()
            avg_rss = actor_stats["max_rss_mb"].mean()
            avg_uss = actor_stats["max_uss_mb"].mean()
            num_workers = len(actor_stats)
        else:
            total_rss = total_uss = avg_rss = avg_uss = num_workers = 0
    else:
        total_rss = memory_stats["peak_total_rss"]
        total_uss = memory_stats["peak_total_uss"]
        avg_rss = memory_stats["peak_avg_rss"] if "peak_avg_rss" in memory_stats else 0
        avg_uss = 0
        num_workers = workers

    return {
        "peak_total_rss_mb": total_rss,
        "peak_total_uss_mb": total_uss,
        "avg_rss_per_worker_mb": avg_rss,
        "avg_uss_per_worker_mb": avg_uss,
        "runtime_s": runtime,
        "throughput_batches_per_s": batches / runtime if runtime > 0 else 0,  # Total batches from dataset
        "num_workers": num_workers,
    }


# ============================================================================
# Results Output
# ============================================================================


def print_comparison_table(results: Dict[str, Dict[str, Any]], model_size_mb: float):
    """Print a comparison table of results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 80)

    approaches = ["normal", "task_based", "actor_based"]
    approach_names = {
        "normal": "Normal Actor",
        "task_based": "Task-based",
        "actor_based": "Actor-based",
    }

    # Print header
    print(f"\n{'Approach':<20} {'RSS (MB)':>12} {'USS (MB)':>12} {'Runtime (s)':>12} {'Throughput':>12}")
    print("-" * 80)

    # Print results for each approach
    for approach in approaches:
        if approach in results:
            r = results[approach]
            name = approach_names[approach]
            print(
                f"{name:<20} {r['peak_total_rss_mb']:>12.1f} {r['peak_total_uss_mb']:>12.1f} "
                f"{r['runtime_s']:>12.2f} {r['throughput_batches_per_s']:>12.2f}"
            )

    # Calculate savings
    if "normal" in results and "actor_based" in results:
        normal_uss = results["normal"]["peak_total_uss_mb"]
        actor_uss = results["actor_based"]["peak_total_uss_mb"]
        if normal_uss > 0:
            savings_pct = (1 - actor_uss / normal_uss) * 100
            print("\n" + "-" * 80)
            print(f"Memory Savings (Actor-based vs Normal): {savings_pct:.1f}%")
            print(f"Expected savings: ~{(1 - 1/results['normal']['num_workers'])*100:.0f}%")
        print(f"\nModel size: {model_size_mb:.1f} MB")
        print(f"Expected normal mode memory: ~{results['normal']['num_workers'] * model_size_mb:.0f} MB")
        print(f"Expected zero-copy memory: ~{model_size_mb:.0f} MB")

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
    parser.add_argument("--batches", type=int, default=20, help="Number of batches to process")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration to keep workers alive (seconds)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MEMORY BENCHMARK")
    print("=" * 80)
    print(f"Execution Mode: {args.mode.upper()}")
    print(f"Workers: {args.workers}")
    print(f"Batches: {args.batches}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Duration: {args.duration}s")
    print("=" * 80)

    # Initialize Ray
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True)
        except Exception as e:
            print(f"Warning: Ray init failed with {e}, trying with explicit object store memory...")
            try:
                ray.init(object_store_memory=4 * 1024 * 1024 * 1024, ignore_reinit_error=True)
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
        "duration": args.duration,
    }

    results = {}

    try:
        if args.mode == "ray_core":
            # Run all three approaches with Ray Core
            results["normal"] = run_ray_core_normal(
                model, args.workers, args.batches, args.batch_size, args.duration
            )
            time.sleep(2)  # Brief pause between runs

            results["task_based"] = run_ray_core_task_based(
                model, args.workers, args.batches, args.batch_size, args.duration
            )
            time.sleep(2)  # Brief pause between runs

            results["actor_based"] = run_ray_core_actor_based(
                model, args.workers, args.batches, args.batch_size, args.duration
            )

        elif args.mode == "ray_data":
            # Run all three approaches with Ray Data
            results["normal"] = run_ray_data_normal(
                model, args.workers, args.batches, args.batch_size, args.duration
            )
            time.sleep(2)  # Brief pause between runs

            results["task_based"] = run_ray_data_task_based(
                model, args.workers, args.batches, args.batch_size, args.duration
            )
            time.sleep(2)  # Brief pause between runs

            results["actor_based"] = run_ray_data_actor_based(
                model, args.workers, args.batches, args.batch_size, args.duration
            )

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
