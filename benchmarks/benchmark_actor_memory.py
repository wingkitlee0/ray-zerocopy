"""
Memory benchmark comparing three approaches:
1. Normal (no zero-copy) - Each actor has its own model copy
2. Actor zero-copy - Actors load model from object store with zero-copy using ActorWrapper
3. Task zero-copy - TaskWrapper approach (spawns Ray tasks from actors for each call)

Usage:
    # Normal (baseline)
    python benchmark_actor_memory.py --mode normal --workers 4

    # Actor zero-copy (ActorWrapper - RECOMMENDED for actors)
    python benchmark_actor_memory.py --mode actor --workers 4

    # Task zero-copy (TaskWrapper - spawns tasks, less efficient for actors)
    python benchmark_actor_memory.py --mode task --workers 4
"""

import argparse
import gc
import os
import time
from contextlib import contextmanager

import numpy as np
import psutil
import ray
import torch
from ray.data import ActorPoolStrategy

from ray_zerocopy import ActorWrapper, TaskWrapper
from ray_zerocopy.benchmark import (
    create_large_model,
    get_memory_mb,
    monitor_memory_context,
)


@contextmanager
def log_memory(name: str):
    print(f"[{name}] Loading model (full copy)...")
    yield
    gc.collect()
    mem = get_memory_mb(os.getpid())
    print(f"[{name}] Ready. Memory: {mem:.1f} MB")


class NormalActor:
    """Actor that loads model normally (full copy in each actor)."""

    def __init__(self, model_ref):
        self.pid = os.getpid()
        with log_memory(f"Normal Actor {self.pid}"):
            self.model = ray.get(model_ref)

    def __call__(self, batch):
        with torch.no_grad():
            # batch["size"] is a numpy array with batch_size values
            batch_size = (
                int(batch["size"][0])
                if hasattr(batch["size"], "__len__")
                else int(batch["size"])
            )
            inputs = torch.randn(batch_size, 5000)
            outputs = self.model(inputs)

        # Measure memory after inference
        gc.collect()
        rss_mb = get_memory_mb(self.pid)
        try:
            process = psutil.Process(self.pid)
            uss_mb = process.memory_full_info().uss / 1024 / 1024
        except:
            uss_mb = 0

        return {
            "result": [outputs.shape[0]],
            "memory_rss_mb": [rss_mb],
            "memory_uss_mb": [uss_mb],
            "actor_pid": [self.pid],
        }


class ActorZeroCopyActor:
    """Actor that loads model using zero-copy from object store with ActorWrapper."""

    def __init__(self, actor_wrapper: ActorWrapper):
        self.pid = os.getpid()
        with log_memory(f"Actor ZeroCopy {self.pid}"):
            # Use ActorWrapper.load() to reconstruct the pipeline with zero-copy
            self.pipeline = actor_wrapper.load()

    def __call__(self, batch: dict[str, np.ndarray]):
        with torch.no_grad():
            batch_size = int(batch["size"][0])
            inputs = torch.randn(batch_size, 5000)
            outputs = self.pipeline(inputs)

        # Measure memory after inference
        gc.collect()
        rss_mb = get_memory_mb(self.pid)
        try:
            process = psutil.Process(self.pid)
            uss_mb = process.memory_full_info().uss / 1024 / 1024
        except:
            uss_mb = 0

        return {
            "result": [outputs.shape[0]],
            "memory_rss_mb": [rss_mb],
            "memory_uss_mb": [uss_mb],
            "actor_pid": [self.pid],
        }


class TaskZeroCopyActor:
    """Actor that uses TaskWrapper (spawns Ray tasks for inference on each call)."""

    def __init__(self, wrapped_pipeline):
        self.pid = os.getpid()
        with log_memory(f"Task ZeroCopy {self.pid}"):
            # Store the wrapped pipeline (will spawn Ray tasks for inference)
            self.wrapped_pipeline = wrapped_pipeline

    def __call__(self, batch):
        batch_size = int(batch["size"][0])

        with torch.no_grad():
            inputs = torch.randn(batch_size, 5000)
            outputs = self.wrapped_pipeline(inputs)

        # Measure memory after inference
        gc.collect()
        rss_mb = get_memory_mb(self.pid)
        try:
            process = psutil.Process(self.pid)
            uss_mb = process.memory_full_info().uss / 1024 / 1024
        except:
            uss_mb = 0

        return {
            "result": [outputs.shape[0]],
            "memory_rss_mb": [rss_mb],
            "memory_uss_mb": [uss_mb],
            "actor_pid": [self.pid],
        }


def main():
    parser = argparse.ArgumentParser(description="Memory benchmark for Ray Data actors")
    parser.add_argument(
        "--mode",
        choices=["normal", "actor", "task"],
        required=True,
        help="Mode: normal (no zero-copy), actor (new zero-copy), task (old rewrite_pipeline)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--batches", type=int, default=20, help="Number of batches to process"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--duration", type=int, default=30, help="Duration to run (seconds)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f"MEMORY BENCHMARK: {args.mode.upper()} MODE")
    print("=" * 80)
    print(f"Workers: {args.workers}")
    print(f"Batches: {args.batches}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Duration: {args.duration}s")
    print("=" * 80)

    # Create model
    print("\nCreating large model (~500MB)...")
    model = create_large_model()
    model_size_mb = (
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    )
    print(f"Model size: {model_size_mb:.1f} MB")

    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = False

    # Create dataset
    print(f"\nCreating dataset with {args.batches} batches...")
    ds = ray.data.range(args.batches).map(lambda x: {"size": args.batch_size})

    with monitor_memory_context() as memory_stats:
        print(f"\nStarting {args.mode} mode with {args.workers} workers...")
        print("=" * 80)

        start_time = time.time()

        if args.mode == "normal":
            # Normal: Each actor gets full model copy
            model_ref = ray.put(model)
            results = ds.map_batches(
                NormalActor,
                fn_constructor_kwargs={"model_ref": model_ref},
                batch_size=1,
                compute=ActorPoolStrategy(size=args.workers),
            )

        elif args.mode == "actor":
            # Actor zero-copy: Use ActorWrapper
            class Pipeline:
                def __init__(self, model):
                    self.model = model

                def __call__(self, x):
                    return self.model(x)

            pipeline = Pipeline(model)
            actor_wrapper = ActorWrapper(pipeline)

            results = ds.map_batches(
                ActorZeroCopyActor,
                fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
                batch_size=1,
                compute=ActorPoolStrategy(size=args.workers),
            )

        elif args.mode == "task":
            # Task zero-copy: Use TaskWrapper
            # This spawns Ray tasks for inference (inefficient for actors!)
            class Pipeline:
                def __init__(self, model):
                    self.model = model

                def __call__(self, x):
                    return self.model(x)

            pipeline = Pipeline(model)
            wrapped = TaskWrapper(pipeline)

            results = ds.map_batches(
                TaskZeroCopyActor,
                fn_constructor_kwargs={"wrapped_pipeline": wrapped},
                batch_size=1,
                compute=ActorPoolStrategy(size=args.workers),
            )
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

        # Materialize results and aggregate using Ray Data APIs
        print("\nProcessing batches...")

        # Use Ray Data groupby to aggregate memory stats per actor
        def aggregate_actor_memory(group):
            """Aggregate memory stats for one actor."""
            # group is a dict with arrays
            pid = group["actor_pid"][0]
            max_rss = np.max(group["memory_rss_mb"])
            max_uss = np.max(group["memory_uss_mb"])
            samples = len(group["memory_rss_mb"])

            return {
                "actor_pid": [pid],
                "max_rss_mb": [float(max_rss)],
                "max_uss_mb": [float(max_uss)],
                "samples": [samples],
            }

        # Aggregate per actor
        from ray.data.aggregate import Count, Max

        actor_stats_df = (
            results.groupby("actor_pid")
            .aggregate(
                Max("memory_rss_mb", alias_name="max_rss_mb"),
                Max("memory_uss_mb", alias_name="max_uss_mb"),
                Count(alias_name="samples"),
            )
            .to_pandas()
        )

        # Calculate totals
        if not actor_stats_df.empty:
            actor_stats_df = actor_stats_df.rename(
                columns={
                    "max_rss_mb": "max_rss",
                    "max_uss_mb": "max_uss",
                }
            )
            actor_memory_stats = actor_stats_df.set_index("actor_pid").to_dict("index")

            total_rss = actor_stats_df["max_rss"].sum()
            total_uss = actor_stats_df["max_uss"].sum()
            avg_rss = actor_stats_df["max_rss"].mean()
            avg_uss = actor_stats_df["max_uss"].mean()
            num_actors = len(actor_stats_df)
        else:
            actor_memory_stats = {}
            total_rss = total_uss = avg_rss = avg_uss = num_actors = 0

        # Keep running for the duration to observe memory
        elapsed = time.time() - start_time
        remaining = max(0, args.duration - elapsed)
        if remaining > 0:
            print(f"\nKeeping actors alive for {remaining:.1f} more seconds...")
            time.sleep(remaining)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Batches processed: {results.count()}")
    print(f"Throughput: {results.count() / total_time:.2f} batches/s")
    print("=" * 80)

    # Memory usage summary from actors
    print("\nMEMORY USAGE SUMMARY:")
    print(f"  Model size: {model_size_mb:.0f} MB")
    print(f"  Actors requested: {args.workers}")

    if num_actors > 0:
        print(f"  Actors detected: {num_actors}")
        print("\n  MEASURED (from actors during inference):")
        print(f"    Total RSS: {total_rss:.0f} MB (across {num_actors} actors)")
        print(f"    Total USS: {total_uss:.0f} MB (private memory)")
        print(f"    Avg RSS/actor: {avg_rss:.0f} MB")
        print(f"    Avg USS/actor: {avg_uss:.0f} MB")

        # Per-actor breakdown
        print("\n  Per-Actor Memory:")
        for pid, stats in sorted(actor_memory_stats.items()):
            print(
                f"    PID {pid}: RSS={stats['max_rss']:.0f} MB, USS={stats['max_uss']:.0f} MB ({stats['samples']} samples)"
            )
    else:
        print("\n  MEASURED: No actor memory data captured")

    print("\n  EXPECTED (model memory only):")
    print(
        f"    Normal mode: ~{args.workers * model_size_mb:.0f} MB (each actor has copy)"
    )
    print(f"    Actor mode:  ~{model_size_mb:.0f} MB (shared via zero-copy)")
    print(f"    Task mode:   ~{model_size_mb:.0f} MB (shared but loads on each call)")

    # Calculate efficiency if we have data
    if num_actors > 0 and args.mode != "normal":
        normal_expected = args.workers * model_size_mb
        savings_pct = (1 - total_uss / normal_expected) * 100
        print("\n  MEMORY SAVINGS:")
        print(f"    vs Normal mode: ~{savings_pct:.0f}% reduction (expected ~50%)")

    print("=" * 80)


if __name__ == "__main__":
    main()
