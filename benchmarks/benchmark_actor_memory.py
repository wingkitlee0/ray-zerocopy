"""
Memory benchmark comparing three approaches:
1. Normal (no zero-copy) - Each actor has its own model copy
2. Actor zero-copy - Actors load model from object store with zero-copy
3. Task zero-copy - rewrite_pipeline approach (spawns tasks from actors)

Usage:
    # Normal (baseline)
    python benchmark_actor_memory.py --mode normal --actors 4

    # Actor zero-copy (NEW approach)
    python benchmark_actor_memory.py --mode actor --actors 4

    # Task zero-copy (OLD rewrite_pipeline)
    python benchmark_actor_memory.py --mode task --actors 4
"""

import argparse
import gc
import os
import time

import numpy as np
import psutil
import ray
import torch
from ray.data import ActorPoolStrategy

from ray_zerocopy import ZeroCopyModel
from ray_zerocopy.actor import load_model_in_actor, prepare_model_for_actors
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


# ============================================================================
# Approach 1: Normal (No Zero-Copy) - Baseline
# ============================================================================


class NormalActor:
    """Actor that loads model normally (full copy in each actor)."""

    def __init__(self, model_ref):
        self.pid = os.getpid()
        print(f"[Normal Actor {self.pid}] Loading model (full copy)...")
        # Get model from object store (this creates a copy in actor memory)
        self.model = ray.get(model_ref)
        gc.collect()
        mem = get_memory_mb(self.pid)
        print(f"[Normal Actor {self.pid}] Ready. Memory: {mem:.1f} MB")

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


# ============================================================================
# Approach 2: Actor Zero-Copy (NEW)
# ============================================================================


class ActorZeroCopyActor:
    """Actor that loads model using zero-copy from object store."""

    def __init__(self, model_ref):
        self.pid = os.getpid()
        print(f"[Actor ZeroCopy {self.pid}] Loading model (zero-copy)...")
        self.model = load_model_in_actor(model_ref)
        gc.collect()
        mem = get_memory_mb(self.pid)
        print(f"[Actor ZeroCopy {self.pid}] Ready. Memory: {mem:.1f} MB")

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


# ============================================================================
# Approach 3: Task Zero-Copy (using ZeroCopyModel - loads on each call)
# ============================================================================


class TaskZeroCopyActor:
    """Actor that uses ZeroCopyModel (loads model from object store on each call)."""

    def __init__(self, model_ref):
        self.pid = os.getpid()
        print(f"[Task ZeroCopy {self.pid}] Setting up zero-copy model...")
        self.model_ref = model_ref
        gc.collect()
        mem = get_memory_mb(self.pid)
        print(f"[Task ZeroCopy {self.pid}] Ready. Memory: {mem:.1f} MB")

    def __call__(self, batch):
        # This loads the model from object store on EVERY call!
        batch_size = (
            int(batch["size"][0])
            if hasattr(batch["size"], "__len__")
            else int(batch["size"])
        )
        inputs = torch.randn(batch_size, 5000)

        # Load model using ZeroCopyModel API (zero-copy from object store)
        # Note: from_object_ref expects the dereferenced tuple
        model_data = ray.get(self.model_ref)
        model = ZeroCopyModel.from_object_ref(model_data)

        with torch.no_grad():
            outputs = model(inputs)

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
            # Actor zero-copy: Prepare model for actors
            model_ref = prepare_model_for_actors(model)
            results = ds.map_batches(
                ActorZeroCopyActor,
                fn_constructor_kwargs={"model_ref": model_ref},
                batch_size=1,
                compute=ActorPoolStrategy(size=args.workers),
            )

        elif args.mode == "task":
            # Task zero-copy: Use ZeroCopyModel API
            # This loads model on each call (inefficient for actors!)
            class Pipeline:
                def __init__(self, model):
                    self.model = model

            pipeline = Pipeline(model)
            rewritten = ZeroCopyModel.rewrite(pipeline)
            model_ref = ZeroCopyModel.to_object_ref(rewritten.model)

            results = ds.map_batches(
                TaskZeroCopyActor,
                fn_constructor_kwargs={"model_ref": model_ref},
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
