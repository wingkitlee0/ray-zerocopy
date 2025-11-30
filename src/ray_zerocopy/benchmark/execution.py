"""Execution functions for benchmarks that work with both ModelWrapper and JIT wrappers."""

import time
from typing import Any, Callable, Union

import ray
from ray.data import ActorPoolStrategy

from ray_zerocopy import ModelWrapper
from ray_zerocopy.benchmark import monitor_memory_context
from ray_zerocopy.benchmark.results import (
    aggregate_memory_stats_from_results,
    format_ray_core_results,
)
from ray_zerocopy.benchmark.workers import (
    ActorBasedRayData,
    ActorBasedWorker,
    NormalActorRayData,
    normal_actor_worker,
    task_based_function,
    task_based_worker,
)
from ray_zerocopy.wrappers import JITActorWrapper, JITTaskWrapper

# Type alias for wrapper types
WrapperType = Union[ModelWrapper, JITTaskWrapper, JITActorWrapper]


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


def run_ray_core_task_based(
    model, workers, batches, batch_size, create_wrapper: Callable[[Any], Any]
):
    """Run task-based approach with Ray Core.

    Args:
        model: The model (regular nn.Module or TorchScript model)
        workers: Number of workers
        batches: Number of batches
        batch_size: Batch size
        create_wrapper: Function that takes a Pipeline and returns a wrapped pipeline
                        (e.g., ModelWrapper.for_tasks or JITTaskWrapper)
    """
    print("\n" + "=" * 80)
    print("RAY CORE - TASK-BASED")
    print("=" * 80)

    # Create pipeline wrapper
    class Pipeline:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    pipeline = Pipeline(model)
    wrapped_pipeline = create_wrapper(pipeline)

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


def run_ray_core_actor_based(
    model, workers, batches, batch_size, create_wrapper: Callable[[Any], WrapperType]
):
    """Run actor-based approach with Ray Core.

    Args:
        model: The model (regular nn.Module or TorchScript model)
        workers: Number of workers
        batches: Number of batches
        batch_size: Batch size
        create_wrapper: Function that takes a Pipeline and returns a wrapper
                        (e.g., ModelWrapper.from_model(..., mode="actor") or JITActorWrapper)
    """
    print("\n" + "=" * 80)
    print("RAY CORE - ACTOR-BASED")
    print("=" * 80)

    # Create pipeline wrapper
    class Pipeline:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    pipeline = Pipeline(model)
    model_wrapper = create_wrapper(pipeline)

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


def run_ray_data_task_based(
    model, workers, batches, batch_size, create_wrapper: Callable[[Any], Any]
):
    """Run task-based approach with Ray Data.

    Args:
        model: The model (regular nn.Module or TorchScript model)
        workers: Number of workers
        batches: Number of batches
        batch_size: Batch size
        create_wrapper: Function that takes a Pipeline and returns a wrapped pipeline
                        (e.g., ModelWrapper.for_tasks or JITTaskWrapper)
    """
    print("\n" + "=" * 80)
    print("RAY DATA - TASK-BASED")
    print("=" * 80)

    # Create pipeline wrapper
    class Pipeline:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    pipeline = Pipeline(model)
    wrapped_pipeline = create_wrapper(pipeline)

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
        results = ds.map(
            task_based_function,
            fn_kwargs={"wrapped_pipeline": wrapped_pipeline},
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


def run_ray_data_actor_based(
    model, workers, batches, batch_size, create_wrapper: Callable[[Any], WrapperType]
):
    """Run actor-based approach with Ray Data.

    Args:
        model: The model (regular nn.Module or TorchScript model)
        workers: Number of workers
        batches: Number of batches
        batch_size: Batch size
        create_wrapper: Function that takes a Pipeline and returns a wrapper
                        (e.g., ModelWrapper.from_model(..., mode="actor") or JITActorWrapper)
    """
    print("\n" + "=" * 80)
    print("RAY DATA - ACTOR-BASED")
    print("=" * 80)

    # Create pipeline wrapper
    class Pipeline:
        def __init__(self, model):
            self.model = model

        def __call__(self, x):
            return self.model(x)

    pipeline = Pipeline(model)
    model_wrapper = create_wrapper(pipeline)

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
