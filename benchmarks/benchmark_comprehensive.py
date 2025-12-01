"""
Comprehensive memory benchmark comparing three approaches:
1. Normal Actor - Baseline with full model copies per actor
2. Task-based - Using ModelWrapper.for_tasks() or JITModelWrapper.for_tasks()
3. Actor-based - Using ModelWrapper.from_model(..., mode="actor") or JITModelWrapper.from_model(..., mode="actor")

Supports both Ray Core and Ray Data execution modes.
Supports both regular nn.Module models and TorchScript (JIT) models.

Usage:
    # Ray Core mode with regular model
    python benchmark_comprehensive.py --mode ray_core --workers 4 --batches 20

    # Ray Core mode with TorchScript model
    python benchmark_comprehensive.py --mode ray_core --workers 4 --batches 20 --jit

    # Ray Data mode with regular model
    python benchmark_comprehensive.py --mode ray_data --workers 4 --batches 20

    # Ray Data mode with TorchScript model
    python benchmark_comprehensive.py --mode ray_data --workers 4 --batches 20 --jit

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
import time
from functools import partial

import ray

from ray_zerocopy import JITModelWrapper, ModelWrapper
from ray_zerocopy.benchmark import (
    create_large_model,
    estimate_model_size_mb,
    print_comparison_table,
    run_ray_core_actor_based,
    run_ray_core_normal,
    run_ray_core_task_based,
    run_ray_data_actor_based,
    run_ray_data_normal,
    run_ray_data_task_based,
    save_results_json,
)


def create_wrapper_for_tasks(pipeline, use_jit: bool):
    """Create a task-based wrapper for the pipeline.

    Args:
        pipeline: Pipeline object with model
        use_jit: If True, use JITModelWrapper.for_tasks(); otherwise use ModelWrapper.for_tasks()

    Returns:
        Wrapped pipeline for task-based execution
    """
    if use_jit:
        return JITModelWrapper.for_tasks(pipeline)
    else:
        return ModelWrapper.for_tasks(pipeline)


def create_wrapper_for_actors(pipeline, use_jit: bool):
    """Create an actor-based wrapper for the pipeline.

    Args:
        pipeline: Pipeline object with model
        use_jit: If True, use JITModelWrapper.from_model(..., mode="actor"); otherwise use ModelWrapper.from_model(..., mode="actor")

    Returns:
        Wrapper for actor-based execution
    """
    if use_jit:
        return JITModelWrapper.from_model(pipeline, mode="actor")
    else:
        return ModelWrapper.from_model(pipeline, mode="actor")


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
        "--jit",
        "--torchscript",
        dest="use_jit",
        default=False,
        help="Use TorchScript (JIT) models instead of regular nn.Module models",
        action="store_true",
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
    print(f"Model Type: {'TorchScript (JIT)' if args.use_jit else 'Regular nn.Module'}")
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
    model = create_large_model(use_jit=args.use_jit)

    model_size_mb = estimate_model_size_mb(model)
    print(f"Model size: {model_size_mb:.1f} MB")

    config = {
        "workers": args.workers,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "use_jit": args.use_jit,
    }

    results = {}

    # Create wrapper factory functions
    def create_task_wrapper(pipeline):
        return create_wrapper_for_tasks(pipeline, args.use_jit)

    def create_actor_wrapper(pipeline):
        return create_wrapper_for_actors(pipeline, args.use_jit)

    try:
        if args.mode == "ray_core":
            # Run all three approaches with Ray Core
            approaches = {
                "normal": partial(
                    run_ray_core_normal,
                    model=model,
                    workers=args.workers,
                    batches=args.batches,
                    batch_size=args.batch_size,
                    use_jit=args.use_jit,
                ),
                "task_based": partial(
                    run_ray_core_task_based,
                    model,
                    args.workers,
                    args.batches,
                    args.batch_size,
                    create_task_wrapper,
                ),
                "actor_based": partial(
                    run_ray_core_actor_based,
                    model,
                    args.workers,
                    args.batches,
                    args.batch_size,
                    create_actor_wrapper,
                ),
            }

            for approach_name, run_func in approaches.items():
                results[approach_name] = run_func()
                print("\nWaiting 5 seconds after job completion...")
                time.sleep(5)

        elif args.mode == "ray_data":
            # Run all three approaches with Ray Data

            approaches = {
                "normal": partial(
                    run_ray_data_normal,
                    model=model,
                    workers=args.workers,
                    batches=args.batches,
                    batch_size=args.batch_size,
                    use_jit=args.use_jit,
                ),
                "task_based": partial(
                    run_ray_data_task_based,
                    model=model,
                    workers=args.workers,
                    batches=args.batches,
                    batch_size=args.batch_size,
                    create_wrapper=create_task_wrapper,
                ),
                "actor_based": partial(
                    run_ray_data_actor_based,
                    model=model,
                    workers=args.workers,
                    batches=args.batches,
                    batch_size=args.batch_size,
                    create_wrapper=create_actor_wrapper,
                ),
            }

            for approach_name, run_func in approaches.items():
                results[approach_name] = run_func()
                print("\nWaiting 5 seconds after job completion...")
                time.sleep(5)

                if args.ray_reset and ray.is_initialized():
                    ray.shutdown()

        # Print comparison table
        print_comparison_table(results, model_size_mb)

        # Save results to JSON
        save_results_json(results, config, args.mode, args.output, use_jit=args.use_jit)

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
