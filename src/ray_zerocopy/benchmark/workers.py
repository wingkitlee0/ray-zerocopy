"""Common worker implementations for benchmarks.

These workers can work with both ModelWrapper and JIT wrappers.
The wrappers are used via duck typing - any object with the appropriate interface will work.
"""

import gc
import os
from typing import Any

import psutil
import ray
import torch

from ray_zerocopy import ModelWrapper
from .monitor import get_memory_mb, log_memory

# Type alias for wrapper types - using Any to be generic
WrapperType = Any


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
    """Task-based worker using ModelWrapper.for_tasks() or JIT wrappers."""
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
    """Actor-based worker using ModelWrapper.from_model(..., mode="actor") or JIT wrappers."""

    def __init__(self, model_wrapper: WrapperType):
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
    """Task-based function for Ray Data using ModelWrapper.for_tasks() or JIT wrappers."""

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
    """Actor-based worker for Ray Data using ModelWrapper.from_model(..., mode="actor") or JIT wrappers."""

    def __init__(self, model_wrapper: WrapperType):
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
