"""
Tests for the benchmark utilities module.
"""

import os
import threading
import time

import torch
import torch.nn as nn

from ray_zerocopy.benchmark import (
    create_large_model,
    estimate_model_size_mb,
    get_memory_mb,
    monitor_memory,
    monitor_memory_context,
)


def test_create_large_model():
    """Test that create_large_model creates a model of expected size."""
    model = create_large_model()

    assert isinstance(model, nn.Sequential)
    assert len(list(model.children())) > 0

    # Test that model can do forward pass
    model.eval()
    test_input = torch.randn(1, 5000)
    with torch.no_grad():
        output = model(test_input)

    assert output.shape == (1, 100)


def test_estimate_model_size_mb():
    """Test model size estimation."""
    # Create a small model
    small_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.Linear(50, 10),
    )

    size_mb = estimate_model_size_mb(small_model)

    assert size_mb > 0
    assert isinstance(size_mb, float)

    # Test with larger model
    large_model = create_large_model()
    large_size_mb = estimate_model_size_mb(large_model)

    assert large_size_mb > size_mb


def test_get_memory_mb():
    """Test memory retrieval function."""
    # Test with current process
    memory = get_memory_mb()

    assert memory > 0
    assert isinstance(memory, float)

    # Test with explicit PID
    current_pid = os.getpid()
    memory_explicit = get_memory_mb(current_pid)

    assert memory_explicit > 0

    # Test with invalid PID (should return 0)
    invalid_memory = get_memory_mb(999999999)
    assert invalid_memory == 0


def test_monitor_memory_basic():
    """Test basic memory monitoring functionality."""
    stats = {
        "peak_total_rss": 0.0,
        "peak_total_uss": 0.0,
        "peak_avg_rss": 0.0,
    }

    stop_event = threading.Event()

    # Start monitoring in a thread
    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(stop_event, stats, 0.1, False),
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    # Let it run for a short time
    time.sleep(0.3)

    # Stop monitoring
    stop_event.set()
    monitor_thread.join(timeout=1.0)

    # Stats should have been updated (even if no Ray workers found)
    assert "peak_total_rss" in stats
    assert "peak_total_uss" in stats
    assert "peak_avg_rss" in stats


def test_monitor_memory_with_parent():
    """Test memory monitoring with parent process tracking."""
    stats = {
        "peak_total_rss": 0.0,
        "peak_total_uss": 0.0,
        "peak_avg_rss": 0.0,
    }

    stop_event = threading.Event()

    # Start monitoring with parent tracking
    monitor_thread = threading.Thread(
        target=monitor_memory,
        args=(stop_event, stats, 0.1, True),
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    # Let it run briefly
    time.sleep(0.3)

    # Stop monitoring
    stop_event.set()
    monitor_thread.join(timeout=1.0)

    assert "peak_total_rss" in stats


def test_monitor_memory_context():
    """Test memory monitoring context manager."""
    with monitor_memory_context(interval=0.1, show_parent=False) as stats:
        # Do some work
        model = create_large_model()
        model.eval()
        test_input = torch.randn(1, 5000)
        with torch.no_grad():
            _ = model(test_input)

        time.sleep(0.2)  # Let monitor run

    # Check that stats were collected
    assert "peak_total_rss" in stats
    assert "peak_total_uss" in stats
    assert "peak_avg_rss" in stats


def test_monitor_memory_context_with_parent():
    """Test memory monitoring context manager with parent tracking."""
    with monitor_memory_context(interval=0.1, show_parent=True) as stats:
        time.sleep(0.2)

    assert "peak_total_rss" in stats


def test_estimate_model_size_mb_with_buffers():
    """Test model size estimation includes buffers."""

    class ModelWithBuffers(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
            self.register_buffer("buffer", torch.randn(10, 10))

    model = ModelWithBuffers()
    size_mb = estimate_model_size_mb(model)

    assert size_mb > 0

    # Model with buffers should be larger than without
    model_no_buffers = nn.Linear(10, 5)
    size_no_buffers = estimate_model_size_mb(model_no_buffers)

    assert size_mb > size_no_buffers
