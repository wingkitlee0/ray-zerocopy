"""
Pytest configuration and shared fixtures for ray-zerocopy tests.
"""

import pytest
import ray
import torch
import torch.nn as nn


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    return nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )


@pytest.fixture
def large_model():
    """
    Create a larger PyTorch model with decent size for memory testing.
    This model has multiple layers and should be large enough to see
    memory differences.
    """
    return nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1500),
        nn.ReLU(),
        nn.Linear(1500, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
    )


@pytest.fixture
def transformer_like_model():
    """
    Create a transformer-like model with attention mechanism.
    This is more representative of real-world models.
    """

    class TransformerBlock(nn.Module):
        def __init__(self, dim=512, num_heads=8):
            super().__init__()
            self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )

        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10000, 512)
            self.blocks = nn.ModuleList([TransformerBlock() for _ in range(4)])
            self.output = nn.Linear(512, 1000)

        def forward(self, x):
            x = self.embedding(x)
            for block in self.blocks:
                x = block(x)
            x = x.mean(dim=1)  # Global average pooling
            return self.output(x)

    return Model()


@pytest.fixture
def simple_jit_model():
    """Create a simple TorchScript model for testing."""
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )
    model.eval()
    example_input = torch.randn(1, 100)
    return torch.jit.trace(model, example_input)


@pytest.fixture
def large_jit_model():
    """Create a larger TorchScript model for testing."""
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1500),
        nn.ReLU(),
        nn.Linear(1500, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
    )
    model.eval()
    example_input = torch.randn(1, 1000)
    return torch.jit.trace(model, example_input)


@pytest.fixture
def conv_jit_model():
    """Create a convolutional TorchScript model for testing."""

    class ConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(32 * 8 * 8, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = ConvModel()
    model.eval()
    example_input = torch.randn(1, 3, 32, 32)
    return torch.jit.trace(model, example_input)


@pytest.fixture
def sample_pipeline():
    """Create a sample pipeline object with a model attribute."""

    class Pipeline:
        def __init__(self, model):
            self.model = model
            self.config = {"batch_size": 32}

    return Pipeline


@pytest.fixture(scope="session")
def ray_cluster():
    """
    Shared Ray cluster fixture for all tests.
    Initializes Ray once per test session with multiple CPUs.
    This fixture is shared across all test modules.

    Pattern based on Ray's own test fixtures:
    - Checks if Ray is already initialized
    - Handles connection to existing clusters gracefully
    - Only initializes new cluster if needed
    """
    # Track if we initialized Ray in this fixture
    initialized_here = False

    # Check if Ray is already initialized
    if ray.is_initialized():
        # Ray is already running - use existing cluster
        yield
        return

    # Check for existing Ray cluster by trying to connect without num_cpus
    # This is the pattern used in Ray's own tests
    import os

    ray_address = os.environ.get("RAY_ADDRESS", "auto")

    # If RAY_ADDRESS is set or we're auto-detecting, try connecting first
    if ray_address != "auto" or os.path.exists("/tmp/ray"):
        # Likely connecting to existing cluster - don't specify num_cpus
        try:
            ray.init(
                address=ray_address if ray_address != "auto" else None,
                ignore_reinit_error=True,
            )
            initialized_here = False  # We connected, didn't create
        except Exception:
            # Connection failed, will try local init below
            pass

    # If not connected yet, try local initialization
    if not ray.is_initialized():
        try:
            # Try with num_cpus first (for new local cluster)
            ray.init(
                num_cpus=4,
                object_store_memory=4
                * 1024
                * 1024
                * 1024,  # 4GB object store (increased for larger models)
                ignore_reinit_error=True,
                _system_config={
                    # Optimize for testing
                    "object_timeout_milliseconds": 10000,
                },
            )
            initialized_here = True
        except ValueError as e:
            # ValueError usually means connecting to existing cluster
            # Try without num_cpus (pattern from Ray's test suite)
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in [
                    "num_cpus",
                    "num_gpus",
                    "existing cluster",
                    "connecting",
                ]
            ):
                try:
                    ray.init(
                        ignore_reinit_error=True,
                    )
                    initialized_here = False  # We connected, didn't create
                except Exception as e2:
                    if not ray.is_initialized():
                        raise RuntimeError(f"Failed to initialize Ray: {e2}") from e2
            else:
                raise RuntimeError(f"Failed to initialize Ray: {e}") from e
        except Exception as e:
            # Check if Ray got initialized despite the exception
            if not ray.is_initialized():
                raise RuntimeError(f"Failed to initialize Ray: {e}") from e

    yield

    # Cleanup: only shutdown if we initialized Ray in this fixture
    # Don't shutdown if Ray was already running (e.g., from outside the test)
    if initialized_here:
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            # Ignore shutdown errors - Ray might have been shut down already
            pass
