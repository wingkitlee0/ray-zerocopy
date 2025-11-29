"""
Test script to demonstrate that wrappers preserve docstrings and metadata.
"""

import inspect

import torch
import torch.nn as nn

from ray_zerocopy import ActorWrapper, JITActorWrapper, JITTaskWrapper, TaskWrapper


class MyPipeline:
    """
    Custom pipeline for processing data through encoder and decoder.

    This is a sample pipeline that demonstrates the preservation of docstrings
    and type hints when wrapped by TaskWrapper or other wrappers.
    """

    def __init__(self):
        """Initialize the pipeline with encoder and decoder models."""
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 2)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process data through the pipeline.

        Args:
            data: Input tensor of shape (batch_size, 10)

        Returns:
            Output tensor of shape (batch_size, 2)
        """
        encoded = self.encoder(data)
        return self.decoder(encoded)

    def process(self, data: torch.Tensor) -> torch.Tensor:
        """Alternative method to process data."""
        return self(data)


def test_task_wrapper_metadata():
    """Test that TaskWrapper preserves metadata."""
    print("=" * 70)
    print("Testing TaskWrapper metadata preservation")
    print("=" * 70)

    pipeline = MyPipeline()
    wrapped = TaskWrapper(pipeline)

    # Test __wrapped__ attribute (Python convention)
    assert hasattr(wrapped, "__wrapped__"), "Should have __wrapped__ attribute"
    assert wrapped.__wrapped__ is pipeline, "Should reference original pipeline"
    print("✓ __wrapped__ attribute present and correct")

    # Test docstring preservation
    print("\nOriginal Pipeline docstring:")
    print(pipeline.__class__.__doc__)
    print("\nWrapped object docstring:")
    print(wrapped.__doc__)

    assert pipeline.__class__.__doc__ in wrapped.__doc__, (
        "Should contain original docstring"
    )
    print("✓ Docstring preserved")

    # Test __call__ signature
    print("\nOriginal __call__ signature:")
    print(inspect.signature(pipeline.__call__))
    print("\nWrapped __call__ signature:")
    print(inspect.signature(wrapped.__call__))

    # Test __call__ docstring
    print("\nOriginal __call__ docstring:")
    print(pipeline.__call__.__doc__)
    print("\nWrapped __call__ docstring:")
    print(wrapped.__call__.__doc__)

    if pipeline.__call__.__doc__ == wrapped.__call__.__doc__:
        print("✓ __call__ docstring preserved")
    else:
        print("⚠ __call__ docstring differs (may be expected)")

    print("\n✓ TaskWrapper metadata preservation working!")


def test_actor_wrapper_metadata():
    """Test that ActorWrapper preserves metadata."""
    print("\n" + "=" * 70)
    print("Testing ActorWrapper metadata preservation")
    print("=" * 70)

    pipeline = MyPipeline()
    wrapped = ActorWrapper(pipeline)

    # Test __wrapped__ attribute
    assert hasattr(wrapped, "__wrapped__"), "Should have __wrapped__ attribute"
    assert wrapped.__wrapped__ is pipeline, "Should reference original pipeline"
    print("✓ __wrapped__ attribute present and correct")

    # Test docstring preservation
    print("\nWrapped ActorWrapper docstring (first 200 chars):")
    print(wrapped.__doc__[:200] + "...")

    assert pipeline.__class__.__doc__ in wrapped.__doc__, (
        "Should contain original docstring"
    )
    print("✓ Original pipeline docstring included in wrapper's docstring")

    print("\n✓ ActorWrapper metadata preservation working!")


def test_jit_task_wrapper_metadata():
    """Test that JITTaskWrapper preserves metadata."""
    print("\n" + "=" * 70)
    print("Testing JITTaskWrapper metadata preservation")
    print("=" * 70)

    # Create a simple nn.Module for JIT
    class JITPipeline(nn.Module):
        """Pipeline for JIT compilation."""

        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 5)
            self.decoder = nn.Linear(5, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through pipeline."""
            return self.decoder(self.encoder(x))

    pipeline = JITPipeline()

    # Note: We wrap the pipeline object, not the traced version
    wrapped = JITTaskWrapper(pipeline)

    # Test __wrapped__ attribute
    assert hasattr(wrapped, "__wrapped__"), "Should have __wrapped__ attribute"
    print("✓ __wrapped__ attribute present")

    # Test docstring preservation
    if pipeline.__class__.__doc__:
        assert pipeline.__class__.__doc__ in wrapped.__doc__, (
            "Should contain original docstring"
        )
        print("✓ Original pipeline docstring included in wrapper's docstring")

    print("✓ JITTaskWrapper metadata preservation working!")


def test_jit_actor_wrapper_metadata():
    """Test that JITActorWrapper preserves metadata."""
    print("\n" + "=" * 70)
    print("Testing JITActorWrapper metadata preservation")
    print("=" * 70)

    # Create a simple nn.Module for JIT
    class JITPipeline(nn.Module):
        """Pipeline for JIT actor compilation."""

        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 5)
            self.decoder = nn.Linear(5, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through pipeline."""
            return self.decoder(self.encoder(x))

    pipeline = JITPipeline()
    wrapped = JITActorWrapper(pipeline)

    # Test __wrapped__ attribute
    assert hasattr(wrapped, "__wrapped__"), "Should have __wrapped__ attribute"
    print("✓ __wrapped__ attribute present")

    # Test docstring preservation
    if pipeline.__class__.__doc__:
        assert pipeline.__class__.__doc__ in wrapped.__doc__, (
            "Should contain original docstring"
        )
        print("✓ Original pipeline docstring included in wrapper's docstring")

    print("✓ JITActorWrapper metadata preservation working!")


def test_help_function():
    """Test that help() works well with wrapped objects."""
    print("\n" + "=" * 70)
    print("Testing help() function compatibility")
    print("=" * 70)

    pipeline = MyPipeline()
    wrapped = TaskWrapper(pipeline)

    print("\nCalling help(wrapped) (showing first 500 chars):")
    print("-" * 70)

    # Capture help output
    import contextlib
    import io

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        help(wrapped)
    help_output = f.getvalue()

    print(help_output[:500] + "...")

    # Check that original docstring is in help output
    assert "Custom pipeline for processing data" in help_output, (
        "Original docstring should be in help"
    )
    print("\n✓ help() function shows original docstring!")


if __name__ == "__main__":
    test_task_wrapper_metadata()
    test_actor_wrapper_metadata()
    test_jit_task_wrapper_metadata()
    test_jit_actor_wrapper_metadata()
    test_help_function()

    print("\n" + "=" * 70)
    print("All metadata preservation tests passed! ✓")
    print("=" * 70)
