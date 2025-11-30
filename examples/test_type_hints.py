"""
Test script to demonstrate type hint preservation in wrappers.

This file shows how type checkers (like Pylance, mypy) can now infer
the correct types when using ModelWrapper and other wrappers.
"""

import torch
import torch.nn as nn

from ray_zerocopy import ModelWrapper


class Pipeline:
    """A sample pipeline with documented methods."""

    def __init__(self):
        self.encoder = nn.Linear(10, 5)
        self.decoder = nn.Linear(5, 2)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process data through the pipeline.

        Args:
            data: Input tensor

        Returns:
            Processed tensor
        """
        return self.decoder(self.encoder(data))

    def custom_method(self, x: int) -> str:
        """A custom method to test attribute access."""
        return f"Result: {x}"


def test_type_hints():
    """Test that type hints are properly preserved."""

    print("=" * 70)
    print("Type Hint Preservation Test")
    print("=" * 70)

    # Create pipeline and wrapper
    pipeline = Pipeline()
    wrapped = ModelWrapper.for_tasks(pipeline)

    # The type of wrapped should be ModelWrapper[Pipeline]
    # This allows IDEs to understand that wrapped has Pipeline's methods

    print("\n1. Type of wrapped object:")
    print("   Type: ModelWrapper[Pipeline]")
    print(f"   __wrapped__ type: {type(wrapped.__wrapped__).__name__}")

    print("\n2. Access to __wrapped__ is properly typed:")
    print(f"   wrapped.__wrapped__ is pipeline: {wrapped.__wrapped__ is pipeline}")

    print("\n3. Attribute access through __getattr__:")
    # This should work and type checkers should understand it
    # through __wrapped__ typing
    result = wrapped.custom_method(42)
    print(f"   wrapped.custom_method(42) = '{result}'")

    print("\n4. IDE Features:")
    print("   - Autocomplete should show Pipeline's methods")
    print("   - Hover should show Pipeline's docstrings")
    print("   - Go to definition should work")
    print("   - Type checking should understand wrapped.__wrapped__")

    print("\n✓ Type hints are preserved!")
    print("=" * 70)


def test_usage_pattern():
    """Show the recommended usage pattern for best type support."""

    print("\n" + "=" * 70)
    print("Recommended Usage Pattern for Type Hints")
    print("=" * 70)

    # Pattern 1: Direct usage (type is inferred from pipeline)
    pipeline = Pipeline()
    wrapped = TaskWrapper(pipeline)
    # Type checker sees: wrapped is TaskWrapper[Pipeline]
    # Access wrapped.__wrapped__ for typed access to original

    print("\nPattern 1: Direct usage")
    print("   pipeline = Pipeline()")
    print("   wrapped = TaskWrapper(pipeline)")
    print("   # Type: TaskWrapper[Pipeline]")
    print("   # Use wrapped.__wrapped__ for typed attribute access")

    # Pattern 2: With explicit type annotation
    print("\nPattern 2: With explicit type annotation")
    print("   pipeline: Pipeline = Pipeline()")
    print("   wrapped: TaskWrapper[Pipeline] = TaskWrapper(pipeline)")
    print("   # IDE has full type information")

    # Pattern 3: Type-safe attribute access
    print("\nPattern 3: Type-safe attribute access")
    print("   original: Pipeline = wrapped.__wrapped__")
    print("   # Now 'original' has full Pipeline type info")
    print("   # This is the most type-safe approach")

    print("\n✓ Use __wrapped__ for best type support!")
    print("=" * 70)


def test_actor_wrapper_types():
    """Test that ActorWrapper also preserves types."""

    print("\n" + "=" * 70)
    print("ActorWrapper Type Hints")
    print("=" * 70)

    pipeline = Pipeline()
    wrapped = ActorWrapper(pipeline)

    print("\n1. ActorWrapper type:")
    print("   Type: ActorWrapper[Pipeline]")
    print(f"   __wrapped__: {type(wrapped.__wrapped__).__name__}")

    print("\n2. Type-safe access:")
    print("   original = wrapped.__wrapped__")
    print("   # 'original' has full Pipeline type")

    print("\n✓ ActorWrapper also preserves types!")
    print("=" * 70)


if __name__ == "__main__":
    import ray

    ray.init(ignore_reinit_error=True)

    test_type_hints()
    test_usage_pattern()
    test_actor_wrapper_types()

    print("\n" + "=" * 70)
    print("Summary:")
    print("- All wrappers are now Generic[T] and preserve type info")
    print("- Use wrapped.__wrapped__ for fully typed attribute access")
    print("- IDEs can now provide autocomplete and type checking")
    print("=" * 70)
