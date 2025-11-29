# Metadata and Type Hint Preservation in Wrappers

## Overview

All wrapper classes (`TaskWrapper`, `JITTaskWrapper`, `ActorWrapper`, `JITActorWrapper`) now preserve docstrings, metadata, and **type hints** from the wrapped pipeline classes, making the API more Pythonic, type-safe, and discoverable by IDEs.

## What's Preserved

### 1. **`__wrapped__` Attribute**
Following Python's standard wrapper convention (used by `functools.wraps`), all wrappers now expose a `__wrapped__` attribute that references the original pipeline object.

```python
pipeline = MyPipeline()
wrapped = TaskWrapper(pipeline)

assert wrapped.__wrapped__ is pipeline  # True
```

### 2. **Class Docstrings**
The original pipeline's docstring is appended to the wrapper's docstring, so `help()` and IDE documentation show both:

```python
class MyPipeline:
    """Custom pipeline for data processing."""
    # ...

wrapped = TaskWrapper(pipeline)
print(wrapped.__doc__)
# Shows both TaskWrapper's docs AND MyPipeline's docs
```

### 3. **Type Annotations**
If the pipeline has `__annotations__`, they are preserved on the wrapper:

```python
class MyPipeline:
    input_dim: int = 784
    output_dim: int = 10
    # ...

wrapped = TaskWrapper(pipeline)
print(wrapped.__annotations__)  # {'input_dim': int, 'output_dim': int}
```

### 4. **Method Signatures (Attempted)**
The wrappers attempt to preserve the `__call__` method's signature using `functools.update_wrapper`. This is done on a best-effort basis.

### 5. **Type Hints (Generic[T])**
All wrapper classes are now generic (`Generic[T]`) and properly typed to preserve the wrapped type for IDEs and type checkers:

```python
from ray_zerocopy import TaskWrapper

class Pipeline:
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # ...
        pass

pipeline = Pipeline()
wrapped = TaskWrapper(pipeline)  # Type is TaskWrapper[Pipeline]

# Type-safe access to original
original: Pipeline = wrapped.__wrapped__  # Fully typed!
```

This means:
- IDEs (Pylance, PyCharm) understand the wrapped type
- Autocomplete works on `wrapped.__wrapped__`
- Type checkers can verify correct usage
- `mypy` and `pyright` can check types properly

## Implementation Details

### Key Changes to Each Wrapper Class

1. **Made classes generic:**
   - All wrappers now inherit from `Generic[T]`
   - Type variable `T` represents the wrapped pipeline type
   - Enables proper type inference in IDEs and type checkers

2. **Added in `__init__`:**
   - Set `__wrapped__: T` attribute (Python convention, fully typed)
   - Merge docstrings from original class
   - Copy `__annotations__` if present
   - Apply `functools.update_wrapper` to `__call__` method

3. **Added `py.typed` marker:**
   - Enables PEP 561 type checking support
   - Allows external type checkers to use package's type hints

### Example Implementation (TaskWrapper)

```python
import functools

class TaskWrapper:
    def __init__(self, pipeline: Any, method_names: tuple = ("__call__",)):
        self._rewritten = rzc_nn.rewrite_pipeline(pipeline, method_names)

        # Store reference to original (Python convention for wrappers)
        self.__wrapped__ = pipeline

        # Copy class-level metadata from the wrapped pipeline
        if pipeline.__class__.__doc__:
            # Append original docstring to wrapper's docstring
            self.__doc__ = (
                f"{self.__class__.__doc__}\n\n"
                f"Wrapped class documentation:\n{pipeline.__class__.__doc__}"
            )

        # Preserve annotations if available
        if hasattr(pipeline, '__annotations__'):
            self.__annotations__ = pipeline.__annotations__

        # Preserve __call__ signature and docstring if it exists
        if hasattr(pipeline, '__call__') and callable(getattr(pipeline, '__call__', None)):
            original_call = getattr(pipeline.__class__, '__call__', None)
            if original_call is not None:
                try:
                    functools.update_wrapper(self.__call__, original_call)
                except (AttributeError, TypeError):
                    pass
```

## Benefits

### 1. **Better IDE Support**
IDEs can now show documentation from the wrapped class when autocompleting or hovering over wrapper objects.

### 2. **Improved `help()` Output**
When users call `help(wrapped)`, they see documentation from both the wrapper and the original pipeline:

```python
>>> help(wrapped)
Help on TaskWrapper in module ray_zerocopy.wrappers:

<ray_zerocopy.wrappers.TaskWrapper object>
    Wrapper for zero-copy nn.Module inference via Ray tasks.
    ...

    Wrapped class documentation:
    Custom pipeline for processing data through encoder and decoder.
    ...
```

### 3. **Standard Python Convention**
Using `__wrapped__` follows the same pattern as `functools.wraps`, making the API familiar to Python developers:

```python
# Standard way to access the original wrapped object
original = wrapped.__wrapped__
```

### 4. **Introspection and Debugging**
Tools that inspect objects (debuggers, profilers, testing frameworks) can now discover and work with the original pipeline through the `__wrapped__` attribute.

## Testing

Two test scripts demonstrate the functionality:

1. **`examples/test_wrapper_metadata.py`** - Comprehensive tests for all wrapper classes
2. **`examples/new_api_examples.py`** - Includes `example_metadata_preservation()` function

Run the tests:

```bash
# Comprehensive test
python examples/test_wrapper_metadata.py

# As part of all examples
python examples/new_api_examples.py
```

## Backward Compatibility

✅ **Fully backward compatible** - All existing code continues to work without changes. The metadata preservation is purely additive and follows standard Python conventions.

## Future Enhancements

Potential improvements for the future:

1. **Better signature preservation** - Use `inspect.Signature` to create wrapper methods with exact signatures
2. **Preserve more methods** - Apply signature wrapping to other forwarded methods beyond `__call__`
3. **Type stub generation** - Auto-generate `.pyi` files for better static type checking

## References

- [PEP 362 - Function Signature Object](https://www.python.org/dev/peps/pep-0362/)
- [functools.wraps documentation](https://docs.python.org/3/library/functools.html#functools.wraps)
- [Python descriptor protocol](https://docs.python.org/3/howto/descriptor.html)
