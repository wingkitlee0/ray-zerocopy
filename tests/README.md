# Test Suite for ray-zerocopy

This directory contains comprehensive tests for the ray-zerocopy package.

## Test Structure

- **`test_rewrite.py`**: Tests for the core rewrite functionality (`extract_tensors` and `replace_tensors`)
- **`test_invoke.py`**: Tests for the pipeline rewriting and remote invocation (`rewrite_pipeline` and `call_model`)
- **`test_memory_footprint.py`**: **Key tests** that verify zero-copy behavior by checking memory footprint with multiple workers
- **`conftest.py`**: Shared fixtures for creating test models

## Running Tests

### Install Dependencies

First, install the package with dev dependencies:

```bash
pip install -e ".[dev]"
```

Or if using uv:

```bash
uv pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Files

```bash
# Test basic functionality
pytest tests/test_rewrite.py

# Test pipeline rewriting
pytest tests/test_invoke.py

# Test memory footprint (most important for zero-copy verification)
pytest tests/test_memory_footprint.py -v -s
```

The `-s` flag shows print statements, which is useful for seeing memory statistics.

### Run Specific Tests

```bash
# Test memory footprint with multiple workers
pytest tests/test_memory_footprint.py::test_memory_footprint_multiple_workers -v -s
```

## Key Tests for Zero-Copy Verification

The most important tests are in `test_memory_footprint.py`:

1. **`test_memory_footprint_single_worker`**: Establishes baseline memory usage
2. **`test_memory_footprint_multiple_workers`**: **Critical test** that verifies memory footprint remains constant (not scaling with number of workers)
3. **`test_memory_footprint_sequential_calls`**: Verifies no memory accumulation with repeated calls
4. **`test_object_store_sharing`**: Verifies that multiple references share the same object store data

## Test Models

The tests use models of varying sizes:

- **Simple model**: Small sequential model for basic functionality tests
- **Large model**: ~50-100MB model with multiple large layers for memory testing
- **Transformer-like model**: More realistic model with attention mechanisms

## Expected Behavior

With zero-copy working correctly:

- Memory footprint should **not** scale linearly with the number of workers
- Multiple workers should share the same model weights from Ray's object store
- Sequential calls should not accumulate memory
- Model outputs should match the original model exactly


