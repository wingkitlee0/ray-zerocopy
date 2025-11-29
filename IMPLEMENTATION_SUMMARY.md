# Consolidate nn API into Single ModelWrapper - Implementation Summary

## Overview

Successfully consolidated `TaskWrapper`, `ActorWrapper`, and `ModelWrapper` into a unified `ModelWrapper` class that handles both task and actor execution modes. The old wrapper classes remain as thin wrappers for backward compatibility.

## Changes Made

### 1. Updated `model_wrappers.py`

**Key Changes:**
- Added `_mode` field to track execution mode (`"task"` or `"actor"`)
- Added `_rewritten` field for task mode (stores immediately-usable rewritten pipeline)
- Added `_model_info` field for task mode (stores method tracking info)
- Updated `from_model()` method with `mode` parameter:
  - Task mode: Calls `prepare_pipeline()` then `load_pipeline_for_tasks()` immediately
  - Actor mode: Only calls `prepare_pipeline_for_actors()`, no loading until `to_pipeline()` is called
- Added `for_tasks()` class method as convenience shortcut (equivalent to `rewrite_pipeline()`)
- Updated `to_pipeline()` to handle both modes
- Added `__call__()` and `__getattr__()` to forward to rewritten pipeline in task mode
- Updated `__getstate__()` and `__setstate__()` for proper pickling
- Updated `serialize()` and `deserialize()` methods

**New API:**

```python
# Task mode - immediately loads and returns usable wrapper
wrapper = ModelWrapper.from_model(pipeline, mode="task", method_names=None)
result = wrapper.process(data)  # Ready to use immediately

# Convenience shortcut
wrapper = ModelWrapper.for_tasks(pipeline, method_names=None)
result = wrapper.process(data)  # Equivalent to rewrite_pipeline()

# Actor mode - prepares but doesn't load (load happens in actor)
wrapper = ModelWrapper.from_model(pipeline, mode="actor", model_attr_names=None)
# In actor:
pipeline = wrapper.to_pipeline(device="cuda:0")
```

### 2. Updated `wrappers.py`

**Key Changes:**
- `TaskWrapper` now delegates to `ModelWrapper.from_model(..., mode="task")`
- `ActorWrapper` now delegates to `ModelWrapper.from_model(..., mode="actor")`
- Both maintain backward compatibility with existing API
- Added notes in docstrings recommending the new ModelWrapper API

**Backward Compatibility:**
```python
# Still works, delegates to ModelWrapper
wrapper = TaskWrapper(pipeline, method_names=("__call__",))
wrapper = ActorWrapper(pipeline, model_attr_names=None)
```

### 3. Added Tests

**New Test File: `test_model_wrapper_unified.py`**
- 13 comprehensive tests covering:
  - Task mode with standalone models and pipelines
  - `for_tasks()` convenience method
  - Actor mode with standalone models and pipelines
  - Pickling for both modes
  - Mode validation
  - Custom method names
  - Standalone vs pipeline handling
  - Serialize/deserialize
  - Attribute access behavior

**Test Results:**
- All 84 tests pass (71 existing + 13 new)
- No linting errors
- Backward compatibility maintained

### 4. Added Example

**New Example: `unified_model_wrapper_example.py`**
- 6 examples demonstrating:
  - Task mode basic usage
  - Task mode shortcut (`for_tasks()`)
  - Actor mode with parallel inference
  - Pipeline in task mode
  - Pipeline in actor mode
  - Mode comparison

## Key Design Decisions

### 1. Why Single Wrapper Works
Both modes share:
- Same preparation logic (`prepare_pipeline()`)
- Same model extraction and storage
- Only differ in loading mechanism and method tracking

### 2. Mode Parameter Design
- **Task mode**: Immediately callable after creation
  - Internally: `prepare_pipeline()` + `load_pipeline_for_tasks()`
  - Use case: Dynamic workloads with automatic task scheduling
  
- **Actor mode**: Requires explicit `to_pipeline()` call
  - Internally: Only `prepare_pipeline_for_actors()`
  - Use case: Stateful workloads with persistent actors

### 3. Backward Compatibility
- Old `TaskWrapper` and `ActorWrapper` classes remain functional
- They delegate to the new unified `ModelWrapper`
- All existing tests pass without modification
- `constructor_kwargs` property maintains old format for compatibility

## Migration Guide

### For Task-Based Execution

**Old API:**
```python
from ray_zerocopy import TaskWrapper
wrapper = TaskWrapper(pipeline)
result = wrapper(data)
```

**New API (Recommended):**
```python
from ray_zerocopy import ModelWrapper
wrapper = ModelWrapper.for_tasks(pipeline)
result = wrapper(data)
```

### For Actor-Based Execution

**Old API:**
```python
from ray_zerocopy import ActorWrapper
wrapper = ActorWrapper(pipeline)
# In actor:
loaded = wrapper.load(device="cuda:0")
```

**New API (Recommended):**
```python
from ray_zerocopy import ModelWrapper
wrapper = ModelWrapper.from_model(pipeline, mode="actor")
# In actor:
loaded = wrapper.to_pipeline(device="cuda:0")
```

## Benefits

1. **Unified API**: Single class handles both execution modes
2. **Clearer Intent**: Mode parameter makes the execution model explicit
3. **Better Composability**: Easier to switch between task and actor modes
4. **Maintained Compatibility**: Existing code continues to work
5. **Reduced Code Duplication**: Task and actor wrappers share implementation

## Testing

All tests pass:
- 71 existing tests (unchanged)
- 13 new tests for unified ModelWrapper
- No regressions
- No linting errors

## Files Modified

1. `src/ray_zerocopy/model_wrappers.py` - Unified ModelWrapper implementation
2. `src/ray_zerocopy/wrappers.py` - Updated TaskWrapper and ActorWrapper to delegate
3. `tests/test_model_wrapper_unified.py` - New comprehensive tests
4. `examples/unified_model_wrapper_example.py` - New example demonstrating API

## Conclusion

The consolidation is complete and fully functional. The new unified ModelWrapper provides a cleaner, more intuitive API while maintaining full backward compatibility with existing code.
