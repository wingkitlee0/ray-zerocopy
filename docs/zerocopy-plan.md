# Unified Pipeline-Based API for ray_zerocopy

## Current State Analysis

The package has **inconsistent APIs** across use cases:

1. **nn.Module + Ray Tasks**: `rewrite_pipeline(pipeline)` in `invoke.py` - returns shim objects
2. **nn.Module + Ray Actors**: `prepare_model_for_actors(model)` + `rewrite_pipeline_for_actors(pipeline)` in `actor.py` - mix of model-level and pipeline-level APIs
3. **JIT + Ray Tasks**: `jit.rewrite_pipeline(pipeline)` in `jit/invoke.py` - separate module
4. **JIT + Ray Actors**: **Not currently supported**

## Proposed New API

Create four new wrapper classes in `src/ray_zerocopy/wrappers.py`:

### 1. TaskWrapper (nn.Module + Task-based)

```python
class TaskWrapper:
    """Wrapper for zero-copy nn.Module inference via Ray tasks.

    Wraps a pipeline (object with torch.nn.Module attributes) to enable
    zero-copy model loading when calling methods via Ray tasks.
    """

    def __init__(self, pipeline, method_names=("__call__",)):
        """
        Args:
            pipeline: Object containing torch.nn.Module models as attributes
            method_names: Model methods to expose via remote tasks
        """
        # Internally uses invoke.rewrite_pipeline()
        self._rewritten = rewrite_pipeline(pipeline, method_names)

    def __getattr__(self, name):
        # Forward attribute access to rewritten pipeline
        return getattr(self._rewritten, name)
```

### 2. JITTaskWrapper (JIT + Task-based)

```python
class JITTaskWrapper:
    """Wrapper for zero-copy TorchScript inference via Ray tasks.

    Wraps a pipeline (object with torch.jit.ScriptModule attributes) to enable
    zero-copy model loading when calling methods via Ray tasks.
    """

    def __init__(self, pipeline, method_names=("__call__", "forward")):
        # Internally uses jit.invoke.rewrite_pipeline()
        from ray_zerocopy.jit.invoke import rewrite_pipeline as jit_rewrite
        self._rewritten = jit_rewrite(pipeline, method_names)

    def __getattr__(self, name):
        return getattr(self._rewritten, name)
```

### 3. ActorWrapper (nn.Module + Actor-based)

```python
class ActorWrapper:
    """Wrapper for zero-copy nn.Module inference in Ray actors.

    Prepares a pipeline for use in Ray Data with ActorPoolStrategy.
    Models are stored in Ray object store and loaded with zero-copy in each actor.
    """

    def __init__(self, pipeline, model_attr_names=None, device=None, use_fast_load=False):
        """
        Args:
            pipeline: Object containing torch.nn.Module models
            model_attr_names: List of model attribute names (auto-detected if None)
            device: Target device for actors (e.g., "cuda:0")
            use_fast_load: Use faster but riskier loading method
        """
        from ray_zerocopy.actor import rewrite_pipeline_for_actors
        self._skeleton, self._model_refs = rewrite_pipeline_for_actors(
            pipeline, model_attr_names, device, use_fast_load
        )
        self._device = device
        self._use_fast_load = use_fast_load

    def load(self, device=None, use_fast_load=None):
        """Load the pipeline in an actor (call from actor __init__)."""
        from ray_zerocopy.actor import load_pipeline_in_actor
        return load_pipeline_in_actor(
            self._skeleton,
            self._model_refs,
            device=device or self._device,
            use_fast_load=use_fast_load or self._use_fast_load
        )

    @property
    def constructor_kwargs(self):
        """Get kwargs for Ray Data fn_constructor_kwargs."""
        return {
            "pipeline_skeleton": self._skeleton,
            "model_refs": self._model_refs,
            "device": self._device,
            "use_fast_load": self._use_fast_load
        }
```

### 4. JITActorWrapper (NEW: JIT + Actor-based)

```python
class JITActorWrapper:
    """Wrapper for zero-copy TorchScript inference in Ray actors.

    Prepares a TorchScript pipeline for use in Ray Data with ActorPoolStrategy.
    Models are stored in Ray object store and loaded with zero-copy in each actor.
    """

    def __init__(self, pipeline, model_attr_names=None, device=None):
        # NEW implementation - create actor support for JIT models
```

## Implementation Plan

### Step 1: Create new wrappers module

- **File**: `src/ray_zerocopy/wrappers.py`
- Implement all four classes: `TaskWrapper`, `ActorWrapper`, `JITTaskWrapper`, `JITActorWrapper`
- Each class wraps existing low-level functions
- Add comprehensive docstrings with usage examples

### Step 2: Implement JIT actor support

- **File**: `src/ray_zerocopy/jit/actor.py` (NEW)
- Create `prepare_jit_model_for_actors(jit_model)`
- Create `load_jit_model_in_actor(model_ref, device)`
- Create `rewrite_pipeline_for_actors(pipeline, model_attr_names, device)` for JIT
- Similar to `src/ray_zerocopy/actor.py` but for JIT models

### Step 3: Add deprecation warnings

- Add deprecation warnings to old functions:
  - `invoke.rewrite_pipeline()` → point to `TaskWrapper`
  - `actor.prepare_model_for_actors()` → point to `ActorWrapper`
  - `actor.rewrite_pipeline_for_actors()` → point to `ActorWrapper`
  - `jit.invoke.rewrite_pipeline()` → point to `JITTaskWrapper`
- Keep functions working (backward compatible)

### Step 4: Update public API

- **File**: `src/ray_zerocopy/__init__.py`
- Export new classes: `TaskWrapper`, `ActorWrapper`, `JITTaskWrapper`, `JITActorWrapper`
- Keep old exports but document as deprecated
- Update docstrings

### Step 5: Create migration guide and examples

- **File**: `examples/new_api_examples.py`
- Show all four use cases side-by-side
- Include migration examples from old API
- **File**: `docs/migration_guide.md`
- Document old → new API mappings
- Explain when to use each class

### Step 6: Update existing examples

- Modernize `examples/ray_data_actor_example.py` to use `ZeroCopyActorUDF`
- Modernize `examples/example_jit.py` to use `ZeroCopyJITModel`
- Add new example: `examples/jit_actor_example.py` for `ZeroCopyJITActorUDF`

### Step 7: Update tests

- Create `tests/test_pipeline_api.py` for new classes
- Ensure backward compatibility tests still pass
- Add tests for JIT actor support

## Key Design Decisions

1. **Pipeline-first**: All classes expect a pipeline object (with models as attributes), not raw models
2. **Consistent naming**: `ZeroCopy[Type][ExecutionMode]` pattern
3. **Backward compatible**: Old functions still work with deprecation warnings
4. **Unified interface**: Similar methods across all classes where applicable

## File Changes Summary

- **NEW**: `src/ray_zerocopy/pipeline.py` - main new API
- **NEW**: `src/ray_zerocopy/jit/actor.py` - JIT actor support
- **MODIFY**: `src/ray_zerocopy/__init__.py` - exports
- **MODIFY**: `src/ray_zerocopy/invoke.py` - add deprecation warnings
- **MODIFY**: `src/ray_zerocopy/actor.py` - add deprecation warnings
- **MODIFY**: `src/ray_zerocopy/jit/invoke.py` - add deprecation warnings
- **NEW**: `docs/migration_guide.md`
- **NEW**: `examples/new_api_examples.py`
- **MODIFY**: Various example files
- **NEW**: `tests/test_pipeline_api.py`