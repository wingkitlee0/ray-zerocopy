# ModelWrapper

Primary API for zero-copy model sharing with nn.Module models.

## Overview

`ModelWrapper` provides a unified API for wrapping models that supports both task and actor execution modes:

- `from_model()` - Create wrapper from model/pipeline
- `for_tasks()` - Convenience method for task mode
- `load()` - Load model in actor
- Supports both task and actor execution modes

## ModelWrapper Class

```{eval-rst}
.. autoclass:: ray_zerocopy.model_wrappers.ModelWrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

## Basic Usage

### Wrapping a Model

```python
from ray_zerocopy import ModelWrapper

model = MyModel()
wrapper = ModelWrapper.from_model(model)  # "actor" mode by default
```

### Using in Actors

```python
class InferenceActor:
    def __init__(self, model_wrapper):
        # Load pipeline with zero-copy
        self.model = model_wrapper.load()

    def __call__(self, batch):
        return self.model(batch["data"])

# Pass wrapper to actor
ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": wrapper}
)
```

### Using for Tasks

```python
# Create wrapper for task mode
wrapped = ModelWrapper.for_tasks(pipeline)

# Use immediately - each call spawns a Ray task
result = wrapped(data)
```

## Methods

### from_model

Class method to create wrapper from model or pipeline.

```python
# Actor mode (default)
wrapper = ModelWrapper.from_model(model, mode="actor")

# Task mode
wrapper = ModelWrapper.from_model(model, mode="task")
```

**Parameters:**
- `model` - PyTorch model (`nn.Module`) or pipeline (object with nn.Module attributes)
- `mode` - Execution mode: "actor" (default) or "task"

**Returns:**
- `ModelWrapper` instance

### for_tasks

Convenience method for task mode.

```python
wrapped = ModelWrapper.for_tasks(pipeline)
```

**Parameters:**
- `pipeline` - PyTorch model or pipeline

**Returns:**
- A converted model or pipeline

### load

Load the model/pipeline (actor mode only).

```python
model = wrapper.load()
```

**Parameters:**
- `_use_fast_load` (optional) - Enable fast loading (experimental)

**Returns:**
- Loaded model or pipeline (on CPU). Users should handle device placement themselves.

## Supported Model Types

### Standalone nn.Module

```python
model = MyModel()
wrapper = ModelWrapper.from_model(model)
```

### Pipeline with Multiple Models

```python
class Pipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, x):
        return self.decoder(self.encoder(x))

pipeline = Pipeline()
wrapper = ModelWrapper.from_model(pipeline)
```

The wrapper automatically detects all `nn.Module` attributes.

## Execution Modes

### Actor Mode

For Ray Data and long-running actors:

```python
# Create wrapper in actor mode
wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# Use in actor
class Actor:
    def __init__(self, wrapper):
        self.model = wrapper.load()
```

### Task Mode

For ad-hoc inference with Ray tasks:

```python
# Create wrapper in task mode
wrapped = ModelWrapper.for_tasks(pipeline)

# Use immediately
result = wrapped(data)  # Each call spawns a Ray task
```

## Advanced Usage

### Conditional Fast Loading

```python
class Actor:
    def __init__(self, wrapper, use_fast):
        self.model = wrapper.load(_use_fast_load=use_fast)
```

### Inspecting Wrapper State

```python
wrapper = ModelWrapper.from_model(pipeline)

# Check if it's a standalone module
print(wrapper._is_standalone_module)

# See detected models
print(wrapper._model_info.keys())
```

## See Also

- [User Guide: Core Concepts](../user_guide/core_concepts.md)
- [User Guide: Ray Data Integration](../user_guide/ray_data_integration.md)
- [Tutorials](../tutorials/index.md)
