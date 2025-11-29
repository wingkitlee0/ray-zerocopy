# ModelWrapper

Alternative API with explicit serialization/deserialization methods.

## Overview

`ModelWrapper` provides a more explicit API for wrapping models:

- `from_model()` - Create wrapper from model/pipeline
- `unwrap()` - Deserialize model in actor
- `to_actor()` - Get serialized form for actor

This API is more verbose but gives you more control over the serialization process.

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
wrapper = ModelWrapper.from_model(model)
```

### Using in Actors

```python
class InferenceActor:
    def __init__(self, model_wrapper):
        # Unwrap to get the model
        self.model = model_wrapper.unwrap(device="cuda:0")

    def __call__(self, batch):
        return self.model(batch["data"])

# Pass wrapper to actor
ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": wrapper}
)
```

### Alternative: Explicit Actor Preparation

```python
# Get serialized form
actor_config = wrapper.to_actor()

class InferenceActor:
    def __init__(self, actor_config):
        # Recreate wrapper from config
        wrapper = ModelWrapper.from_actor(**actor_config)
        self.model = wrapper.unwrap(device="cuda:0")

    def __call__(self, batch):
        return self.model(batch["data"])

ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_config": actor_config}
)
```

## Methods

### from_model

Class method to create wrapper from model or pipeline.

```python
wrapper = ModelWrapper.from_model(model)
```

**Parameters:**
- `model` - PyTorch model (`nn.Module`) or pipeline (object with nn.Module attributes)

**Returns:**
- `ModelWrapper` instance

### unwrap

Deserialize and load the model.

```python
model = wrapper.unwrap(device="cuda:0")
```

**Parameters:**
- `device` (optional) - Target device ("cpu", "cuda:0", etc.)
- `use_fast_load` (optional) - Enable fast loading (experimental)

**Returns:**
- Loaded model or pipeline

### to_actor

Get serialized form for passing to actors.

```python
actor_config = wrapper.to_actor()
```

**Returns:**
- Dictionary with serialization data

### from_actor

Class method to recreate wrapper from actor config.

```python
wrapper = ModelWrapper.from_actor(**actor_config)
```

**Parameters:**
- `**actor_config` - Unpacked config dictionary from `to_actor()`

**Returns:**
- `ModelWrapper` instance

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

## Comparison with Primary API

### ModelWrapper (Alternative API)

```python
# Wrap
wrapper = ModelWrapper.from_model(model)

# Use in actor
class Actor:
    def __init__(self, wrapper):
        self.model = wrapper.unwrap(device="cuda:0")
```

### ActorWrapper (Primary API)

```python
# Wrap
wrapper = ActorWrapper(model, device="cuda:0")

# Use in actor
class Actor:
    def __init__(self, wrapper):
        self.model = wrapper.load()
```

**Differences:**
- `ModelWrapper` has explicit `from_model()` and `unwrap()`
- `ActorWrapper` has simpler `__init__()` and `load()`
- Both achieve the same zero-copy sharing

**Recommendation:** Use `ActorWrapper` unless you need the explicit control of `ModelWrapper`.

## Advanced Usage

### Custom Device Selection

You can override the device when unwrapping:

```python
wrapper = ModelWrapper.from_model(model)

class Actor:
    def __init__(self, wrapper, gpu_id):
        # Different actors can use different GPUs
        self.model = wrapper.unwrap(device=f"cuda:{gpu_id}")
```

### Conditional Fast Loading

```python
class Actor:
    def __init__(self, wrapper, use_fast):
        self.model = wrapper.unwrap(
            device="cuda:0",
            use_fast_load=use_fast
        )
```

### Inspecting Wrapper State

```python
wrapper = ModelWrapper.from_model(pipeline)

# Check if it's a standalone module
print(wrapper._is_standalone_module)

# See detected models
print(wrapper._model_refs.keys())
```

## See Also

- [ActorWrapper](wrappers.md#actorwrapper) - Simpler alternative
- [User Guide: ActorWrapper](../user_guide/actors.md)
- [Tutorials](../tutorials/index.md)
