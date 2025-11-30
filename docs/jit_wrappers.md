# JIT Wrappers

## Overview

`JITModelWrapper` provides zero-copy model sharing for TorchScript (compiled) models. It supports both task-based and actor-based execution modes, similar to `ModelWrapper` but works with `torch.jit.ScriptModule` objects instead of `nn.Module` objects.

## API Reference

### JITModelWrapper

A unified wrapper that supports both task and actor execution modes for TorchScript models.

#### Task Mode

For task-based inference with TorchScript models:

```python
from ray_zerocopy import JITModelWrapper

# Compile your model
jit_model = torch.jit.trace(model, example_input)

# Wrap for task-based execution (immediate use)
wrapped = JITModelWrapper.for_tasks(jit_model)
result = wrapped(data)  # Each call spawns a Ray task
```

Or using `from_model()`:

```python
wrapper = JITModelWrapper.from_model(jit_model, mode="task")
wrapped = wrapper.load()
result = wrapped(data)
```

#### Actor Mode

For actor-based inference with TorchScript models:

```python
from ray_zerocopy import JITModelWrapper

# Compile your model
jit_model = torch.jit.trace(model, example_input)

# Wrap for actor-based execution
wrapper = JITModelWrapper.from_model(jit_model, mode="actor")

# Use in actor
class InferenceActor:
    def __init__(self, wrapper):
        self.model = wrapper.load()
```

## Usage Examples

### Task Mode with Pipeline

```python
class Pipeline:
    def __init__(self):
        self.encoder = torch.jit.trace(EncoderModel(), example_input)
        self.decoder = torch.jit.trace(DecoderModel(), example_encoded)
    
    def __call__(self, x):
        return self.decoder(self.encoder(x))

pipeline = Pipeline()
wrapped = JITModelWrapper.for_tasks(pipeline)
result = wrapped(data)
```

### Actor Mode with Ray Data

```python
wrapper = JITModelWrapper.from_model(jit_pipeline, mode="actor")

class InferenceActor:
    def __init__(self, wrapper):
        self.pipeline = wrapper.load()
    
    def __call__(self, batch):
        return self.pipeline(batch["data"])

ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"wrapper": wrapper},
    compute=ActorPoolStrategy(size=4)
)
```

## Comparison with ModelWrapper

`JITModelWrapper` provides the same unified API as `ModelWrapper`:
- `from_model()` - Create wrapper from model/pipeline
- `for_tasks()` - Convenience method for task mode
- `load()` - Load model in actor (actor mode) or get callable pipeline (task mode)
- Supports both task and actor execution modes

The main difference is that `JITModelWrapper` works with `torch.jit.ScriptModule` objects instead of `nn.Module` objects.

