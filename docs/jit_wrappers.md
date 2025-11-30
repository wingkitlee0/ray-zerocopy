# JIT Wrappers

**Note:** JIT wrappers for TorchScript models are currently under development.

## Overview

JIT wrappers (`JITTaskWrapper` and `JITActorWrapper`) provide zero-copy model sharing for TorchScript (compiled) models. These wrappers are similar to `ModelWrapper` but work with `torch.jit.ScriptModule` objects instead of `nn.Module` objects.

## Status

JIT wrappers are under active development. The API may change in future releases.

## API Reference

### JITTaskWrapper

For task-based inference with TorchScript models:

```python
from ray_zerocopy import JITTaskWrapper

# Compile your model
jit_model = torch.jit.trace(model, example_input)

# Wrap for task-based execution
wrapped = JITTaskWrapper(jit_model)
```

### JITActorWrapper

For actor-based inference with TorchScript models:

```python
from ray_zerocopy import JITActorWrapper

# Compile your model
jit_model = torch.jit.trace(model, example_input)

# Wrap for actor-based execution
actor_wrapper = JITActorWrapper(jit_model)

# Use in actor
class InferenceActor:
    def __init__(self, actor_wrapper):
        self.model = actor_wrapper.load()
```

## When Available

For production use, we recommend using `ModelWrapper` with standard `nn.Module` models, which is fully supported and stable.

For JIT wrapper updates and availability, please check the project repository.

