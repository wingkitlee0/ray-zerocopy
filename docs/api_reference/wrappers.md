# Wrapper Classes

Primary API for ray-zerocopy with four wrapper classes.

## TaskWrapper

```{eval-rst}
.. autoclass:: ray_zerocopy.wrappers.TaskWrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__
```

### Usage Example

```python
from ray_zerocopy import TaskWrapper

pipeline = MyPipeline()
wrapped = TaskWrapper(pipeline, device="cuda:0")

# Each call spawns a Ray task
result = wrapped(data)
```

## ActorWrapper

```{eval-rst}
.. autoclass:: ray_zerocopy.wrappers.ActorWrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

### Usage Example

```python
from ray_zerocopy import ActorWrapper
from ray.data import ActorPoolStrategy

pipeline = MyPipeline()
actor_wrapper = ActorWrapper(pipeline, device="cuda:0")

class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()

    def __call__(self, batch):
        return self.pipeline(batch["data"])

results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    compute=ActorPoolStrategy(size=4)
)
```

## JITTaskWrapper

```{eval-rst}
.. autoclass:: ray_zerocopy.wrappers.JITTaskWrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__
```

### Usage Example

```python
import torch
from ray_zerocopy import JITTaskWrapper

# Compile model
model = MyModel()
jit_model = torch.jit.trace(model, example_input)

# Wrap and use
wrapped = JITTaskWrapper(jit_model, device="cuda:0")
result = wrapped(data)
```

## JITActorWrapper

```{eval-rst}
.. autoclass:: ray_zerocopy.wrappers.JITActorWrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

### Usage Example

```python
import torch
from ray_zerocopy import JITActorWrapper
from ray.data import ActorPoolStrategy

# Compile model
jit_model = torch.jit.trace(model, example_input)

# Wrap for actors
actor_wrapper = JITActorWrapper(jit_model, device="cuda:0")

class InferenceActor:
    def __init__(self, actor_wrapper):
        self.model = actor_wrapper.load()

    def __call__(self, batch):
        return self.model(batch["data"])

results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    compute=ActorPoolStrategy(size=4)
)
```

## Comparison Table

| Wrapper | Input Type | Execution | Use Case |
|---------|-----------|-----------|----------|
| TaskWrapper | nn.Module | Ray tasks | Ad-hoc inference |
| ActorWrapper | nn.Module | Ray actors | Batch processing |
| JITTaskWrapper | TorchScript | Ray tasks | Ad-hoc with compiled models |
| JITActorWrapper | TorchScript | Ray actors | Batch with compiled models |

## Common Parameters

### device

All wrappers accept a `device` parameter:

- `"cpu"` - Load model on CPU
- `"cuda:0"` - Load on first GPU
- `"cuda:1"` - Load on second GPU
- etc.

```python
wrapper = TaskWrapper(pipeline, device="cuda:0")
```

### use_fast_load

Enable experimental fast loading:

```python
wrapper = TaskWrapper(pipeline, use_fast_load=True)
```

**Note:** Fast loading is experimental and may not work with all models.

## See Also

- [User Guide: TaskWrapper](../user_guide/tasks.md)
- [User Guide: ActorWrapper](../user_guide/actors.md)
- [User Guide: TorchScript](../user_guide/torchscript.md)
- [Tutorials](../tutorials/index.md)
