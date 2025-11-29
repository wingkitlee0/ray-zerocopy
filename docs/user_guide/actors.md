# ActorWrapper Usage

`ActorWrapper` enables zero-copy model loading inside Ray actors. This is ideal for Ray Data batch processing and long-running inference services.

## Why ActorWrapper?

The key difference from `TaskWrapper`:

- **TaskWrapper**: Creates nested remote tasks (overhead for batch processing)
  ```
  Ray Data Actor → calls model → spawns Ray Task → waits for result
  ```

- **ActorWrapper**: Direct zero-copy loading inside actors
  ```
  Ray Data Actor → loads model once (zero-copy) → runs inference locally
  ```

## Basic Usage

```python
from ray_zerocopy import ActorWrapper

# 1. Wrap your pipeline
class MyPipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, data):
        return self.decoder(self.encoder(data))

pipeline = MyPipeline()
actor_wrapper = ActorWrapper(pipeline, device="cuda:0")

# 2. Load in actor
class InferenceActor:
    def __init__(self, actor_wrapper):
        # Zero-copy load - happens once per actor
        self.pipeline = actor_wrapper.load()

    def __call__(self, batch):
        # Direct inference - no remote calls
        return self.pipeline(batch["data"])
```

## Ray Data Integration

Perfect for `map_batches` with Ray Data:

```python
import ray
from ray.data import ActorPoolStrategy
from ray_zerocopy import ActorWrapper

# Prepare wrapper
pipeline = MyPipeline()
actor_wrapper = ActorWrapper(pipeline, device="cuda:0")

# Define actor class
class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()

    def __call__(self, batch):
        inputs = torch.tensor(batch["data"])
        with torch.no_grad():
            outputs = self.pipeline(inputs)
        return {"predictions": outputs.cpu().numpy()}

# Use with Ray Data
ds = ray.data.read_parquet("s3://my-data/")
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),  # 4 actors share model via zero-copy
    num_gpus=1
)
```

## Memory Savings

**Without zero-copy** (traditional approach):
- 4 actors × 5GB model = 20GB total

**With ActorWrapper**:
- Object store: 5GB (shared)
- 4 actors: minimal overhead each
- Total: ~5GB

This is a **4x memory reduction** for 4 actors!

## Use Cases

### Pattern 1: Single Model

```python
from ray_zerocopy import ActorWrapper

model = YourModel()
actor_wrapper = ActorWrapper(model, device="cuda:0")

class Actor:
    def __init__(self, actor_wrapper):
        self.model = actor_wrapper.load()

    def __call__(self, batch):
        return self.model(preprocess(batch))

ds.map_batches(
    Actor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    compute=ActorPoolStrategy(size=4),
    num_gpus=1
)
```

### Pattern 2: Multi-Model Pipeline

```python
class Pipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, x):
        return self.decoder(self.encoder(x))

pipeline = Pipeline()
actor_wrapper = ActorWrapper(pipeline, device="cuda:0")

class Actor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()

    def __call__(self, batch):
        return self.pipeline(preprocess(batch))

ds.map_batches(
    Actor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    compute=ActorPoolStrategy(size=4),
    num_gpus=1
)
```

### Pattern 3: Preprocessing + Inference + Postprocessing

```python
class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()
        self.preprocessor = Preprocessor()
        self.postprocessor = Postprocessor()

    def __call__(self, batch):
        # Preprocess
        inputs = self.preprocessor(batch["raw_data"])

        # Inference (zero-copy model)
        with torch.no_grad():
            outputs = self.pipeline(inputs)

        # Postprocess
        return self.postprocessor(outputs)

ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    compute=ActorPoolStrategy(size=4)
)
```

## Ray Actors (Non-Ray Data)

You can also use `ActorWrapper` with plain Ray actors:

```python
import ray
from ray_zerocopy import ActorWrapper

actor_wrapper = ActorWrapper(pipeline, device="cuda:0")

@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()

    def predict(self, data):
        return self.pipeline(data)

# Create actors
actors = [InferenceActor.remote(actor_wrapper) for _ in range(4)]

# Use actors
results = ray.get([actor.predict.remote(data) for actor in actors])
```

## Device Placement

Specify which device models should load on:

```python
# CPU
actor_wrapper = ActorWrapper(pipeline, device="cpu")

# First GPU
actor_wrapper = ActorWrapper(pipeline, device="cuda:0")

# Second GPU
actor_wrapper = ActorWrapper(pipeline, device="cuda:1")
```

The device is applied when `.load()` is called in the actor.

## Configuration Options

### Fast Loading

Enable experimental fast loading:

```python
actor_wrapper = ActorWrapper(pipeline, device="cuda:0", use_fast_load=True)
```

### Custom Loading

You can also manually control loading:

```python
class InferenceActor:
    def __init__(self, actor_wrapper):
        # Load to different device than wrapper specifies
        self.pipeline = actor_wrapper.load(device="cuda:1")
```

## Performance Tips

1. **Load once per actor** - Call `.load()` only in `__init__`, not in `__call__`
2. **Use actor pools** - `ActorPoolStrategy(size=N)` for parallelism
3. **Match batch size to GPU** - Larger batches = better GPU utilization
4. **Pin to GPUs** - Use `num_gpus=1` in Ray Data or actor decorator

## When to Use ActorWrapper

| Use ActorWrapper | Use TaskWrapper |
|------------------|-----------------|
| Ray Data `map_batches` with actors | Ad-hoc task-based inference |
| Batch processing workloads | Sporadic inference calls |
| Need max throughput | Need automatic load balancing |
| GPU pinning required | Stateless tasks preferred |
| Long-running inference service | One-off predictions |

## Complete Example

Here's a full Ray Data pipeline with preprocessing, inference, and postprocessing:

```python
import ray
from ray.data import ActorPoolStrategy
from ray_zerocopy import ActorWrapper
import torch

# Define pipeline
class MyPipeline:
    def __init__(self):
        self.model = MyModel()

    def __call__(self, x):
        return self.model(x)

# Create wrapper
pipeline = MyPipeline()
actor_wrapper = ActorWrapper(pipeline, device="cuda:0")

# Define actor
class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()

    def __call__(self, batch):
        # Preprocess
        inputs = torch.tensor(batch["features"]).to("cuda:0")

        # Inference
        with torch.no_grad():
            outputs = self.pipeline(inputs)

        # Postprocess
        return {
            "predictions": outputs.cpu().numpy(),
            "batch_size": len(inputs)
        }

# Load data
ds = ray.data.read_parquet("s3://my-bucket/data/")

# Run inference
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),
    num_gpus=1
)

# Write results
results.write_parquet("s3://my-bucket/results/")
```

## Next Steps

- See [Ray Data Integration](ray_data_integration.md) for more patterns
- Check [Tutorials](../tutorials/index.md) for complete examples
- Read [API Reference](../api_reference/wrappers.md) for all options
