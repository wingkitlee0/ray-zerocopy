# Core Concepts

## What is Zero-Copy?

Zero-copy refers to sharing data between processes without duplicating it in memory. In ray-zerocopy, this means:

1. Model weights are stored **once** in Ray's object store
2. Multiple workers **reference** the same memory location
3. No duplication = massive memory savings

## Pipelines

A **Pipeline** is any Python class that contains `torch.nn.Module` attributes. The library automatically identifies and shares all models in your pipeline.

### Example: Simple Pipeline

```python
class MyPipeline:
    def __init__(self):
        self.feature_extractor = FeatureExtractorModel()  # nn.Module
        self.classifier = ClassifierModel()  # nn.Module
        self.config = {"threshold": 0.5}  # Non-model attributes preserved

    def __call__(self, data):
        features = self.feature_extractor(data)
        return self.classifier(features)
```

**What gets shared:**
- ✅ `self.feature_extractor` (nn.Module)
- ✅ `self.classifier` (nn.Module)

**What gets copied:**
- ✅ `self.config` (regular Python object)

### Example: Complex Pipeline

```python
class ComplexPipeline:
    def __init__(self):
        # Multiple models
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()
        self.discriminator = DiscriminatorModel()

        # Nested models
        self.embeddings = {
            "text": TextEmbedding(),
            "image": ImageEmbedding()
        }

        # Non-model state
        self.preprocessing_config = {...}
        self.stats = {"calls": 0}
```

All `nn.Module` instances are automatically identified and shared via zero-copy, regardless of nesting.

## Wrapper Classes

### TaskWrapper

Wraps a pipeline for execution in Ray tasks. Each method call spawns a new Ray task with zero-copy model loading.

```python
from ray_zerocopy import TaskWrapper

pipeline = MyPipeline()
wrapped = TaskWrapper(pipeline)

# This spawns a Ray task
result = wrapped(data)
```

**When to use:**
- Ad-hoc inference calls
- Don't need persistent state
- Simple parallelism

### ActorWrapper

Wraps a pipeline for loading inside Ray actors. Actors load the model once and reuse it.

```python
from ray_zerocopy import ActorWrapper

pipeline = MyPipeline()
actor_wrapper = ActorWrapper(pipeline)

class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()  # Load once

    def __call__(self, batch):
        return self.pipeline(batch)  # Reuse loaded model
```

**When to use:**
- Ray Data `map_batches`
- Long-running inference service
- Batch processing workloads
- Need to maintain state

## Memory Model

### Without Zero-Copy

Each actor/task loads its own copy:

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Actor 1 │  │ Actor 2 │  │ Actor 3 │  │ Actor 4 │
│  5GB    │  │  5GB    │  │  5GB    │  │  5GB    │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
                Total: 20GB
```

### With Zero-Copy

All actors reference shared memory:

```
┌───────────────────────────────────────┐
│        Ray Object Store (5GB)         │
└───────────────────────────────────────┘
     ↑         ↑         ↑         ↑
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Actor 1 │  │ Actor 2 │  │ Actor 3 │  │ Actor 4 │
│  ref    │  │  ref    │  │  ref    │  │  ref    │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
                Total: ~5GB
```

## Device Placement

You can specify which device to load models on:

```python
# CPU
actor_wrapper = ActorWrapper(pipeline, device="cpu")

# GPU
actor_wrapper = ActorWrapper(pipeline, device="cuda:0")

# Specific GPU
actor_wrapper = ActorWrapper(pipeline, device="cuda:1")
```

The device is applied when calling `.load()` in the actor:

```python
class InferenceActor:
    def __init__(self, actor_wrapper):
        # Model loaded to the device specified in ActorWrapper
        self.pipeline = actor_wrapper.load()
```

## TorchScript Support

ray-zerocopy also supports TorchScript (compiled) models:

```python
from ray_zerocopy import JITActorWrapper

# Compile your model
jit_pipeline = torch.jit.trace(pipeline, example_input)

# Use with actors
actor_wrapper = JITActorWrapper(jit_pipeline, device="cuda:0")
```

See [TorchScript Guide](torchscript.md) for details.

## Performance Considerations

### When Zero-Copy Helps Most

- **Large models** (multi-GB weights)
- **Multiple workers** (more duplication avoided)
- **Limited memory** (can't fit N copies)

### When Zero-Copy Matters Less

- **Small models** (<100MB)
- **Single worker**
- **Abundant memory**

### Best Practices

1. **Use actors for batch processing** - `ActorWrapper` with Ray Data
2. **Load once, infer many** - Actors amortize loading overhead
3. **Match device to hardware** - Use `device="cuda:X"` for GPU inference
4. **Profile your workload** - Measure memory usage with/without zero-copy
