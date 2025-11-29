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

### ModelWrapper

Unified wrapper for both task and actor execution modes. Supports zero-copy model sharing for nn.Module models.

**Task Mode** - For ad-hoc inference with Ray tasks:

```python
from ray_zerocopy import ModelWrapper

pipeline = MyPipeline()
wrapped = ModelWrapper.for_tasks(pipeline)

# Each call spawns a Ray task
result = wrapped(data)
```

**Actor Mode** - For Ray Data and long-running actors:

```python
from ray_zerocopy import ModelWrapper

pipeline = MyPipeline()
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load(device="cuda:0")  # Load once

    def __call__(self, batch):
        return self.pipeline(batch)  # Reuse loaded model
```

**When to use:**
- **Task mode**: Ad-hoc inference calls, don't need persistent state
- **Actor mode**: Ray Data `map_batches`, long-running inference service, batch processing workloads

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

You can specify which device to load models on when calling `.load()`:

```python
# CPU
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load(device="cpu")

# GPU
class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load(device="cuda:0")

# Specific GPU
class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load(device="cuda:1")
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

1. **Use actors for batch processing** - `ModelWrapper.from_model(..., mode="actor")` with Ray Data
2. **Load once, infer many** - Actors amortize loading overhead
3. **Match device to hardware** - Use `device="cuda:X"` in `load()` for GPU inference
4. **Profile your workload** - Measure memory usage with/without zero-copy
