# ray-zerocopy

**Zero-copy model sharing for PyTorch inference in Ray**

This library enables efficient model sharing across Ray workers using zero-copy mechanisms, eliminating the need to duplicate large model weights in memory when performing inference.

## Features

- 🚀 **Zero-copy sharing** - Share model weights across Ray workers without duplication
- 🎯 **Flexible inference** - Use with Ray Tasks, Ray Actors, or Ray Data Actor UDFs
- 💾 **Memory efficient** - 4 actors with 5GB model = ~5GB total (not 20GB)
- ⚡ **High throughput** - Direct inference without model loading overhead
- 🔧 **Pipeline support** - Share entire pipelines (classes with `nn.Module` attributes)

## Quick Start

### For Ray Data Actor UDFs (Recommended for Batch Inference)

```python
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper

# 1. Create your pipeline (a class with nn.Module attributes)
class MyPipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, data):
        encoded = self.encoder(data)
        return self.decoder(encoded)

pipeline = MyPipeline()

# 2. Wrap with ModelWrapper for zero-copy sharing
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# 3. Define actor UDF that loads the pipeline
class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load()

    def __call__(self, batch):
        with torch.no_grad():
            return self.pipeline(batch["data"])

# 4. Use with Ray Data
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    compute=ActorPoolStrategy(size=4),  # 4 actors share the model
)
```

### For Ray Actors (General Purpose)

```python
import ray
from ray_zerocopy import ModelWrapper

# Wrap pipeline for actors
pipeline = MyPipeline()
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# Define inference actor
@ray.remote
class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load()

    def predict(self, data):
        with torch.no_grad():
            return self.pipeline(data)

# Create actors that share the model
actors = [InferenceActor.remote(model_wrapper) for _ in range(4)]
results = ray.get([actor.predict.remote(data) for actor in actors])
```

### For Ray Tasks (Ad-hoc Inference)

```python
from ray_zerocopy import ModelWrapper

# A Pipeline is a class with nn.Module attributes
class MyPipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, data):
        encoded = self.encoder(data)
        return self.decoder(encoded)

pipeline = MyPipeline()
wrapped = ModelWrapper.for_tasks(pipeline)

# Each call spawns a Ray task with zero-copy model loading
result = wrapped(data)
```

## Installation

```bash
pip install ray-zerocopy
```

Or install from source:

```bash
git clone https://github.com/yourusername/ray-zerocopy.git
cd ray-zerocopy
pip install -e .
```

## When to Use What

| Scenario | Use This |
|----------|----------|
| Ray Data `map_batches` batch inference | `ModelWrapper.from_model(..., mode="actor")` with Ray Data Actor UDF |
| High-throughput batch inference | `ModelWrapper.from_model(..., mode="actor")` with Ray Data Actor UDF |
| Long-running inference service | `ModelWrapper.from_model(..., mode="actor")` with Ray Actor |
| Ad-hoc task-based inference | `ModelWrapper.for_tasks()` with Ray Task |
| Sporadic inference calls | `ModelWrapper.for_tasks()` with Ray Task |

## Memory Savings Example

**Without zero-copy:**
```
Actor 1: 5GB model
Actor 2: 5GB model
Actor 3: 5GB model
Actor 4: 5GB model
Total: 20GB
```

**With zero-copy:**
```
Ray Object Store: 5GB (shared)
Actor 1-4: reference object store
Total: ~5GB
```

## Pipelines

A **Pipeline** is a class with `nn.Module` attributes. The library automatically identifies and shares all models in a pipeline:

```python
class MyPipeline:
    def __init__(self):
        self.feature_extractor = FeatureExtractorModel()
        self.classifier = ClassifierModel()
        self.config = {"threshold": 0.5}  # Non-model attributes are preserved

    def __call__(self, data):
        features = self.feature_extractor(data)
        return self.classifier(features)

# For Ray Actors and Ray Data
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")
# ... in actor:
# self.pipeline = model_wrapper.load()

# For Ray Tasks
wrapped = ModelWrapper.for_tasks(pipeline)
```

The library automatically identifies `nn.Module` attributes and applies zero-copy sharing to them, while preserving other attributes like config dictionaries.

## API Overview

### Wrapper Classes

```python
from ray_zerocopy import ModelWrapper

# ModelWrapper - For Ray Tasks
wrapped = ModelWrapper.for_tasks(pipeline)
result = wrapped(data)  # Runs in Ray task with zero-copy

# ModelWrapper - For Ray Actors and Ray Data
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")
# ... in actor __init__:
pipeline = model_wrapper.load()  # Load with zero-copy in actor
```

### TorchScript Support

```python
from ray_zerocopy import JITTaskWrapper, JITActorWrapper

# JITTaskWrapper - For TorchScript models with Ray Tasks
jit_pipeline = torch.jit.trace(pipeline, example_input)
wrapped = JITTaskWrapper(jit_pipeline)

# JITActorWrapper - For TorchScript models with Ray Actors
actor_wrapper = JITActorWrapper(jit_pipeline)
```

## Requirements

- Python 3.8+
- PyTorch
- Ray

## Origin

Based on [project-codeflare/zero-copy-model-loading](https://github.com/project-codeflare/zero-copy-model-loading)

## License

Apache License 2.0 (see [LICENSE](LICENSE))

## Contributing

Contributions welcome! Please open an issue or PR.
