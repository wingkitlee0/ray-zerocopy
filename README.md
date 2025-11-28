# ray-zerocopy

**Zero-copy model loading for PyTorch in Ray**

This library enables efficient model loading across Ray workers using zero-copy mechanisms, eliminating the need to duplicate large model weights in memory.

## Features

- 🚀 **Zero-copy loading** - Share model weights across Ray workers without duplication
- 🎯 **Ray Data integration** - Optimized for `map_batches` with `ActorPoolStrategy`
- 💾 **Memory efficient** - 4 actors with 5GB model = ~5GB total (not 20GB)
- ⚡ **High throughput** - Direct inference in actors (no task spawning overhead)
- 🎮 **GPU support** - Pin models to specific GPUs for maximum performance

## Quick Start

### For Ray Data Actors (Recommended)

```python
from ray.data import ActorPoolStrategy
from ray_zerocopy.actor import prepare_model_for_actors, load_model_in_actor

# 1. Prepare model for zero-copy sharing
model = YourPyTorchModel()
model_ref = prepare_model_for_actors(model)

# 2. Define actor that loads model with zero-copy
class InferenceActor:
    def __init__(self, model_ref, device="cuda:0"):
        self.model = load_model_in_actor(model_ref, device=device)

    def __call__(self, batch):
        with torch.no_grad():
            return self.model(batch["data"])

# 3. Use with Ray Data
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_ref": model_ref, "device": "cuda:0"},
    compute=ActorPoolStrategy(size=4),  # 4 actors share the model
    num_gpus=1
)
```

### For Task-Based Inference

```python
from ray_zerocopy.invoke import rewrite_pipeline

pipeline = YourPipeline()
rewritten = rewrite_pipeline(pipeline)

# Each call spawns a Ray task that loads the model with zero-copy
result = rewritten.model(data)
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
| Ray Data `map_batches` with actors | `actor.prepare_model_for_actors()` |
| High-throughput batch inference | `actor.prepare_model_for_actors()` |
| GPU-pinned inference | `actor.prepare_model_for_actors()` |
| Ad-hoc task-based inference | `invoke.rewrite_pipeline()` |
| Sporadic inference calls | `invoke.rewrite_pipeline()` |

## Documentation

- **[Actor Usage Guide](docs/actor_usage.md)** - Complete guide for Ray Data actors
- **[Comparison](docs/comparison.md)** - Detailed comparison of approaches
- **[Examples](examples/ray_data_actor_example.py)** - Working code examples

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
Object Store: 5GB (shared)
Actor 1-4: reference object store
Total: ~5GB
```

## API Overview

### Actor-Based (for Ray Data)

```python
# Prepare model for actors
model_ref = prepare_model_for_actors(model)

# Load in actor
model = load_model_in_actor(model_ref, device="cuda:0")

# For pipelines with multiple models
skeleton, refs = rewrite_pipeline_for_actors(pipeline)
pipeline = load_pipeline_in_actor(skeleton, refs, device="cuda:0")
```

### Task-Based (for ad-hoc inference)

```python
# Rewrite pipeline to use Ray tasks
rewritten = rewrite_pipeline(pipeline)

# Direct task invocation
result = call_model.remote(model_ref, args, kwargs)
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
