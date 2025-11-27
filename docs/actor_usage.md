# Zero-Copy Model Loading for Ray Data Actors

## The Problem with `rewrite_pipeline` for Ray Data

The existing `rewrite_pipeline` in `invoke.py` creates **nested remote tasks** when used inside Ray Data actors:

```
Ray Data Actor → calls model → spawns Ray Task → waits for result
```

This adds unnecessary overhead for batch processing workloads.

## The Solution: `actor.py` Module

The new `actor` module enables **direct zero-copy loading inside actors**:

```
Ray Data Actor → loads model once (zero-copy) → runs inference locally
```

## Quick Start

```python
import ray
from ray.data import ActorPoolStrategy
from ray_zerocopy.actor import prepare_model_for_actors, load_model_in_actor

# 1. Prepare model once
model = YourPyTorchModel()
model_ref = prepare_model_for_actors(model)

# 2. Define actor class
class InferenceActor:
    def __init__(self, model_ref, device="cuda:0"):
        # Zero-copy load in each actor
        self.model = load_model_in_actor(model_ref, device=device)

    def __call__(self, batch):
        inputs = torch.tensor(batch["data"])
        with torch.no_grad():
            outputs = self.model(inputs)
        return {"predictions": outputs.cpu().numpy()}

# 3. Use with Ray Data
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_ref": model_ref, "device": "cuda:0"},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),  # 4 actors share model via zero-copy
    num_gpus=1
)
```

## API Functions

### `prepare_model_for_actors(model)`
Stores model in Ray object store for zero-copy sharing across actors.

### `load_model_in_actor(model_ref, device=None, use_fast_load=False)`
Loads model inside an actor using zero-copy from object store.

### `rewrite_pipeline_for_actors(pipeline, ...)`
For pipelines with multiple models - returns `(skeleton, model_refs)`.

### `load_pipeline_in_actor(skeleton, model_refs, device=None, ...)`
Reconstructs pipeline with all models loaded via zero-copy.

## Key Patterns

### Pattern 1: Single Model

```python
model_ref = prepare_model_for_actors(model)

class Actor:
    def __init__(self, model_ref, device="cuda:0"):
        self.model = load_model_in_actor(model_ref, device=device)

    def __call__(self, batch):
        return self.model(preprocess(batch))

ds.map_batches(
    Actor,
    fn_constructor_kwargs={"model_ref": model_ref, "device": "cuda:0"},
    compute=ActorPoolStrategy(size=4),
    num_gpus=1
)
```

### Pattern 2: Pipeline with Multiple Models

```python
class Pipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, x):
        return self.decoder(self.encoder(x))

skeleton, refs = rewrite_pipeline_for_actors(Pipeline())

class Actor:
    def __init__(self, skeleton, refs, device="cuda:0"):
        self.pipeline = load_pipeline_in_actor(skeleton, refs, device=device)

    def __call__(self, batch):
        return self.pipeline(preprocess(batch))

ds.map_batches(
    Actor,
    fn_constructor_kwargs={"skeleton": skeleton, "refs": refs, "device": "cuda:0"},
    compute=ActorPoolStrategy(size=4),
    num_gpus=1
)
```

## Memory Savings

**Without zero-copy** (traditional approach):
- 4 actors × 5GB model = 20GB total

**With zero-copy** (this module):
- Object store: 5GB (shared)
- 4 actors: minimal overhead each
- Total: ~5GB

## When to Use This vs `invoke.rewrite_pipeline`

| Use `actor.py` | Use `invoke.py` |
|----------------|-----------------|
| Ray Data `map_batches` with actors | Ad-hoc task-based inference |
| Batch processing workloads | Sporadic inference calls |
| Need max throughput | Need automatic load balancing |
| GPU pinning required | Stateless tasks preferred |

## Complete Example

See `examples/ray_data_actor_example.py` for runnable examples with:
- Simple model inference
- Multi-model pipelines
- Pre/post processing
- GPU inference

## Key Difference

**Old way** (creates nested tasks):
```python
from ray_zerocopy.invoke import rewrite_pipeline
# Creates tasks from within actors - overhead!
```

**New way** (loads directly in actor):
```python
from ray_zerocopy.actor import prepare_model_for_actors, load_model_in_actor
# No nested tasks - just local inference!
```
