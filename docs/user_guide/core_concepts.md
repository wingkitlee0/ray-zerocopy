# Core Concepts

## What is Zero-Copy?

Zero-copy refers to sharing data between processes without duplicating it in memory. In ray-zerocopy, this means:

1. Model weights are stored **once** in Ray's object store
2. Multiple workers **reference** the same memory location via object references
3. No duplication = significant memory savings

## How It Works

When you wrap a model with `ModelWrapper`, the library:

1. Extracts model weights and stores them in Ray's object store using `ray.put()`
2. Creates a skeleton (pipeline structure without weights) that can be serialized
3. In actors, reconstructs the model by loading weights from object references using `ray.get()`

Ray's object store uses shared memory (on the same node) or efficient serialization (across nodes), enabling multiple actors to reference the same model weights without copying them.

## Memory Model

**Without zero-copy:**
- Each actor loads its own copy: 4 actors × 5GB = 20GB

**With zero-copy:**
- Model stored once in object store: 5GB
- All actors reference the same memory: ~5GB total

## Pipelines

A **Pipeline** is any Python class that contains `torch.nn.Module` attributes. ModelWrapper automatically identifies and shares all `nn.Module` instances in your pipeline via zero-copy.

Non-model attributes (like configuration dictionaries) are copied to each actor, but model weights are shared.

## Device Placement

Models are loaded on CPU by default. You can move them to any device after loading using standard PyTorch methods.

## Next Steps

- See the [ModelWrapper Guide](../model_wrapper_guide.md) for usage examples
- Check the [API Reference](../api_reference/index.md) for detailed documentation
