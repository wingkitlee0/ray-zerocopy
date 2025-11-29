# TorchScript Support

ray-zerocopy supports TorchScript (compiled PyTorch models) via `JITTaskWrapper` and `JITActorWrapper`.

## What is TorchScript?

TorchScript is PyTorch's way to compile models for optimized inference. Compiled models:
- Run faster (optimized execution)
- Are portable (can run without Python)
- Are serializable (save/load easily)

Learn more: [PyTorch TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)

## JITTaskWrapper

Use for TorchScript models with Ray tasks.

### Basic Usage

```python
import torch
from ray_zerocopy import JITTaskWrapper

# 1. Define your model
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# 2. Compile to TorchScript
model = MyModel()
example_input = torch.randn(1, 10)
jit_model = torch.jit.trace(model, example_input)

# 3. Wrap with JITTaskWrapper
wrapped = JITTaskWrapper(jit_model, device="cuda:0")

# 4. Use like normal
result = wrapped(data)  # Runs in Ray task with zero-copy
```

### Tracing vs Scripting

TorchScript supports two compilation modes:

**Tracing** (recommended for most cases):
```python
jit_model = torch.jit.trace(model, example_input)
```

**Scripting** (for control flow):
```python
jit_model = torch.jit.script(model)
```

Both work with ray-zerocopy.

## JITActorWrapper

Use for TorchScript models with Ray actors and Ray Data.

### Basic Usage

```python
import torch
from ray.data import ActorPoolStrategy
from ray_zerocopy import JITActorWrapper

# 1. Compile model
model = MyModel()
jit_model = torch.jit.trace(model, example_input)

# 2. Create wrapper
actor_wrapper = JITActorWrapper(jit_model, device="cuda:0")

# 3. Define actor
class InferenceActor:
    def __init__(self, actor_wrapper):
        self.model = actor_wrapper.load()

    def __call__(self, batch):
        inputs = torch.tensor(batch["data"])
        with torch.no_grad():
            outputs = self.model(inputs)
        return {"predictions": outputs.cpu().numpy()}

# 4. Use with Ray Data
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    compute=ActorPoolStrategy(size=4),
    num_gpus=1
)
```

## TorchScript Pipelines

You can compile entire pipelines:

```python
class Pipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Compile the entire pipeline
pipeline = Pipeline()
jit_pipeline = torch.jit.trace(pipeline, example_input)

# Use with actors
actor_wrapper = JITActorWrapper(jit_pipeline, device="cuda:0")
```

## Why Use TorchScript with Zero-Copy?

Combining TorchScript with zero-copy gives you:

1. **Faster inference** - TorchScript optimizations
2. **Memory efficiency** - Zero-copy sharing
3. **Production ready** - Compiled models are stable

### Performance Example

```python
# Regular nn.Module with zero-copy
actor_wrapper = ActorWrapper(model, device="cuda:0")

# TorchScript with zero-copy (faster inference!)
jit_model = torch.jit.trace(model, example_input)
jit_wrapper = JITActorWrapper(jit_model, device="cuda:0")
```

Both save memory via zero-copy, but JITActorWrapper gives faster inference.

## Limitations

### 1. Must Be Compilable

Not all models can be traced/scripted. Check PyTorch's TorchScript documentation for compatibility.

### 2. Static Shapes (Tracing)

When using `torch.jit.trace`, input shapes must match:

```python
# Traced with shape (1, 10)
jit_model = torch.jit.trace(model, torch.randn(1, 10))

# ✅ Works: Same shape
jit_model(torch.randn(1, 10))

# ❌ May fail: Different shape
jit_model(torch.randn(2, 10))
```

Use `torch.jit.script` for dynamic shapes, or trace with multiple examples.

### 3. No Python Dependencies

TorchScript models can't use arbitrary Python code. Only supported PyTorch operations.

## Complete Example

Here's a full example with Ray Data:

```python
import torch
import ray
from ray.data import ActorPoolStrategy
from ray_zerocopy import JITActorWrapper

# Define model
class ResNetClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

    def forward(self, x):
        return self.resnet(x)

# Compile model
model = ResNetClassifier()
model.eval()
example = torch.randn(1, 3, 224, 224)
jit_model = torch.jit.trace(model, example)

# Create wrapper
actor_wrapper = JITActorWrapper(jit_model, device="cuda:0")

# Define actor
class ImageClassifier:
    def __init__(self, actor_wrapper):
        self.model = actor_wrapper.load()

    def __call__(self, batch):
        # Preprocess
        images = torch.stack([preprocess(img) for img in batch["image"]])
        images = images.to("cuda:0")

        # Inference
        with torch.no_grad():
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)

        # Return top-5 predictions
        top5 = torch.topk(probs, 5, dim=1)
        return {
            "top5_classes": top5.indices.cpu().numpy(),
            "top5_probs": top5.values.cpu().numpy()
        }

# Run inference
ds = ray.data.read_images("s3://my-bucket/images/")
results = ds.map_batches(
    ImageClassifier,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),
    num_gpus=1
)

results.write_parquet("s3://my-bucket/results/")
```

## Comparison Table

| Feature | nn.Module Wrappers | JIT Wrappers |
|---------|-------------------|--------------|
| Input type | `nn.Module` | `ScriptModule` |
| Inference speed | Normal | Faster (compiled) |
| Memory usage | Same (both use zero-copy) | Same (both use zero-copy) |
| Flexibility | More flexible | Less flexible |
| Production | Good | Better |
| Setup | Simple | Requires compilation |

## Best Practices

1. **Profile first** - Measure speedup before committing to TorchScript
2. **Test compilation** - Ensure model compiles correctly
3. **Validate outputs** - Compare JIT vs non-JIT outputs
4. **Use eval mode** - Set `model.eval()` before tracing
5. **Consider shapes** - Use scripting for dynamic shapes

## Next Steps

- See [PyTorch TorchScript Tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- Check [API Reference](../api_reference/wrappers.md) for JIT wrapper options
- Try [Tutorials](../tutorials/index.md) with TorchScript models
