# Getting Started

## Installation

### From PyPI

```bash
pip install ray-zerocopy
```

### From Source

```bash
git clone https://github.com/yourusername/ray-zerocopy.git
cd ray-zerocopy
pip install -e .
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- Ray 2.43+
- NumPy

## Quick Start

### Actor Mode (Recommended for Batch Inference)

```python
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper

# Wrap your model
model = YourModel()
model.eval()
model_wrapper = ModelWrapper.from_model(model, mode="actor")

# Define actor
class InferenceActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()

    def __call__(self, batch):
        with torch.no_grad():
            return self.model(batch["data"])

# Use with Ray Data
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    compute=ActorPoolStrategy(size=4),
)
```

### Task Mode (Ad-hoc Inference)

```python
from ray_zerocopy import ModelWrapper

model = YourModel()
model.eval()
wrapped = ModelWrapper.for_tasks(model)

# Use immediately
result = wrapped(data)
```

## Next Steps

- Read the [ModelWrapper Guide](model_wrapper_guide.md) for detailed examples and usage
- See [JIT Wrappers](jit_wrappers.md) for TorchScript support (under development)
- Check the [API Reference](api_reference/index.md) for complete API documentation
