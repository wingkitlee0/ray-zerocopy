# Ray Data Integration

ModelWrapper works seamlessly with Ray Data for batch inference. The same actor pattern works for both Ray Data `map_batches` and Ray Actors.

## Basic Pattern

```python
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper

# Wrap model
model_wrapper = ModelWrapper.from_model(model, mode="actor")

# Define actor (same pattern for Ray Data and Ray Actors)
class InferenceActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()

    def __call__(self, batch):
        return self.model(batch["data"])

# Use with Ray Data
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    compute=ActorPoolStrategy(size=4),
)
```

## Next Steps

- See the [ModelWrapper Guide](../model_wrapper_guide.md) for complete examples
- Check the [API Reference](../api_reference/index.md) for detailed documentation
