# ray-zerocopy

**Zero-copy model sharing for PyTorch inference in Ray**

ray-zerocopy enables efficient model sharing across Ray workers using zero-copy mechanisms, eliminating the need to duplicate large model weights in memory when performing inference.

## Features

- 🚀 **Zero-copy sharing** - Share model weights across Ray workers without duplication
- 🎯 **Flexible inference** - Use with Ray Tasks, Ray Actors, or Ray Data Actor UDFs
- 💾 **Memory efficient** - 4 actors with 5GB model = ~5GB total (not 20GB)
- ⚡ **High throughput** - Direct inference without model loading overhead
- 🔧 **Pipeline support** - Share entire pipelines (classes with `nn.Module` attributes)

## Quick Example

```python
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper

# 1. Wrap your pipeline
class MyPipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, data):
        return self.decoder(self.encoder(data))

pipeline = MyPipeline()
actor_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# 2. Use in Ray Data actors
class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.to_pipeline()

    def __call__(self, batch):
        with torch.no_grad():
            return self.pipeline(batch["data"])

# 3. Process with multiple actors sharing the model
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    compute=ActorPoolStrategy(size=4),  # 4 actors share the model
)
```

## Memory Savings

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

## Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
user_guide/index
tutorials/index
api_reference/index
```

## Origin

Based on [project-codeflare/zero-copy-model-loading](https://github.com/project-codeflare/zero-copy-model-loading)

## License

Apache License 2.0
