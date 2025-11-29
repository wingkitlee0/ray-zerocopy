# Tutorial: Multi-Model Pipeline

This tutorial demonstrates using ray-zerocopy with pipelines containing multiple models, such as encoder-decoder architectures.

## Goal

Share a complex pipeline with multiple models across actors using zero-copy, maintaining the full pipeline structure.

## What is a Pipeline?

A **pipeline** is any Python class with `nn.Module` attributes. ray-zerocopy automatically detects and shares all models:

```python
class MyPipeline:
    def __init__(self):
        self.encoder = EncoderModel()  # nn.Module - shared
        self.decoder = DecoderModel()  # nn.Module - shared
        self.config = {"temp": 1.0}    # Regular attribute - copied
```

## Complete Code

```python
import ray
import torch
import torch.nn as nn
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper

# Step 1: Define model components
class Encoder(nn.Module):
    """Encoder that compresses input."""
    def __init__(self, input_dim=128, latent_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

class Decoder(nn.Module):
    """Decoder that processes encoded features."""
    def __init__(self, latent_dim=64, output_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.ReLU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, x):
        return self.network(x)

# Step 2: Create pipeline class
class EncoderDecoderPipeline:
    """Pipeline combining encoder and decoder."""
    def __init__(self, input_dim=128, latent_dim=64, output_dim=10):
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

        # Non-model attributes are preserved
        self.temperature = 1.0

    def __call__(self, x):
        # Encode
        latent = self.encoder(x)
        # Decode
        output = self.decoder(latent)
        # Apply temperature scaling
        return output / self.temperature

    def set_eval(self):
        self.encoder.eval()
        self.decoder.eval()

# Step 3: Create and wrap the pipeline
pipeline = EncoderDecoderPipeline(input_dim=128, latent_dim=64, output_dim=10)
pipeline.set_eval()

# Wrap entire pipeline
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# Step 4: Define inference actor
class PipelineInferenceActor:
    def __init__(self, model_wrapper):
        # Load complete pipeline (both models via zero-copy, on CPU)
        self.pipeline = model_wrapper.load()
        print("Actor initialized with pipeline")

    def __call__(self, batch):
        inputs = torch.tensor(batch["data"], dtype=torch.float32)

        with torch.no_grad():
            outputs = self.pipeline(inputs)

        return {"predictions": outputs.numpy()}

# Step 5: Create dataset
import numpy as np

num_samples = 500
data = [{"data": np.random.randn(128).tolist()} for _ in range(num_samples)]
ds = ray.data.from_items(data)

# Step 6: Run inference with 3 actors
results = ds.map_batches(
    PipelineInferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=25,
    compute=ActorPoolStrategy(size=3),
)

# Step 7: Collect results
print(f"Processed {results.count()} samples")
first_batch = results.take_batch(1)
print(f"Prediction shape: {first_batch['predictions'].shape}")
```

## Step-by-Step Explanation

### Step 1: Define Model Components

Create separate models as `nn.Module` classes:

```python
class Encoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 96),
            nn.ReLU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, x):
        return self.network(x)
```

Each model is independent and can be tested separately.

### Step 2: Create Pipeline Class

Combine models into a pipeline:

```python
class EncoderDecoderPipeline:
    def __init__(self, input_dim=128, latent_dim=64, output_dim=10):
        # Models (will be zero-copy shared)
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

        # Regular attributes (will be copied)
        self.temperature = 1.0

    def __call__(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output / self.temperature
```

Key points:
- `self.encoder` and `self.decoder` are `nn.Module` instances - **zero-copy shared**
- `self.temperature` is a regular Python value - **copied**
- The pipeline orchestrates the flow between models

### Step 3: Wrap the Pipeline

Wrap the entire pipeline with `ModelWrapper`:

```python
pipeline = EncoderDecoderPipeline()
pipeline.set_eval()

model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")
```

ray-zerocopy automatically:
- Detects both `encoder` and `decoder` models
- Stores both in Ray's object store
- Preserves the pipeline structure with `temperature`

### Step 4: Define Inference Actor

The actor loads the complete pipeline:

```python
class PipelineInferenceActor:
    def __init__(self, model_wrapper):
        # Loads both encoder and decoder via zero-copy
        self.pipeline = model_wrapper.load()

    def __call__(self, batch):
        inputs = torch.tensor(batch["data"], dtype=torch.float32)

        with torch.no_grad():
            outputs = self.pipeline(inputs)

        return {"predictions": outputs.numpy()}
```

When `.load()` is called:
1. Encoder is loaded from object store (zero-copy)
2. Decoder is loaded from object store (zero-copy)
3. Pipeline structure is reconstructed
4. Temperature value is restored

### Step 5-7: Run Inference

Same as the basic tutorial:

```python
ds = ray.data.from_items(data)

results = ds.map_batches(
    PipelineInferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=25,
    compute=ActorPoolStrategy(size=3),
)

print(f"Processed {results.count()} samples")
```

## Memory Savings

**Without zero-copy:**
- Encoder: 50MB × 3 actors = 150MB
- Decoder: 30MB × 3 actors = 90MB
- **Total: 240MB**

**With zero-copy:**
- Encoder: 50MB (shared)
- Decoder: 30MB (shared)
- **Total: ~80MB**

**67% memory reduction!**

## More Complex Pipelines

### Nested Models

Pipelines can have nested structure:

```python
class ComplexPipeline:
    def __init__(self):
        # Direct models
        self.preprocessor = PreprocessorModel()

        # Nested models in dict
        self.encoders = {
            "text": TextEncoder(),
            "image": ImageEncoder()
        }

        # Nested models in list
        self.decoders = [
            Decoder1(),
            Decoder2(),
            Decoder3()
        ]

    def __call__(self, x):
        x = self.preprocessor(x)

        # Use multiple encoders
        text_enc = self.encoders["text"](x["text"])
        image_enc = self.encoders["image"](x["image"])

        # Use multiple decoders
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(text_enc + image_enc))

        return outputs
```

ray-zerocopy automatically finds ALL `nn.Module` instances, regardless of nesting!

### Stateful Pipelines

Add preprocessing or stateful components:

```python
class StatefulPipeline:
    def __init__(self):
        self.model = MyModel()

        # These are preserved but not zero-copy shared
        self.preprocessor = Preprocessor()  # Custom class
        self.config = {"threshold": 0.5}
        self.stats = {"calls": 0}

    def __call__(self, x):
        self.stats["calls"] += 1
        x = self.preprocessor(x)
        return self.model(x)
```

Each actor gets its own copy of `preprocessor`, `config`, and `stats`, but shares the model weights.

## GPU Inference

Works the same as basic tutorial:

```python
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# Update actor to move to GPU after loading
class PipelineInferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load()

results = ds.map_batches(
    PipelineInferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=25,
    compute=ActorPoolStrategy(size=1),
    num_gpus=1
)
```

Both encoder and decoder are loaded to GPU.

## Real-World Example: Text Processing

Here's a realistic text processing pipeline:

```python
class TextProcessingPipeline:
    def __init__(self):
        # Pretrained models
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 2)  # Binary classification

    def __call__(self, texts):
        # Tokenize (not zero-copy shared, but lightweight)
        tokens = self.tokenizer(texts, padding=True, return_tensors="pt")

        # Encode (zero-copy shared - large model)
        with torch.no_grad():
            embeddings = self.bert(**tokens).last_hidden_state[:, 0, :]

        # Classify (zero-copy shared)
        logits = self.classifier(embeddings)
        return torch.softmax(logits, dim=1)

# Use with ray-zerocopy
pipeline = TextProcessingPipeline()
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# In actor:
# self.pipeline = model_wrapper.load()
# The BERT model (400MB+) and classifier are zero-copy shared
# The tokenizer is copied to each actor (small overhead)
```

## Next Steps

- Learn about [Ray Data Integration](../user_guide/ray_data_integration.md)
- See [TorchScript Support](../user_guide/torchscript.md) for compiled pipelines
- Check the [API Reference](../api_reference/model_wrappers.md) for all options
