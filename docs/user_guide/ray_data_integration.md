# Ray Data Integration

ray-zerocopy is designed to work seamlessly with Ray Data for scalable batch inference.

## Why Ray Data + Zero-Copy?

Ray Data provides:
- Distributed data loading
- Batch processing
- Resource management
- Fault tolerance

ray-zerocopy adds:
- Memory-efficient model sharing
- Zero-copy model loading
- Minimal overhead

Together, they enable large-scale inference with minimal memory footprint.

## Basic Pattern

```python
import ray
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper

# 1. Load data
ds = ray.data.read_parquet("s3://my-data/")

# 2. Prepare model wrapper
pipeline = MyPipeline()
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# 3. Define actor
class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load()

    def __call__(self, batch):
        return self.pipeline(batch["data"])

# 4. Run inference
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    compute=ActorPoolStrategy(size=4),
)

# 5. Write results
results.write_parquet("s3://my-results/")
```

## ActorPoolStrategy

Use `ActorPoolStrategy` to control parallelism:

```python
from ray.data import ActorPoolStrategy

# 4 actors sharing the model
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    compute=ActorPoolStrategy(size=4),
)
```

**Memory usage**: 4 actors × 5GB model = ~5GB (not 20GB!)

## Batch Size Configuration

Control how much data each actor processes:

```python
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=32,  # Process 32 items at a time
    compute=ActorPoolStrategy(size=4)
)
```

**Rule of thumb:**
- Larger batch = better throughput
- Smaller batch = more frequent checkpoints
- Start with 32-128 and tune

## Data Sources

### Reading Data

Ray Data supports many formats:

```python
# Parquet
ds = ray.data.read_parquet("s3://bucket/data.parquet")

# CSV
ds = ray.data.read_csv("s3://bucket/data.csv")

# JSON
ds = ray.data.read_json("s3://bucket/data.json")

# Images
ds = ray.data.read_images("s3://bucket/images/")

# Binary files
ds = ray.data.read_binary_files("s3://bucket/files/")

# From Python list
ds = ray.data.from_items([1, 2, 3, 4])
```

### Writing Results

```python
# Parquet (recommended for structured data)
results.write_parquet("s3://bucket/results/")

# JSON
results.write_json("s3://bucket/results/")

# CSV
results.write_csv("s3://bucket/results/")
```

## Preprocessing and Postprocessing

### In-Actor Processing

```python
class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load()
        self.preprocessor = Preprocessor()
        self.postprocessor = Postprocessor()

    def __call__(self, batch):
        # Preprocess
        inputs = self.preprocessor(batch["raw_data"])

        # Inference
        outputs = self.pipeline(inputs)

        # Postprocess
        return self.postprocessor(outputs)
```

### Separate Map Operations

```python
# Preprocess
ds = ds.map_batches(preprocess_fn)

# Inference
ds = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    compute=ActorPoolStrategy(size=4)
)

# Postprocess
ds = ds.map_batches(postprocess_fn)
```

## Error Handling

Ray Data automatically retries failed batches:

```python
class InferenceActor:
    def __call__(self, batch):
        try:
            return self.pipeline(batch["data"])
        except Exception as e:
            # Log error
            print(f"Error processing batch: {e}")
            # Return empty or error markers
            return {"error": str(e)}
```

## Performance Tuning

### 1. Batch Size

```python
# Too small: Overhead dominates
batch_size=1  # ❌

# Too large: Memory issues
batch_size=10000  # ❌

# Just right: Saturate GPU
batch_size=32-128  # ✅
```

### 2. Actor Pool Size

```python
# Too few: Underutilized
ActorPoolStrategy(size=1)  # ❌

# Too many: Memory issues
ActorPoolStrategy(size=100)  # ❌

# Match available resources
ActorPoolStrategy(size=4)  # ✅
```

### 3. Prefetch

Ray Data prefetches data by default. Increase for better pipelining:

```python
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    compute=ActorPoolStrategy(size=4),
    prefetch_batches=2  # Prefetch 2 batches
)
```

## Complete Example: Image Classification

```python
import ray
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper
import torch
from torchvision import transforms

# Define pipeline
class ImageClassifier:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.model.eval()

    def __call__(self, images):
        with torch.no_grad():
            return self.model(images)

# Create wrapper
pipeline = ImageClassifier()
model_wrapper = ModelWrapper.from_model(pipeline, mode="actor")

# Define actor with preprocessing
class InferenceActor:
    def __init__(self, model_wrapper):
        self.pipeline = model_wrapper.load()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, batch):
        # Preprocess images
        images = [self.transform(img) for img in batch["image"]]
        images = torch.stack(images)

        # Inference
        logits = self.pipeline(images)
        probs = torch.softmax(logits, dim=1)

        # Get top prediction
        top1 = torch.argmax(probs, dim=1)

        return {
            "class": top1.cpu().numpy(),
            "confidence": probs.max(dim=1).values.cpu().numpy()
        }

# Load images
ds = ray.data.read_images("s3://my-bucket/images/")

# Run inference
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),
)

# Write results
results.write_parquet("s3://my-bucket/results/")
```

## Best Practices

1. **Use ModelWrapper.from_model(..., mode="actor")** - For Ray Data
2. **Load in __init__** - Not in __call__
3. **Tune batch size** - Balance throughput and memory
4. **Profile first** - Measure before optimizing
5. **Handle errors** - Implement try/except in actors
6. **Monitor memory** - Watch for OOM errors

## Next Steps

- See [Ray Data Documentation](https://docs.ray.io/en/latest/data/data.html)
- Try [Tutorials](../tutorials/index.md) with Ray Data examples
- Check [ModelWrapper API](../api_reference/model_wrappers.md)
