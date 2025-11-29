# Tutorial: Ray Data Batch Inference

This tutorial covers production-ready patterns for batch inference with Ray Data and ray-zerocopy.

## Overview

Ray Data is designed for scalable data processing. Combined with ray-zerocopy, you get:

- **Distributed data loading** from S3, GCS, or local storage
- **Zero-copy model sharing** across actors
- **Automatic batching** and prefetching
- **Fault tolerance** with automatic retries

## Complete Example: Image Classification

```python
import ray
import torch
import torch.nn as nn
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper
from torchvision import transforms
from PIL import Image

# Step 1: Define model
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pretrained ResNet
        self.resnet = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

    def forward(self, x):
        return self.resnet(x)

# Step 2: Wrap model
model = ImageClassifier()
model.eval()
model_wrapper = ModelWrapper.from_model(model, mode="actor")

# Step 3: Define actor with preprocessing
class InferenceActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()
        self.model = self.model.to("cuda:0")

        # Preprocessing pipeline
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
        images = []
        for img_bytes in batch["image"]:
            img = Image.open(io.BytesIO(img_bytes))
            img = self.transform(img)
            images.append(img)

        # Stack and move to GPU
        images = torch.stack(images).to("cuda:0")

        # Inference
        with torch.no_grad():
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)

        # Get top predictions
        top5_probs, top5_indices = torch.topk(probs, 5, dim=1)

        return {
            "top5_classes": top5_indices.cpu().numpy(),
            "top5_probs": top5_probs.cpu().numpy()
        }

# Step 4: Load images from S3
ds = ray.data.read_images("s3://my-bucket/images/")

# Step 5: Run inference
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),
    num_gpus=1
)

# Step 6: Write results
results.write_parquet("s3://my-bucket/results/")
print("Done!")
```

## Pattern Breakdown

### Pattern 1: Preprocessing in Actor

Preprocessing is often done inside the actor for efficiency:

```python
class InferenceActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()
        self.model = self.model.to("cuda:0")
        self.preprocessor = MyPreprocessor()  # Initialize once

    def __call__(self, batch):
        # Preprocess batch
        inputs = self.preprocessor(batch["raw_data"])

        # Inference
        outputs = self.model(inputs)

        return {"predictions": outputs}
```

**Benefits:**
- Preprocessing happens on the actor's resources
- Can use actor's GPU for preprocessing
- Reduces data transfer

### Pattern 2: Separate Preprocessing Step

For heavy preprocessing, use a separate step:

```python
# Step 1: Preprocess (CPU-intensive)
ds = ds.map_batches(preprocess_fn, batch_size=100)

# Step 2: Inference (GPU-intensive)
ds = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),
    num_gpus=1
)
```

**Benefits:**
- Separate resource allocation
- CPU preprocessing doesn't block GPU inference
- Better resource utilization

### Pattern 3: Postprocessing

Add postprocessing for clean outputs:

```python
class InferenceActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()
        self.model = self.model.to("cuda:0")

    def __call__(self, batch):
        # Inference
        logits = self.model(batch["inputs"])

        # Postprocess
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        confidence = probs.max(dim=1).values

        return {
            "prediction": predictions.cpu().numpy(),
            "confidence": confidence.cpu().numpy()
        }
```

### Pattern 4: Error Handling

Handle errors gracefully:

```python
class InferenceActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()
        self.model = self.model.to("cuda:0")
        self.error_count = 0

    def __call__(self, batch):
        try:
            inputs = self.preprocess(batch["data"])
            outputs = self.model(inputs)
            return {"predictions": outputs, "error": None}

        except Exception as e:
            self.error_count += 1
            print(f"Error in batch (total errors: {self.error_count}): {e}")

            # Return error markers
            return {
                "predictions": None,
                "error": str(e)
            }
```

## Data Loading

### From Cloud Storage

```python
# S3
ds = ray.data.read_parquet("s3://bucket/data/**/*.parquet")
ds = ray.data.read_images("s3://bucket/images/")
ds = ray.data.read_json("s3://bucket/data.json")

# GCS
ds = ray.data.read_parquet("gs://bucket/data/")

# Azure Blob
ds = ray.data.read_parquet("az://container/data/")
```

### From Local Files

```python
ds = ray.data.read_parquet("/path/to/data/")
ds = ray.data.read_images("/path/to/images/")
```

### From Python Data

```python
# From list
data = [{"id": i, "value": i * 2} for i in range(1000)]
ds = ray.data.from_items(data)

# From pandas
import pandas as pd
df = pd.read_csv("data.csv")
ds = ray.data.from_pandas(df)
```

## Performance Tuning

### Batch Size

Tune batch size for your GPU:

```python
# Too small: Underutilized GPU
batch_size=1  # ❌

# Too large: OOM errors
batch_size=1000  # ❌

# Just right: Saturate GPU without OOM
batch_size=32  # ✅ Start here, then tune
```

**How to tune:**
1. Start with 32
2. Monitor GPU utilization
3. Increase until utilization >80% or OOM
4. Back off 20%

### Actor Pool Size

Match number of GPUs:

```python
# 4 GPUs available
compute=ActorPoolStrategy(size=4)
num_gpus=1  # Each actor gets 1 GPU
```

For CPU inference:

```python
# Use more actors (no GPU constraint)
compute=ActorPoolStrategy(size=16)
```

### Prefetching

Ray Data prefetches batches automatically. Increase for better pipelining:

```python
results = ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"model_wrapper": model_wrapper},
    batch_size=32,
    compute=ActorPoolStrategy(size=4),
    num_gpus=1,
    prefetch_batches=2  # Prefetch 2 batches per actor
)
```

## Writing Results

### To Cloud Storage

```python
# Parquet (recommended for structured data)
results.write_parquet("s3://bucket/results/")

# JSON
results.write_json("s3://bucket/results/")

# CSV
results.write_csv("s3://bucket/results/")
```

### To Local Storage

```python
results.write_parquet("/path/to/results/")
```

### Partitioning

Partition large outputs:

```python
results.write_parquet(
    "s3://bucket/results/",
    num_rows_per_file=10000  # 10k rows per file
)
```

## Monitoring

### Progress Bars

Ray Data shows progress by default:

```
Map_Batches: 100%|████████| 1000/1000 [00:30<00:00, 33.33it/s]
```

Disable if needed:

```python
ctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = False
```

### Custom Logging

Add logging in actors:

```python
class InferenceActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()
        self.batch_count = 0

    def __call__(self, batch):
        self.batch_count += 1

        if self.batch_count % 10 == 0:
            print(f"Processed {self.batch_count} batches")

        return self.model(batch["data"])
```

## Complete Production Example

Here's a full production-ready pipeline:

```python
import ray
import torch
from ray.data import ActorPoolStrategy
from ray_zerocopy import ModelWrapper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model
class ProductionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Your model here
        self.model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

    def forward(self, x):
        return self.model(x)

# Wrap model
model = ProductionModel()
model.eval()
model_wrapper = ModelWrapper.from_model(model, mode="actor")

# Define production actor
class ProductionActor:
    def __init__(self, model_wrapper):
        self.model = model_wrapper.load()
        self.batch_count = 0
        self.error_count = 0
        logger.info("Actor initialized")

    def __call__(self, batch):
        self.batch_count += 1

        try:
            # Preprocess
            inputs = self.preprocess(batch["data"])

            # Inference
            with torch.no_grad():
                outputs = self.model(inputs)

            # Postprocess
            results = self.postprocess(outputs)

            # Log progress
            if self.batch_count % 100 == 0:
                logger.info(f"Processed {self.batch_count} batches, {self.error_count} errors")

            return results

        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in batch {self.batch_count}: {e}")
            return {"error": str(e)}

    def preprocess(self, data):
        # Your preprocessing logic
        return torch.tensor(data)

    def postprocess(self, outputs):
        # Your postprocessing logic
        return {"predictions": outputs.cpu().numpy()}

# Main pipeline
def main():
    # Load data
    logger.info("Loading data from S3...")
    ds = ray.data.read_parquet("s3://my-bucket/input/")
    logger.info(f"Loaded {ds.count()} samples")

    # Run inference
    logger.info("Running inference...")
    results = ds.map_batches(
        ProductionActor,
        fn_constructor_kwargs={"model_wrapper": model_wrapper},
        batch_size=32,
        compute=ActorPoolStrategy(size=4),
        num_gpus=1,
        prefetch_batches=2
    )

    # Write results
    logger.info("Writing results...")
    results.write_parquet("s3://my-bucket/output/")
    logger.info("Done!")

if __name__ == "__main__":
    ray.init()
    main()
```

## Best Practices

1. **Load model in `__init__`** - Not in `__call__`
2. **Use try/except** - Handle errors gracefully
3. **Log progress** - Track batches and errors
4. **Tune batch size** - Balance throughput and memory
5. **Match actors to GPUs** - One actor per GPU
6. **Profile first** - Measure before optimizing
7. **Partition output** - Don't create huge files
8. **Monitor memory** - Watch for OOM errors

## Next Steps

- See [User Guide](../user_guide/index.md) for more patterns
- Check [API Reference](../api_reference/index.md) for all options
- Read [Ray Data Docs](https://docs.ray.io/en/latest/data/data.html) for advanced features

## Troubleshooting

### Out of Memory (OOM)

**Problem:** GPU runs out of memory

**Solutions:**
- Reduce `batch_size`
- Use smaller model
- Clear cache: `torch.cuda.empty_cache()`
- Use gradient checkpointing

### Slow Inference

**Problem:** Low throughput

**Solutions:**
- Increase `batch_size`
- Add more actors
- Increase `prefetch_batches`
- Profile with `torch.profiler`

### Actor Crashes

**Problem:** Actors fail and restart

**Solutions:**
- Add error handling in `__call__`
- Log errors for debugging
- Check data quality
- Monitor resource usage

## Next Steps

- See [User Guide](../user_guide/index.md) for more patterns
- Check [API Reference](../api_reference/index.md) for all options
- Read [Ray Data Docs](https://docs.ray.io/en/latest/data/data.html) for advanced features
