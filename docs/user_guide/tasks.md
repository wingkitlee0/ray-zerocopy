# TaskWrapper Usage

`TaskWrapper` enables zero-copy inference via Ray tasks. Each method call on the wrapped pipeline spawns a Ray task that loads the model using zero-copy.

## Basic Usage

```python
from ray_zerocopy import TaskWrapper

# Define your pipeline
class MyPipeline:
    def __init__(self):
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    def __call__(self, data):
        return self.decoder(self.encoder(data))

# Wrap the pipeline
pipeline = MyPipeline()
wrapped = TaskWrapper(pipeline)

# Call methods - each spawns a Ray task
result = wrapped(data)  # Runs in a Ray task with zero-copy loading
```

## How It Works

1. **Wrapping**: `TaskWrapper(pipeline)` analyzes the pipeline and stores models in Ray's object store
2. **Method calls**: When you call `wrapped(data)`, a Ray task is spawned
3. **Zero-copy loading**: The task loads models from the object store without duplication
4. **Execution**: The task runs inference and returns results

## Use Cases

### Ad-hoc Inference

Perfect for one-off or sporadic inference calls:

```python
wrapped = TaskWrapper(pipeline)

# Process individual items as they arrive
for item in data_stream:
    result = wrapped.process(item)  # Each spawns a task
    handle_result(result)
```

### Simple Parallelism

Ray automatically handles task scheduling:

```python
wrapped = TaskWrapper(pipeline)

# Process multiple items in parallel
results = [wrapped.process(item) for item in items]
# Ray schedules tasks across available workers
```

### Stateless Processing

Good when you don't need to maintain state between calls:

```python
wrapped = TaskWrapper(pipeline)

# Each call is independent
result1 = wrapped.predict(data1)
result2 = wrapped.predict(data2)  # No shared state
```

## Method Forwarding

All methods on your pipeline are available on the wrapper:

```python
class MyPipeline:
    def predict(self, data):
        return self.model(data)

    def predict_proba(self, data):
        return torch.softmax(self.model(data), dim=-1)

wrapped = TaskWrapper(pipeline)

# Both methods available
predictions = wrapped.predict(data)
probabilities = wrapped.predict_proba(data)
```

## Limitations

### Not for Batch Processing

TaskWrapper spawns a new task for each call. For batch processing, use `ActorWrapper` instead:

```python
# ❌ Inefficient: Creates many short-lived tasks
for batch in dataset:
    result = task_wrapped.process(batch)

# ✅ Efficient: Use ActorWrapper with Ray Data
actor_wrapper = ActorWrapper(pipeline)
results = ds.map_batches(InferenceActor, ...)
```

### No State Preservation

Each task is independent - state is not preserved:

```python
wrapped = TaskWrapper(pipeline)

# These run in different tasks
wrapped.set_threshold(0.5)
wrapped.predict(data)  # Does NOT see the threshold change
```

## Configuration Options

### Device Placement

Specify which device to use:

```python
# CPU inference
wrapped = TaskWrapper(pipeline, device="cpu")

# GPU inference
wrapped = TaskWrapper(pipeline, device="cuda:0")
```

### Fast Loading

Enable fast loading (experimental):

```python
wrapped = TaskWrapper(pipeline, use_fast_load=True)
```

## Comparison with ActorWrapper

| Feature | TaskWrapper | ActorWrapper |
|---------|-------------|--------------|
| Execution | Ray tasks | Ray actors |
| State | Stateless | Stateful |
| Loading | Per task | Once per actor |
| Best for | Ad-hoc calls | Batch processing |
| Overhead | Higher | Lower (amortized) |
| Ray Data | Not ideal | Perfect fit |

## Examples

### Simple Classification

```python
class Classifier:
    def __init__(self):
        self.model = ResNet50(pretrained=True)

    def predict(self, image):
        return self.model(image)

classifier = Classifier()
wrapped = TaskWrapper(classifier)

# Classify individual images
for image_path in image_paths:
    image = load_image(image_path)
    result = wrapped.predict(image)
    print(f"{image_path}: {result}")
```

### Multi-model Pipeline

```python
class Pipeline:
    def __init__(self):
        self.detector = ObjectDetector()
        self.classifier = ImageClassifier()

    def process(self, image):
        detections = self.detector(image)
        classifications = self.classifier(detections)
        return {"detections": detections, "classes": classifications}

pipeline = Pipeline()
wrapped = TaskWrapper(pipeline)

result = wrapped.process(image)
```

## Best Practices

1. **Use for sporadic calls** - Not for high-throughput batch processing
2. **Keep tasks stateless** - Don't rely on state between calls
3. **Consider task overhead** - Each call spawns a new task
4. **Switch to actors for batches** - Use `ActorWrapper` for `map_batches`

## Next Steps

- Learn about [ActorWrapper](actors.md) for batch processing
- See [Tutorials](../tutorials/index.md) for complete examples
- Check [API Reference](../api_reference/wrappers.md) for details
