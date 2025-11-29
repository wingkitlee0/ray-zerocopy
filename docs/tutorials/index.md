# Tutorials

Step-by-step tutorials for common ray-zerocopy use cases.

```{toctree}
:maxdepth: 1

basic_inference
pipeline_example
ray_data_batch
```

## Overview

These tutorials walk through complete examples of using ray-zerocopy for different inference scenarios:

1. **Basic Inference** - Simple single-model inference with Ray Data
2. **Pipeline Example** - Multi-model pipelines with encoder-decoder architecture
3. **Ray Data Batch Inference** - Production-ready batch processing patterns

## Before You Start

Make sure you have ray-zerocopy installed:

```bash
pip install ray-zerocopy
```

And Ray initialized:

```python
import ray
ray.init()
```

## Running the Examples

Each tutorial includes complete, runnable code. You can:

1. Copy the code to a Python file
2. Run directly: `python my_tutorial.py`
3. Modify for your use case

## Getting Help

If you run into issues:

- Check the [User Guide](../user_guide/index.md) for concepts
- See the [API Reference](../api_reference/index.md) for details
- Open an issue on GitHub
