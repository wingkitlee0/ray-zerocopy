# User Guide

This guide covers key concepts and usage patterns for ray-zerocopy.

```{toctree}
:maxdepth: 2

core_concepts
ray_data_integration
```

## Overview

For complete usage examples and API details, see the [ModelWrapper Guide](../model_wrapper_guide.md).

## Key Concepts

- **Zero-Copy**: Model weights are stored once in Ray's object store and shared across workers
- **Task Mode**: For ad-hoc inference calls with Ray tasks
- **Actor Mode**: For batch processing with Ray Data or long-running Ray Actors
- **Pipelines**: Classes with `nn.Module` attributes are automatically detected and shared

## Next Steps

- Read the [ModelWrapper Guide](../model_wrapper_guide.md) for complete examples
- See [Core Concepts](core_concepts.md) for zero-copy details
- Check the [API Reference](../api_reference/index.md) for detailed documentation
