# API Reference

Complete API documentation for ray-zerocopy.

```{toctree}
:maxdepth: 2

wrappers
model_wrappers
```

## Quick Links

### Main Wrappers

- {py:class}`~ray_zerocopy.wrappers.TaskWrapper` - nn.Module with Ray tasks
- {py:class}`~ray_zerocopy.wrappers.ActorWrapper` - nn.Module with Ray actors
- {py:class}`~ray_zerocopy.wrappers.JITTaskWrapper` - TorchScript with Ray tasks
- {py:class}`~ray_zerocopy.wrappers.JITActorWrapper` - TorchScript with Ray actors

### Alternative API

- {py:class}`~ray_zerocopy.model_wrappers.ModelWrapper` - Flexible wrapper with explicit methods

## Overview

ray-zerocopy provides two API styles:

1. **Primary API** (`wrappers.py`) - Simple, consistent wrapper classes
2. **Alternative API** (`model_wrappers.py`) - Explicit serialization/deserialization

Most users should use the primary API.

## Importing

```python
# Primary API
from ray_zerocopy import TaskWrapper, ActorWrapper, JITTaskWrapper, JITActorWrapper

# Alternative API
from ray_zerocopy import ModelWrapper
```
