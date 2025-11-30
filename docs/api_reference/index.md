# API Reference

Complete API documentation for ray-zerocopy.

```{toctree}
:maxdepth: 2

model_wrappers
```

## Quick Links

### Main Wrappers

- {py:class}`~ray_zerocopy.model_wrappers.ModelWrapper` - nn.Module with Ray tasks or actors
- {py:class}`~ray_zerocopy.wrappers.JITTaskWrapper` - TorchScript with Ray tasks
- {py:class}`~ray_zerocopy.wrappers.JITActorWrapper` - TorchScript with Ray actors

## Overview

ray-zerocopy provides wrapper classes for zero-copy model sharing:

1. **ModelWrapper** - Primary API for nn.Module models (supports both task and actor modes)
2. **JITTaskWrapper/JITActorWrapper** - For TorchScript (compiled) models

## Importing

```python
# Primary API for nn.Module
from ray_zerocopy import ModelWrapper

# For TorchScript models
from ray_zerocopy import JITTaskWrapper, JITActorWrapper
```
