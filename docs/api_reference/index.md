# API Reference

Complete API documentation for ray-zerocopy.

```{toctree}
:maxdepth: 2

model_wrappers
```

## Quick Links

### Main Wrappers

- {py:class}`~ray_zerocopy.model_wrappers.ModelWrapper` - nn.Module with Ray tasks or actors
- {py:class}`~ray_zerocopy.wrappers.JITModelWrapper` - TorchScript with Ray tasks or actors

## Overview

ray-zerocopy provides wrapper classes for zero-copy model sharing:

1. **ModelWrapper** - Primary API for nn.Module models (supports both task and actor modes)
2. **JITModelWrapper** - Unified API for TorchScript (compiled) models (supports both task and actor modes)

## Importing

```python
# Primary API for nn.Module
from ray_zerocopy import ModelWrapper

# For TorchScript models
from ray_zerocopy import JITModelWrapper
```
