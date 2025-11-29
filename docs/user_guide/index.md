# User Guide

This guide covers the key concepts and usage patterns for ray-zerocopy.

```{toctree}
:maxdepth: 2

core_concepts
ray_data_integration
```

## Overview

ray-zerocopy provides wrapper classes for zero-copy model sharing:

- **ModelWrapper** - For nn.Module models (supports both task and actor modes)
- **JITTaskWrapper** - For TorchScript models with Ray tasks
- **JITActorWrapper** - For TorchScript models with Ray actors

Choose based on:
1. **Execution mode**: Tasks (one-off) vs Actors (stateful, persistent)
2. **Model type**: nn.Module vs TorchScript (compiled)

## Quick Decision Guide

**Use ModelWrapper.for_tasks() when:**
- Running ad-hoc inference with Ray tasks
- Don't need persistent state
- Models are standard nn.Module

**Use ModelWrapper.from_model(..., mode="actor") when:**
- Using Ray Data with `map_batches`
- Need long-running inference service
- Want to maintain state between calls
- Models are standard nn.Module

**Use JITTaskWrapper when:**
- Same as ModelWrapper.for_tasks(), but models are TorchScript compiled

**Use JITActorWrapper when:**
- Same as ModelWrapper.from_model(..., mode="actor"), but models are TorchScript compiled
