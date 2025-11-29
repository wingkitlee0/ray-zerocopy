# User Guide

This guide covers the key concepts and usage patterns for ray-zerocopy.

```{toctree}
:maxdepth: 2

core_concepts
tasks
actors
torchscript
ray_data_integration
```

## Overview

ray-zerocopy provides four main wrapper classes:

- **TaskWrapper** - For nn.Module models with Ray tasks
- **ActorWrapper** - For nn.Module models with Ray actors
- **JITTaskWrapper** - For TorchScript models with Ray tasks
- **JITActorWrapper** - For TorchScript models with Ray actors

Choose based on:
1. **Execution mode**: Tasks (one-off) vs Actors (stateful, persistent)
2. **Model type**: nn.Module vs TorchScript (compiled)

## Quick Decision Guide

**Use TaskWrapper when:**
- Running ad-hoc inference with Ray tasks
- Don't need persistent state
- Models are standard nn.Module

**Use ActorWrapper when:**
- Using Ray Data with `map_batches`
- Need long-running inference service
- Want to maintain state between calls
- Models are standard nn.Module

**Use JITTaskWrapper when:**
- Same as TaskWrapper, but models are TorchScript compiled

**Use JITActorWrapper when:**
- Same as ActorWrapper, but models are TorchScript compiled
