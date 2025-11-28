# Migration Guide: Old API → New Unified Pipeline API

This guide helps you migrate from the old ray_zerocopy API to the new unified wrapper-based API.

## Overview

The old API had **inconsistent interfaces** across different use cases:
- Different function names for tasks vs actors
- Separate imports for JIT vs nn.Module models  
- Manual skeleton/model_ref management for actors

The **new API** provides four consistent wrapper classes:
- `TaskWrapper` - nn.Module + Ray tasks
- `ActorWrapper` - nn.Module + Ray actors  
- `JITTaskWrapper` - TorchScript + Ray tasks
- `JITActorWrapper` - TorchScript + Ray actors

## Migration Examples

### 1. nn.Module with Ray Tasks

**Old API:**
```python
from ray_zerocopy import rewrite_pipeline

pipeline = MyPipeline()
rewritten = rewrite_pipeline(pipeline)
result = rewritten.process(data)
```

**New API:**
```python
from ray_zerocopy import TaskWrapper

pipeline = MyPipeline()
wrapped = TaskWrapper(pipeline)
result = wrapped.process(data)
```

**Benefits:**
- Same interface, clearer name
- Deprecation warning guides migration
- All behavior preserved

---

### 2. nn.Module with Ray Actors (Ray Data)

**Old API:**
```python
from ray_zerocopy import rewrite_pipeline_for_actors, load_pipeline_in_actor

pipeline = MyPipeline()
skeleton, model_refs = rewrite_pipeline_for_actors(pipeline, device="cuda:0")

class InferenceActor:
    def __init__(self, skeleton, model_refs, device):
        self.pipeline = load_pipeline_in_actor(skeleton, model_refs, device=device)
    
    def __call__(self, batch):
        return self.pipeline(batch["data"])

ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={
        "skeleton": skeleton,
        "model_refs": model_refs,
        "device": "cuda:0"
    },
    compute=ActorPoolStrategy(size=4)
)
```

**New API:**
```python
from ray_zerocopy import ActorWrapper

pipeline = MyPipeline()
actor_wrapper = ActorWrapper(pipeline, device="cuda:0")

class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()
    
    def __call__(self, batch):
        return self.pipeline(batch["data"])

ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    compute=ActorPoolStrategy(size=4)
)
```

**Benefits:**
- Single wrapper object encapsulates all state
- No need to manually manage skeleton/model_refs
- Cleaner actor constructor
- `.load()` method is explicit and self-documenting

---

### 3. TorchScript with Ray Tasks

**Old API:**
```python
from ray_zerocopy.jit import rewrite_pipeline

jit_pipeline = torch.jit.trace(pipeline, example)
rewritten = rewrite_pipeline(jit_pipeline)
result = rewritten(data)
```

**New API:**
```python
from ray_zerocopy import JITTaskWrapper

jit_pipeline = torch.jit.trace(pipeline, example)
wrapped = JITTaskWrapper(jit_pipeline)
result = wrapped(data)
```

**Benefits:**
- Consistent with TaskWrapper naming
- No need for separate `jit` import
- Clear that it's for TorchScript models

---

### 4. TorchScript with Ray Actors (NEW!)

**This was not previously supported!**

**New API:**
```python
from ray_zerocopy import JITActorWrapper

jit_pipeline = torch.jit.trace(pipeline, example)
actor_wrapper = JITActorWrapper(jit_pipeline, device="cuda:0")

class InferenceActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()
    
    def __call__(self, batch):
        return self.pipeline(batch["data"])

ds.map_batches(
    InferenceActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper},
    compute=ActorPoolStrategy(size=4)
)
```

**Benefits:**
- Now you can use TorchScript with Ray actors!
- Same interface as ActorWrapper
- Zero-copy loading for compiled models in actors

---

## Quick Reference Table

| Old API | New API | Use Case |
|---------|---------|----------|
| `rewrite_pipeline(pipeline)` | `TaskWrapper(pipeline)` | nn.Module + Tasks |
| `prepare_model_for_actors(model)` + `load_model_in_actor(ref)` | `ActorWrapper(pipeline).load()` | nn.Module + Actors |
| `rewrite_pipeline_for_actors(pipeline)` + `load_pipeline_in_actor(skeleton, refs)` | `ActorWrapper(pipeline).load()` | nn.Module + Actors |
| `jit.rewrite_pipeline(pipeline)` | `JITTaskWrapper(pipeline)` | TorchScript + Tasks |
| ❌ Not available | `JITActorWrapper(pipeline).load()` | TorchScript + Actors |

---

## When to Use Each Wrapper

### TaskWrapper
- You want to run inference via **Ray tasks** (not actors)
- Models are **nn.Module**
- Good for one-off inference calls or simple parallelism

### ActorWrapper  
- You want to run inference in **Ray actors**
- Models are **nn.Module**
- Using **Ray Data** with `ActorPoolStrategy`
- Need stateful actors that load models once

### JITTaskWrapper
- You want to run inference via **Ray tasks** (not actors)
- Models are **TorchScript** (`torch.jit.ScriptModule`)
- Models have been traced or scripted

### JITActorWrapper
- You want to run inference in **Ray actors**
- Models are **TorchScript** (`torch.jit.ScriptModule`)
- Using **Ray Data** with `ActorPoolStrategy`
- Want compiled model performance with actor benefits

---

## Advanced: Alternative Usage Patterns

### Option 1: Pass wrapper directly (recommended)
```python
actor_wrapper = ActorWrapper(pipeline)

class MyActor:
    def __init__(self, actor_wrapper):
        self.pipeline = actor_wrapper.load()

ds.map_batches(
    MyActor,
    fn_constructor_kwargs={"actor_wrapper": actor_wrapper}
)
```

### Option 2: Use `.constructor_kwargs` property
```python
actor_wrapper = ActorWrapper(pipeline)

class MyActor:
    def __init__(self, pipeline_skeleton, model_refs, device, use_fast_load):
        from ray_zerocopy.actor import load_pipeline_in_actor
        self.pipeline = load_pipeline_in_actor(
            pipeline_skeleton, model_refs, device, use_fast_load
        )

ds.map_batches(
    MyActor,
    fn_constructor_kwargs=actor_wrapper.constructor_kwargs
)
```

**Recommendation:** Use Option 1 (pass wrapper directly) for cleaner code.

---

## Backward Compatibility

The old API still works! You'll see deprecation warnings, but no functionality is removed:

```python
# This still works, but shows a deprecation warning
from ray_zerocopy import rewrite_pipeline
rewritten = rewrite_pipeline(pipeline)
```

Output:
```
DeprecationWarning: rewrite_pipeline() is deprecated. 
Use TaskWrapper instead: from ray_zerocopy import TaskWrapper
```

---

## Summary

✅ **More consistent** - Same patterns across all use cases  
✅ **Easier to learn** - Unified naming scheme  
✅ **More features** - TorchScript now works with actors  
✅ **Cleaner code** - Less boilerplate, clearer intent  
✅ **Backward compatible** - Old code still works  

Start migrating today by importing the new wrappers:
```python
from ray_zerocopy import TaskWrapper, ActorWrapper, JITTaskWrapper, JITActorWrapper
```
