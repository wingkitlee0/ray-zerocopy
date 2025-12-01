import torch


def create_large_model(
    use_jit: bool = False,
) -> torch.nn.Module | torch.jit.ScriptModule:
    """Create a model large enough to see memory differences (~500MB)."""
    model = torch.nn.Sequential(
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 100),
    )

    # Convert to TorchScript if JIT mode is enabled
    if use_jit:
        print("Converting model to TorchScript...")
        example_input = torch.randn(1, 5000)
        model = torch.jit.trace(model, example_input)
        model.eval()
        return model
    else:
        return model
