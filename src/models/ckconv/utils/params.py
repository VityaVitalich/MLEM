import torch


def num_params(
    model: torch.nn.Module,
) -> int:
    """
    Calculates the number of parameters of a torch.nn.Module.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
