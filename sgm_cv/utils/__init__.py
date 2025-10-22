import random
import numpy as np
import torch


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    return


def get_norm(x):
    """Compute the average Frobenius norm over a batch of tensors."""
    return torch.norm(x.view(x.shape[0], -1), p='fro', dim=-1).mean().item()


@torch.no_grad()
def batch_forward(model: torch.nn.Module, x1: torch.Tensor, x2: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Model forward pass in batches."""
    # x1: (N, *x1_shape); x2: (N, *x2_shape)
    assert x1.shape[0] == x2.shape[0], "x1 and x2 must have the same number of samples"
    assert x1.device == x2.device, "x1 and x2 must be on the same device"
    out=[]
    num_batches = (len(x1) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        if end_idx > len(x1):
            end_idx = len(x1)
        x1_batch=x1[start_idx:end_idx]
        x2_batch=x2[start_idx:end_idx]
        out.append(model(x1_batch, x2_batch))
    out = torch.cat(out, dim=0)
    return out
