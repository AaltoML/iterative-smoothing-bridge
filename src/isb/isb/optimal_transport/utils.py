import torch


#@torch.jit.script
def squared_distances(x: torch.Tensor, y: torch.Tensor):
    D_xx = (x * x).sum(-1).unsqueeze(2)  # (B,N,1)
    D_xy = torch.matmul(x, y.permute(0, 2, 1))  # (B,N,D) @ (B,D,M) = (B,N,M)
    D_yy = (y * y).sum(-1).unsqueeze(1)  # (B,1,M)
    return D_xx - 2 * D_xy + D_yy


##@torch.jit.script
def softmin(x, epsilon, log_w):
    exponent = -x / epsilon
    exponent = exponent + log_w
    return -epsilon * torch.logsumexp(exponent, 2, True)


def diameter(x, y):
    res = torch.max(
    torch.std(x,1).max(axis=-1).values,
    torch.std(y,1).max(axis=-1).values)
    ones = torch.tensor(1.0)
    ones = ones.type_as(x)
    return torch.where(res == 0.,ones , res)


