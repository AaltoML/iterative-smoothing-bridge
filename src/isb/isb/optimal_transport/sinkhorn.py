"""Compute sinkhorn potentials.

Code from the Differentiable particle filtering codebase, shared by Adrien Corenflos.
"""

from collections.abc import Callable
import torch
from isb.isb.optimal_transport import squared_distances, softmin



#@torch.jit.script
def sinkhorn_mapping(cost_matrix, f, log_w, epsilon):
    B, N, _ = cost_matrix.shape
    f = f.reshape(B, 1, N)
    log_w = log_w.reshape(B, 1, N)
    x = cost_matrix - f
    return softmin(x, epsilon, log_w).reshape(B, 1, N)


def sinkhorn_potentials(x: torch.Tensor,
                        logw_x: torch.Tensor,
                        y: torch.Tensor,
                        logw_y: torch.Tensor,
                        epsilon,
                        num_iterations: int = 1,
                        threshold: float = 10 ** -3,
                        cost_fn: Callable = squared_distances,
                        stable: bool = True):
    B, N = logw_x.shape
    # cost matrices
    cost_xy = cost_fn(x, y)# y detached
    cost_yx = cost_fn(y, x)

 #   if stable:
  #      torch.autograd.set_grad_enabled(False)

    # init potentials
    f: torch.Tensor = torch.zeros((B, 1, N))
    f = f.type_as(x)
    g: torch.Tensor = torch.zeros((B, 1, N))
    g = g.type_as(x)

    keep_going = True
    iteration = 0.
    while keep_going:
        # active_epsilon = torch.max(epsilon, active_epsilon)
        active_epsilon = epsilon
        g_: torch.Tensor = sinkhorn_mapping(cost_yx, f, logw_x, active_epsilon)
        f_: torch.Tensor = sinkhorn_mapping(cost_xy, g, logw_y, active_epsilon)
        if stable:
            f = 0.5 * (f + f_)
            g = 0.5 * (g + g_)

        f_diff: torch.Tensor = torch.norm(f_ - f)
        g_diff: torch.Tensor = torch.norm(g_ - g)
        diff: torch.Tensor = torch.max(f_diff, g_diff)

        if not stable:
            g = g_
            f = f_

        iteration: int = iteration + 1
        keep_going: bool = (iteration < num_iterations) and (diff > threshold)

  #  if stable:
   #     torch.autograd.set_grad_enabled(True)

    g = sinkhorn_mapping(cost_yx, f, logw_x, epsilon)  # f was detached
    f = sinkhorn_mapping(cost_xy, g, logw_y, epsilon)  # g was detached

    return f, g
