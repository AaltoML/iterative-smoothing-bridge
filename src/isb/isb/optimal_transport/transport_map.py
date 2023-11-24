"""Compute transport matrix based on sinkhorn potentials.

Code from the differentiable particle filtering codebase, shared by Adrien Corenflos.
"""
import torch
import math
from isb.isb.optimal_transport import diameter, squared_distances
from isb.isb.optimal_transport import sinkhorn_potentials


@torch.jit.script
def transport_particles(f, g, logw_x, logw_y, epsilon, cost_matrix):
    B, N = logw_x.shape
    p_matrix = f.reshape(B, N, 1) + g.reshape(B, 1, N)
    p_matrix = p_matrix - cost_matrix
    p_matrix = p_matrix / epsilon
    p_matrix = p_matrix + logw_x.reshape(B, N, 1) + logw_y.reshape(B, 1, N)
    p_matrix = p_matrix.exp()
    return p_matrix


@torch.jit.script
def transform_matrix_from_potentials(f, g, logw_x, logw_y, epsilon, cost_matrix):
    B, N = logw_x.shape
    p_matrix = f.reshape(B, N, 1) + g.reshape(B, 1, N)
    p_matrix = p_matrix - cost_matrix
    p_matrix = p_matrix / epsilon

    p_matrix = p_matrix + logw_x.reshape(B, N, 1) + logw_y.reshape(B, 1, N)
    totals = torch.logsumexp(p_matrix, 2, True)
    p_matrix = p_matrix - totals
    p_matrix = p_matrix.exp()
    return p_matrix

def compute_dual(p_matrix, logw_x, logw_y, epsilon, cost_matrix):
    """Compute the dual DOT_epsilon at the optimal point (Wasserstein distance)."""
    weight_term = logw_x + logw_y
    matrix_term = torch.log(p_matrix) - weight_term
    matrix_term = cost_matrix + epsilon*matrix_term
    matrix_term = p_matrix*matrix_term
    
    dual = torch.sum(torch.sum(matrix_term, dim=1), dim=0)
    return dual

def transport_resample(particles, log_weights, eps=0.1, num_iter=100, threshold=0.01, stable=True):
        B, N, d = particles.shape
        logw_x = torch.full_like(log_weights, -math.log(N))
        logw_y = log_weights
        diameter_value = diameter(particles, particles)
        scale = diameter_value * d
        centered_x = particles - torch.mean(particles, dim=1, keepdim=True).detach().repeat(1,N,1)
        scale = scale.reshape(B,1,1)
        scaled_x = centered_x / scale.detach()
        f, g = sinkhorn_potentials(scaled_x, logw_x, scaled_x, logw_y,
                                   epsilon=eps,
                                   num_iterations=num_iter,
                                   threshold=threshold,
                                   cost_fn=squared_distances,
                                   stable=stable)

        cost_matrix = squared_distances(scaled_x, scaled_x)
        ensemble_matrix = transform_matrix_from_potentials(f, g, logw_x, logw_y, eps, cost_matrix)
        transported_particles = torch.bmm(ensemble_matrix, particles) #torch.einsum('bnk, bkd -> bnd', ensemble_matrix, x)
        return transported_particles, logw_x, ensemble_matrix