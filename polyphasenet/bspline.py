import numpy as np
import torch

def cox_de_boor_basis(n_samples: int, n_ctrl: int, degree: int = 3) -> torch.Tensor:
    """
    Constructs a clamped uniform B-Spline basis matrix B of shape (n_samples, n_ctrl)
    over the normalized domain u in [0, 1] using the Cox-de Boor recursion formula.
    """
    d = min(degree, n_ctrl - 1)
    # Clamped knot vector of length n_ctrl + d + 1
    n_internal = n_ctrl - d + 1
    internal_knots = np.linspace(0.0, 1.0, n_internal)
    knots = np.concatenate([
        np.zeros(d),
        internal_knots,
        np.ones(d)
    ])
    
    u = np.linspace(0.0, 1.0, n_samples)
    # Degree 0 basis functions
    B = np.zeros((n_samples, len(knots) - 1), dtype=np.float64)
    for i in range(len(knots) - 1):
        if i == len(knots) - 2 - d:
            # Include right endpoint u = 1.0 in the last active span
            B[:, i] = ((u >= knots[i]) & (u <= knots[i + 1])).astype(np.float64)
        else:
            B[:, i] = ((u >= knots[i]) & (u < knots[i + 1])).astype(np.float64)
            
    # Elevate degree from 1 to d
    for p in range(1, d + 1):
        B_next = np.zeros((n_samples, len(knots) - 1 - p), dtype=np.float64)
        for i in range(len(knots) - 1 - p):
            denom1 = knots[i + p] - knots[i]
            term1 = ((u - knots[i]) / denom1) * B[:, i] if denom1 > 1e-12 else 0.0
            
            denom2 = knots[i + p + 1] - knots[i + 1]
            term2 = ((knots[i + p + 1] - u) / denom2) * B[:, i + 1] if denom2 > 1e-12 else 0.0
            
            B_next[:, i] = term1 + term2
        B = B_next
        
    # Ensure exact partition of unity
    row_sums = B.sum(axis=1, keepdims=True)
    B = B / np.maximum(row_sums, 1e-12)
    return torch.from_numpy(B[:, :n_ctrl]).float()
