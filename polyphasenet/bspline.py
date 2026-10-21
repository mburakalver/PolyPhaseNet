"""
Clamped Uniform Cox-de Boor B-Spline Basis Generator for 2D Non-Separable SAR Autofocus.

This module implements the 1D Cox-de Boor recursion algorithm used to generate the
azimuth and range projection matrices:
    B_az in R^{A x K}  and  B_rg in R^{R x L}
for the 2D tensor-product cubic B-Spline phase screen formulation:
    Phi(eta, r) = B_az * C * B_rg^T

where:
    - eta in [0, A-1] represents slow-time (azimuth) index
    - r in [0, R-1] represents range sample index
    - C in R^{K x L} is the learnable control grid matrix (default: K=64, L=16)
    - Degree d = 3 provides C^2 continuous surfaces with compact local support
      and guaranteed partition of unity: sum_{k=0}^{K-1} B_{k,d}(u) = 1.

Reference:
    "Phase Gradient-Based Deep Learning for 2D Non-Separable SAR Autofocus"
    IEEE Transactions on Aerospace and Electronic Systems (TAES-2026-3321), Section III-B.
"""

import numpy as np
import torch


def cox_de_boor_basis(n_samples: int, n_ctrl: int, degree: int = 3) -> torch.Tensor:
    """
    Constructs a clamped uniform B-Spline basis matrix B of shape (n_samples, n_ctrl)
    over the normalized continuous coordinate u in [0, 1] using the Cox-de Boor recursion.

    Mathematical Formulation:
    -------------------------
    For a knot sequence {u_0, u_1, ..., u_{m}} with m = n_ctrl + d:
    
    1. Base case (Degree p = 0):
       B_{i,0}(u) = 1  if u_i <= u < u_{i+1},  0 otherwise.
       (With right boundary endpoint u = 1.0 assigned to the final active span).

    2. Recursive step (Degree p = 1, ..., d):
       B_{i,p}(u) = ( (u - u_i) / (u_{i+p} - u_i) ) * B_{i,p-1}(u)
                  + ( (u_{i+p+1} - u) / (u_{i+p+1} - u_{i+1}) ) * B_{i+1,p-1}(u)
       where 0 / 0 := 0.

    3. Clamped Knot Vector:
       knots = [ 0, ..., 0,  u_0, u_1, ..., u_{K-d},  1, ..., 1 ]
                 |<- d+1 ->|                           |<- d+1 ->|
       This ensures the generated basis matches function endpoints exactly at u=0 and u=1.

    4. Partition of Unity:
       sum_{i=0}^{n_ctrl - 1} B_{i,d}(u) = 1.0  for all u in [0, 1].

    Parameters:
    -----------
    n_samples : int
        Number of discrete coordinate evaluation samples along the axis (e.g., A=256 or R=256).
    n_ctrl : int
        Number of B-Spline control points / knots (e.g., K=64 for azimuth, L=16 for range).
    degree : int, default=3
        Polynomial spline degree (3 = cubic, producing C^2 continuous surfaces).

    Returns:
    --------
    torch.Tensor of shape (n_samples, n_ctrl) and dtype torch.float32:
        Pre-computed basis projection matrix.
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
            
    # Elevate degree recursively from 1 up to d
    for p in range(1, d + 1):
        B_next = np.zeros((n_samples, len(knots) - 1 - p), dtype=np.float64)
        for i in range(len(knots) - 1 - p):
            denom1 = knots[i + p] - knots[i]
            term1 = ((u - knots[i]) / denom1) * B[:, i] if denom1 > 1e-12 else 0.0
            
            denom2 = knots[i + p + 1] - knots[i + 1]
            term2 = ((knots[i + p + 1] - u) / denom2) * B[:, i + 1] if denom2 > 1e-12 else 0.0
            
            B_next[:, i] = term1 + term2
        B = B_next
        
    # Guarantee exact partition of unity to machine precision
    row_sums = B.sum(axis=1, keepdims=True)
    B = B / np.maximum(row_sums, 1e-12)
    
    return torch.from_numpy(B[:, :n_ctrl]).float()
