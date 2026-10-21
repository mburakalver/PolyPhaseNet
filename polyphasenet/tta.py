"""
Episodic Safeguarded Test-Time Adaptation (TTA) for 2D Non-Separable SAR Autofocus.

This module implements the episodic, entropy-gated adaptation protocol described in
Algorithm 1 of:
    "Phase Gradient-Based Deep Learning for 2D Non-Separable SAR Autofocus"
    IEEE Transactions on Aerospace and Electronic Systems (TAES-2026-3321), Section IV-A.

Key Architectural Guarantees:
------------------------------
1. Freeze CNN Backbone: The convolutional feature extractor parameters theta remain strictly
   frozen; only the low-dimensional 2D B-Spline control matrix C in R^{K x L} is adapted.
2. Trust-Region Regularization: An L2 penalty ||C - C_0||_F^2 anchors the optimization
   trajectory around the feed-forward initial estimate C_0 to prevent overfitting to speckle.
3. Unsupervised Entropy Verification Gate: Evaluates candidate iterates over the trajectory
   {0, C_0, C_1, ..., C_J} using normalized image Shannon entropy H(X). If adaptation increases
   entropy (negative transfer), the gate automatically reverts to the optimal historic iterate.
4. Episodic Reset: Parameters are reset per scene patch, ensuring zero cross-sample weight leakage.
"""

import torch
import torch.optim as optim


def episodic_safeguarded_tta(
    model_base,
    S_in: torch.Tensor,
    steps: int = 6,
    lr: float = 1.5e-2,
    reg_weight: float = 0.05
):
    """
    Executes Episodic Safeguarded Test-Time Adaptation for a single SAR observation patch.

    Optimization Objective:
    -----------------------
    min_{C}  L_TTA(C) = H(X(C)) + lambda_reg * ||C - C_0||_F^2

    where:
    - S_corr(eta, r; C) = S_in(eta, r) * exp(-j * Phi(eta, r; C))
    - X(C) = F_{az}{S_corr}
    - H(X) = -sum_{x,r} p(x, r) * log(p(x, r) + eps) with p(x, r) = |X(x, r)|^2 / sum |X|^2
    - C_0 is the zero-shot feed-forward control matrix predicted by PolyPhaseNet.

    Parameters:
    -----------
    model_base : PolyPhaseNet
        Trained model instance (backbone weights will remain frozen).
    S_in : torch.Tensor
        Complex slow-time observation tensor of shape (1, 1, Azimuth, Range).
    steps : int, default=6
        Number of episodic gradient adaptation steps J.
    lr : float, default=1.5e-2
        Learning rate for the Adam optimizer updating the control matrix C.
    reg_weight : float, default=0.05
        Weight lambda_reg for the L2 trust-region penalty around C_0.

    Returns:
    --------
    best_X : torch.Tensor
        Focused complex Single-Look Complex (SLC) SAR image of shape (1, 1, Azimuth, Range).
    best_phi : torch.Tensor
        Estimated 2D continuous phase error screen of shape (1, 1, Azimuth, Range) in radians.
    """
    device = S_in.device
    model_base.eval()

    with torch.no_grad():
        # Baseline: Uncorrected defocused observation
        X_def = torch.fft.fft(S_in, dim=2)
        I_def = torch.abs(X_def) ** 2
        p_def = I_def / (torch.sum(I_def) + 1e-12)
        ent_def = float(-torch.sum(p_def * torch.log(p_def + 1e-12)).item())

        # Candidate 0: Zero-shot feed-forward prediction C_0
        X_0, phi_0, coeffs_0 = model_base(S_in)
        I_0 = torch.abs(X_0) ** 2
        p_0 = I_0 / (torch.sum(I_0) + 1e-12)
        ent_0 = float(-torch.sum(p_0 * torch.log(p_0 + 1e-12)).item())

    # Initialize from the best performing state between zero-shot C_0 and zero correction
    if ent_0 <= ent_def:
        C_init = coeffs_0.detach().clone()
        best_X = X_0.detach().clone()
        best_phi = phi_0.detach().clone()
        best_ent = ent_0
    else:
        C_init = torch.zeros_like(coeffs_0)
        best_X = X_def.detach().clone()
        best_phi = torch.zeros_like(phi_0)
        best_ent = ent_def

    # Optimize only the control matrix coefficients C
    C_opt = C_init.clone().requires_grad_(True)
    opt = optim.Adam([C_opt], lr=lr)
    scale = model_base.phase_scale.detach()

    for step in range(steps):
        opt.zero_grad()
        if model_base.mode == 'tensor_grid':
            C_mat = C_opt.view(1, model_base.az_points, model_base.rg_points) * scale
            phi_cur = torch.einsum('ak,bkl,rl->bar', model_base.B_az, C_mat, model_base.B_rg).unsqueeze(1)
        else:
            c_az = C_opt[:, :model_base.az_points]
            c_rg = C_opt[:, model_base.az_points:]
            c_2d = torch.bmm(c_az.unsqueeze(2), c_rg.unsqueeze(1)) * scale
            phi_cur = torch.einsum('ak,bkl,rl->bar', model_base.B_az, c_2d, model_base.B_rg).unsqueeze(1)

        # Apply phase compensation and range-compressed azimuth FFT
        S_corr = S_in * torch.exp(-1j * phi_cur)
        X_cur = torch.fft.fft(S_corr, dim=2)
        I_cur = torch.abs(X_cur) ** 2
        p_cur = I_cur / (torch.sum(I_cur) + 1e-12)
        ent_loss = -torch.sum(p_cur * torch.log(p_cur + 1e-12))
        reg_loss = reg_weight * torch.mean((C_opt - C_init) ** 2)
        loss = ent_loss + reg_loss
        loss.backward()
        opt.step()

        # Unsupervised Entropy Verification Gate:
        # Guarantee monotonic non-increasing image entropy across iterates
        cur_ent_val = float(ent_loss.item())
        if cur_ent_val < best_ent:
            best_ent = cur_ent_val
            best_X = X_cur.detach().clone()
            best_phi = phi_cur.detach().clone()

    return best_X, best_phi
