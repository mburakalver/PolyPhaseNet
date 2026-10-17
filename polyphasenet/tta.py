import torch
import torch.optim as optim

def episodic_safeguarded_tta(model_base, S_in, steps=6, lr=1.5e-2, reg_weight=0.05):
    """
    Episodic Safeguarded Test-Time Adaptation (TTA) for PolyPhaseNet.
    
    Optimizes only the 2D B-Spline control matrix C for `steps` iterations
    starting from the zero-shot feed-forward prediction C_0 with an L2 trust-region
    anchor: L_TTA = H(X(C)) + lambda_reg * ||C - C_0||_F^2.
    
    An unsupervised Entropy Verification Gate monitors the candidate iterate trajectory
    {0, C_0, C_1, ..., C_J} and guarantees that the selected output strictly minimizes
    image entropy without cross-sample weight drift.
    
    Parameters:
    -----------
    model_base: PolyPhaseNet model instance (frozen backbone)
    S_in: Complex input slow-time tensor of shape (1, 1, A, R)
    steps: Number of gradient steps (default: 6)
    lr: Adam learning rate for control points (default: 1.5e-2)
    reg_weight: L2 trust-region regularizer weight (default: 0.05)
    
    Returns:
    --------
    best_X: Focused complex SLC image (1, 1, A, R)
    best_phi: Estimated 2D B-Spline phase screen (1, 1, A, R)
    """
    device = S_in.device
    model_base.eval()
    
    with torch.no_grad():
        # Evaluate uncorrected defocused observation
        X_def = torch.fft.fft(S_in, dim=2)
        I_def = torch.abs(X_def) ** 2
        p_def = I_def / (torch.sum(I_def) + 1e-12)
        ent_def = float(-torch.sum(p_def * torch.log(p_def + 1e-12)).item())

        # Evaluate zero-shot feed-forward prediction
        X_0, phi_0, coeffs_0 = model_base(S_in)
        I_0 = torch.abs(X_0) ** 2
        p_0 = I_0 / (torch.sum(I_0) + 1e-12)
        ent_0 = float(-torch.sum(p_0 * torch.log(p_0 + 1e-12)).item())

    # Initialize from the better of feed-forward C_0 vs. uncorrected baseline
    if ent_0 <= ent_def:
        C_init = coeffs_0.detach().clone()
        best_X, best_phi, best_ent = X_0.detach().clone(), phi_0.detach().clone(), ent_0
    else:
        C_init = torch.zeros_like(coeffs_0)
        best_X, best_phi, best_ent = X_def.detach().clone(), torch.zeros_like(phi_0), ent_def

    C_opt = C_init.clone().requires_grad_(True)
    opt = optim.Adam([C_opt], lr=lr)
    scale = model_base.phase_scale.detach()

    for _ in range(steps):
        opt.zero_grad()
        if model_base.mode == 'tensor_grid':
            C_mat = C_opt.view(1, model_base.az_points, model_base.rg_points) * scale
            phi_cur = torch.einsum('ak,bkl,rl->bar', model_base.B_az, C_mat, model_base.B_rg).unsqueeze(1)
        else:
            c_az = C_opt[:, :model_base.az_points]
            c_rg = C_opt[:, model_base.az_points:]
            c_2d = torch.bmm(c_az.unsqueeze(2), c_rg.unsqueeze(1)) * scale
            phi_cur = torch.einsum('ak,bkl,rl->bar', model_base.B_az, c_2d, model_base.B_rg).unsqueeze(1)

        S_corr = S_in * torch.exp(-1j * phi_cur)
        X_cur = torch.fft.fft(S_corr, dim=2)
        I_cur = torch.abs(X_cur) ** 2
        p_cur = I_cur / (torch.sum(I_cur) + 1e-12)
        ent_loss = -torch.sum(p_cur * torch.log(p_cur + 1e-12))
        reg_loss = reg_weight * torch.mean((C_opt - C_init) ** 2)
        loss = ent_loss + reg_loss
        loss.backward()
        opt.step()

        cur_ent_val = float(ent_loss.item())
        if cur_ent_val < best_ent:
            best_ent = cur_ent_val
            best_X = X_cur.detach().clone()
            best_phi = phi_cur.detach().clone()

    return best_X, best_phi
