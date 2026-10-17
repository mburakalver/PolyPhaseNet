import torch
import numpy as np

def pga_1d_autofocus(S_in: torch.Tensor, num_iters: int = 8, win_size: int = 32):
    """
    Classical 1D Phase Gradient Autofocus (Wahl et al., IEEE TAES 1994).
    S_in: Complex tensor of shape (1, 1, Az, Rg) or (Az, Rg) in Range-Doppler (slow-time x range) domain.
    Returns: X_corr (Az, Rg), phi_total (Az, Rg)
    """
    S = S_in.squeeze().clone()
    device = S.device
    Az, Rg = S.shape
    phi_total = torch.zeros((Az, Rg), device=device, dtype=torch.float32)

    x_lin = torch.linspace(-1.0, 1.0, Az, device=device, dtype=torch.float32)
    A_lin = torch.stack([x_lin, torch.ones_like(x_lin)], dim=1) # (Az, 2)
    A_pinv = torch.linalg.pinv(A_lin) # (2, Az)

    for it in range(num_iters):
        # 1. Reconstruct current image along azimuth (dim=0)
        g = torch.fft.fft(S, dim=0)
        mag_g = torch.abs(g)
        
        # 2. Circular shift brightest scatterer in each range bin to center (Az // 2)
        max_idx = torch.argmax(mag_g, dim=0) # (Rg,)
        shifts = (Az // 2) - max_idx
        
        # Vectorized circular shift per column
        row_idx = (torch.arange(Az, device=device).unsqueeze(1) - shifts.unsqueeze(0)) % Az
        g_shifted = torch.gather(g, 0, row_idx)
        
        # 3. Adaptive windowing around center
        cur_w = max(8, win_size // (1 + it // 3))
        mask = torch.zeros((Az, 1), device=device, dtype=torch.float32)
        mask[Az // 2 - cur_w // 2 : Az // 2 + cur_w // 2, 0] = 1.0
        g_win = g_shifted * mask
        
        # 4. Transform windowed image back to slow-time aperture domain
        G_win = torch.fft.ifft(g_win, dim=0)
        
        # 5. Maximum Likelihood Phase Gradient Estimate across range bins
        G_prev = torch.roll(G_win, shifts=1, dims=0)
        cross = G_win * torch.conj(G_prev)
        cross[0, :] = 0.0
        delta_phi = torch.angle(torch.sum(cross, dim=1)) # (Az,)
        
        # 6. Integrate and detrend (remove piston and linear ramp)
        phi_1d = torch.cumsum(delta_phi, dim=0)
        trend = A_lin @ (A_pinv @ phi_1d)
        phi_1d = phi_1d - trend
        
        phi_2d = phi_1d.unsqueeze(1).expand(Az, Rg)
        phi_total = phi_total + phi_2d
        S = S * torch.exp(-1j * phi_2d)

    X_corr = torch.fft.fft(S, dim=0)
    return X_corr, phi_total


def pga_2d_autofocus(S_in: torch.Tensor, num_iters: int = 8, num_subbands: int = 4, win_size: int = 32):
    """
    2D Phase Gradient Autofocus (Range-Dependent Sub-Band / Coupled Polynomial PGA).
    Estimates azimuth phase gradients across range sub-bands and fits a smooth range-coupled 2D phase screen.
    """
    S = S_in.squeeze().clone()
    device = S.device
    Az, Rg = S.shape
    phi_total = torch.zeros((Az, Rg), device=device, dtype=torch.float32)

    x_lin = torch.linspace(-1.0, 1.0, Az, device=device, dtype=torch.float32)
    A_lin = torch.stack([x_lin, torch.ones_like(x_lin)], dim=1)
    A_pinv = torch.linalg.pinv(A_lin)

    # Range basis (constant + linear + quadratic across range)
    r_grid = torch.linspace(-1.0, 1.0, Rg, device=device, dtype=torch.float32)
    sub_centers = torch.linspace(-1.0 + 1.0/num_subbands, 1.0 - 1.0/num_subbands, num_subbands, device=device)
    R_sub = torch.stack([torch.ones_like(sub_centers), sub_centers], dim=1) # (num_subbands, 2)
    R_sub_pinv = torch.linalg.pinv(R_sub) # (2, num_subbands)
    R_full = torch.stack([torch.ones_like(r_grid), r_grid], dim=1) # (Rg, 2)

    sub_w = Rg // num_subbands

    for it in range(num_iters):
        g = torch.fft.fft(S, dim=0)
        mag_g = torch.abs(g)
        max_idx = torch.argmax(mag_g, dim=0)
        shifts = (Az // 2) - max_idx
        row_idx = (torch.arange(Az, device=device).unsqueeze(1) - shifts.unsqueeze(0)) % Az
        g_shifted = torch.gather(g, 0, row_idx)

        cur_w = max(8, win_size // (1 + it // 3))
        mask = torch.zeros((Az, 1), device=device, dtype=torch.float32)
        mask[Az // 2 - cur_w // 2 : Az // 2 + cur_w // 2, 0] = 1.0
        g_win = g_shifted * mask
        G_win = torch.fft.ifft(g_win, dim=0)

        G_prev = torch.roll(G_win, shifts=1, dims=0)
        cross = G_win * torch.conj(G_prev)
        cross[0, :] = 0.0

        # Estimate azimuth phase error per range sub-band
        phi_subbands = []
        for b in range(num_subbands):
            c_start = b * sub_w
            c_end = Rg if b == num_subbands - 1 else (b + 1) * sub_w
            dphi_b = torch.angle(torch.sum(cross[:, c_start:c_end], dim=1))
            phi_b = torch.cumsum(dphi_b, dim=0)
            phi_b = phi_b - (A_lin @ (A_pinv @ phi_b))
            phi_subbands.append(phi_b)

        Phi_sub = torch.stack(phi_subbands, dim=1) # (Az, num_subbands)
        # Fit linear range coupling across sub-bands and interpolate to full (Az, Rg)
        coeffs_r = Phi_sub @ R_sub_pinv.T # (Az, 2)
        phi_2d = coeffs_r @ R_full.T      # (Az, Rg)

        phi_total = phi_total + phi_2d
        S = S * torch.exp(-1j * phi_2d)

    X_corr = torch.fft.fft(S, dim=0)
    return X_corr, phi_total


def mea_autofocus(S_in: torch.Tensor, num_iters: int = 25, lr: float = 0.15):
    """
    Classical Iterative Minimum Entropy Autofocus (MEA, Xi et al., IEEE TAES 1999)
    extended to 2D coupled polynomial/low-order basis optimization via gradient descent on Shannon Entropy.
    """
    S = S_in.squeeze().detach().clone()
    device = S.device
    Az, Rg = S.shape

    # Parameterize 2D phase screen over a coarse 16x4 grid to avoid pixel-wise trivial solutions
    ctrl = torch.zeros((1, 1, 16, 4), device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([ctrl], lr=lr)

    for _ in range(num_iters):
        optimizer.zero_grad()
        phi_2d = torch.nn.functional.interpolate(ctrl, size=(Az, Rg), mode='bilinear', align_corners=True).squeeze()
        S_corr = S * torch.exp(-1j * phi_2d)
        X_corr = torch.fft.fft(S_corr, dim=0)
        I = torch.abs(X_corr) ** 2
        p = I / (torch.sum(I) + 1e-12)
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        entropy.backward()
        optimizer.step()

    with torch.no_grad():
        phi_final = torch.nn.functional.interpolate(ctrl, size=(Az, Rg), mode='bilinear', align_corners=True).squeeze()
        phi_final = phi_final - torch.mean(phi_final)
        X_final = torch.fft.fft(S * torch.exp(-1j * phi_final), dim=0)

    return X_final, phi_final
