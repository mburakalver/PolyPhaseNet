"""
PolyPhaseNet Demo: 2D Non-Separable SAR Autofocus Demonstration
==============================================================
Demonstrates:
  1. Synthetic 2D non-separable phase error synthesis (Moderate tier, coupled 2D)
  2. Phase gradient extraction via complex-conjugate adjacent-sample products
  3. PolyPhaseNet zero-shot inference (2D tensor-product cubic B-Spline)
  4. Episodic Safeguarded Test-Time Adaptation (TTA)
  5. Quantitative metric reporting (PSNR, SSIM, Phase MSE)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from polyphasenet import PolyPhaseNet, episodic_safeguarded_tta, subpixel_align_and_compute_metrics

def generate_synthetic_scene(patch_size=256, num_targets=25, seed=42):
    """Generates a synthetic complex SAR scene with point scatterers and background clutter."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Background speckle clutter (complex circular Gaussian)
    clutter = (torch.randn(patch_size, patch_size) + 1j * torch.randn(patch_size, patch_size)) * 0.05
    
    # Point scatterers (high SCR)
    targets = torch.zeros(patch_size, patch_size, dtype=torch.complex64)
    for _ in range(num_targets):
        r_x = np.random.randint(20, patch_size - 20)
        r_y = np.random.randint(20, patch_size - 20)
        amp = np.random.uniform(1.0, 5.0)
        phase = np.random.uniform(-np.pi, np.pi)
        targets[r_x, r_y] = amp * np.exp(1j * phase)
        
    X_clean = clutter + targets
    X_clean = X_clean / torch.max(torch.abs(X_clean))
    return X_clean

def generate_coupled_2d_phase_error(patch_size=256, max_phase=10.0):
    """Synthesizes a 2D non-separable coupled phase error surface."""
    az = torch.linspace(-1.0, 1.0, patch_size).unsqueeze(1) # (A, 1)
    rg = torch.linspace(-1.0, 1.0, patch_size).unsqueeze(0) # (1, R)
    
    # Multi-harmonic azimuth drift + range-varying slope + coupled cross term
    phi = (1.5 + 0.8 * rg) * (1.2 * torch.sin(2.0 * np.pi * az) + 0.6 * torch.cos(4.0 * np.pi * az)) \
          + 0.5 * torch.cos(np.pi * az) * (rg**2 - 1.0/3.0)
    
    # Scale to desired peak phase deviation
    phi = phi * (max_phase / torch.max(torch.abs(phi)))
    return phi

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Running PolyPhaseNet Demo on device: {device}")
    
    patch_size = 256
    X_gt = generate_synthetic_scene(patch_size=patch_size, seed=123).to(device)
    Phi_true = generate_coupled_2d_phase_error(patch_size=patch_size, max_phase=8.0).to(device)
    
    # Ingest into range-compressed slow-time domain S = F_az^-1{X}
    S_clean = torch.fft.ifft(X_gt, dim=0)
    S_defocused = S_clean * torch.exp(1j * Phi_true)
    X_defocused = torch.fft.fft(S_defocused, dim=0)
    
    S_in = S_defocused.unsqueeze(0).unsqueeze(0) # (1, 1, A, R)
    
    # Initialize PolyPhaseNet with 64x16 B-Spline control grid
    model = PolyPhaseNet(patch_size=patch_size, az_points=64, rg_points=16, degree=3, mode='tensor_grid').to(device)
    
    # Load pre-trained weights if available, else run untrained baseline demonstration
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "best_polyphasenet_2d_full.pth")
    if os.path.exists(ckpt_path):
        print(f"[+] Loading pre-trained weights from {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    else:
        print("[!] Pre-trained weights not found; executing model structure demonstration.")
        
    model.eval()
    
    # 1. Zero-Shot Feed-Forward Inference
    with torch.no_grad():
        X_zero_shot, phi_pred, _ = model(S_in)
        X_zero_shot = X_zero_shot.squeeze(0).squeeze(0)
        phi_pred = phi_pred.squeeze(0).squeeze(0)
        
    # 2. Episodic Safeguarded TTA
    print("[+] Executing Episodic Safeguarded TTA (6 iterations)...")
    X_tta, phi_tta = episodic_safeguarded_tta(model, S_in, steps=6, lr=1.5e-2, reg_weight=0.05)
    X_tta = X_tta.squeeze(0).squeeze(0)
    phi_tta = phi_tta.squeeze(0).squeeze(0)
    
    # 3. Compute Metrics
    def calc_metrics(X_pred, Phi_est):
        piston_mode = torch.mean(Phi_est - Phi_true).item()
        phase_mse = torch.mean((Phi_est - Phi_true - piston_mode)**2).item()
        
        M_gt = torch.abs(X_gt).cpu().numpy()
        M_pred = torch.abs(X_pred).cpu().numpy()
        
        # Simple SSIM
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        
        M_max = max(np.max(M_gt), np.max(M_pred))
        s_val = ssim_fn(M_gt / M_max, M_pred / M_max, data_range=1.0)
        p_val = psnr_fn(M_gt / M_max, M_pred / M_max, data_range=1.0)
        return s_val, p_val, phase_mse

    ssim_def, psnr_def, _ = calc_metrics(X_defocused, torch.zeros_like(Phi_true))
    ssim_zs, psnr_zs, mse_zs = calc_metrics(X_zero_shot, phi_pred)
    ssim_tta, psnr_tta, mse_tta = calc_metrics(X_tta, phi_tta)
    
    print("\n" + "="*55)
    print("DEMO PERFORMANCE SUMMARY:")
    print("="*55)
    print(f"  Defocused (Input)    : SSIM = {ssim_def:.4f} | PSNR = {psnr_def:.2f} dB")
    print(f"  PolyPhaseNet (0-shot): SSIM = {ssim_zs:.4f} | PSNR = {psnr_zs:.2f} dB | Phase MSE = {mse_zs:.4f} rad^2")
    print(f"  PolyPhaseNet + TTA   : SSIM = {ssim_tta:.4f} | PSNR = {psnr_tta:.2f} dB | Phase MSE = {mse_tta:.4f} rad^2")
    print("="*55 + "\n")
    
    # Save visual comparison
    os.makedirs("assets", exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(torch.abs(X_gt).cpu().numpy(), cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[1].imshow(torch.abs(X_defocused).cpu().numpy(), cmap='gray')
    axes[1].set_title(f"Defocused\nSSIM: {ssim_def:.4f}")
    axes[2].imshow(torch.abs(X_zero_shot).cpu().numpy(), cmap='gray')
    axes[2].set_title(f"PolyPhaseNet (Zero-Shot)\nSSIM: {ssim_zs:.4f}")
    axes[3].imshow(torch.abs(X_tta).cpu().numpy(), cmap='gray')
    axes[3].set_title(f"PolyPhaseNet + TTA\nSSIM: {ssim_tta:.4f}")
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    out_img = "assets/demo_result.png"
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"[+] Visual result saved to {out_img}")

if __name__ == '__main__':
    main()
