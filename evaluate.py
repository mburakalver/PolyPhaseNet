"""
Evaluation and Benchmarking Script for PolyPhaseNet (IEEE TAES).

Computes quantitative autofocus metrics (SSIM, PSNR, Phase MSE, Image Entropy) across
Mild, Moderate, and Severe 2D non-separable phase error tiers.

Usage:
------
# Evaluate feed-forward zero-shot PolyPhaseNet on Moderate tier:
python evaluate.py --checkpoint checkpoints/best_polyphasenet_2d_full.pth --tier moderate

# Evaluate with Episodic Safeguarded TTA:
python evaluate.py --checkpoint checkpoints/best_polyphasenet_2d_full.pth --tier severe --use_tta --tta_steps 6
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from polyphasenet.model import PolyPhaseNet
from polyphasenet.tta import episodic_safeguarded_tta


def calculate_metrics(img_pred, img_gt):
    """Computes SSIM and PSNR between magnitude reconstructed and ground-truth SLC images."""
    pred_mag = np.abs(img_pred)
    gt_mag = np.abs(img_gt)
    
    # Normalize to [0, 1]
    p_norm = (pred_mag - pred_mag.min()) / (pred_mag.max() - pred_mag.min() + 1e-12)
    g_norm = (gt_mag - gt_mag.min()) / (gt_mag.max() - gt_mag.min() + 1e-12)
    
    val_ssim = ssim(g_norm, p_norm, data_range=1.0)
    val_psnr = psnr(g_norm, p_norm, data_range=1.0)
    return float(val_ssim), float(val_psnr)


def calculate_phase_mse(phi_pred, phi_true):
    """Piston-aligned Phase Mean Squared Error (rad^2)."""
    diff = phi_pred - phi_true
    # Remove unobservable global constant phase piston
    diff_centered = diff - np.mean(diff)
    return float(np.mean(diff_centered ** 2))


def generate_synthetic_evaluation_patch(patch_size=256, tier="moderate", seed=None):
    """Generates synthetic 2D non-separable coupled phase error realization."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    Az, Rg = patch_size, patch_size
    # Synthetic point targets + distributed background clutter
    clean = torch.randn(Az, Rg, dtype=torch.complex64) * 0.1
    for _ in range(15):
        r_idx = np.random.randint(20, Az - 20)
        c_idx = np.random.randint(20, Rg - 20)
        clean[r_idx, c_idx] = float(np.random.uniform(5.0, 12.0))

    scales = {"mild": 1.5, "moderate": 3.0, "severe": 6.5}
    s = scales.get(tier, 3.0)
    
    u = np.linspace(-1, 1, Az)[:, None]
    v = np.linspace(-1, 1, Rg)[None, :]
    
    # Coupled 2D polynomial + multi-harmonic azimuthal flutter
    phi_2d = s * (u**3 * v + 0.5 * u**2 * v**2 + np.sin(3 * np.pi * u) * (1 + 0.3 * v))
    phi_tensor = torch.from_numpy(phi_2d).float()

    S_clean = torch.fft.ifft(clean, dim=0)
    S_err = S_clean * torch.exp(1j * phi_tensor)
    
    return S_err.unsqueeze(0).unsqueeze(0), clean.numpy(), phi_2d


def main():
    parser = argparse.ArgumentParser(description="PolyPhaseNet Benchmark Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_polyphasenet_2d_full.pth",
                        help="Path to pre-trained weights")
    parser.add_argument("--tier", type=str, default="moderate", choices=["mild", "moderate", "severe"],
                        help="Severity tier to evaluate")
    parser.add_argument("--samples", type=int, default=10, help="Number of evaluation patches")
    parser.add_argument("--use_tta", action="store_true", help="Apply Episodic Safeguarded TTA")
    parser.add_argument("--tta_steps", type=int, default=6, help="TTA adaptation iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device (cuda/cpu)")
    args = parser.parse_args()

    print("===================================================================")
    print(f"[*] Evaluating PolyPhaseNet on Tier: {args.tier.upper()} ({args.samples} patches)")
    print(f"[*] Device: {args.device} | TTA: {'ENABLED (' + str(args.tta_steps) + ' steps)' if args.use_tta else 'DISABLED'}")
    print("===================================================================")

    device = torch.device(args.device)
    model = PolyPhaseNet(patch_size=256, az_points=64, rg_points=16, degree=3).to(device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"[+] Loading weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[!] Warning: Checkpoint {ckpt_path} not found. Running with initialized weights.")

    model.eval()

    defoc_ssims, defoc_psnrs = [], []
    pred_ssims, pred_psnrs, pred_mses = [], [], []

    for idx in range(args.samples):
        S_in, clean_gt, phi_gt = generate_synthetic_evaluation_patch(patch_size=256, tier=args.tier, seed=1000 + idx)
        S_in = S_in.to(device)

        # Defocused observation baseline
        with torch.no_grad():
            X_def = torch.fft.fft(S_in, dim=2).squeeze().cpu().numpy()
        d_ssim, d_psnr = calculate_metrics(X_def, clean_gt)
        defoc_ssims.append(d_ssim)
        defoc_psnrs.append(d_psnr)

        if args.use_tta:
            X_rec, phi_rec = episodic_safeguarded_tta(model, S_in, steps=args.tta_steps)
            X_out = X_rec.squeeze().cpu().numpy()
            phi_out = phi_rec.squeeze().cpu().numpy()
        else:
            with torch.no_grad():
                X_rec, phi_rec, _ = model(S_in)
            X_out = X_rec.squeeze().cpu().numpy()
            phi_out = phi_rec.squeeze().cpu().numpy()

        p_ssim, p_psnr = calculate_metrics(X_out, clean_gt)
        p_mse = calculate_phase_mse(phi_out, phi_gt)

        pred_ssims.append(p_ssim)
        pred_psnrs.append(p_psnr)
        pred_mses.append(p_mse)

    print("\n-------------------------------------------------------------------")
    print("BENCHMARK RESULTS SUMMARY:")
    print("-------------------------------------------------------------------")
    print(f"  Defocused (Input)    : SSIM = {np.mean(defoc_ssims):.4f} +/- {np.std(defoc_ssims):.4f} | PSNR = {np.mean(defoc_psnrs):.2f} dB")
    mode_str = "PolyPhaseNet + TTA   " if args.use_tta else "PolyPhaseNet (0-shot)"
    print(f"  {mode_str}: SSIM = {np.mean(pred_ssims):.4f} +/- {np.std(pred_ssims):.4f} | PSNR = {np.mean(pred_psnrs):.2f} dB | Phase MSE = {np.mean(pred_mses):.4f} rad^2")
    print("-------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
