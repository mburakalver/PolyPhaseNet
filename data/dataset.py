import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
import numpy as np

class SARDataset(Dataset):
    def __init__(self, data_dir="dataset", split='train', seed=42, dynamic_tier=None):
        split_file = Path(f"{split}_split.txt")
        if not split_file.exists():
            candidate = Path(__file__).parent / "splits" / f"{split}_split.txt"
            if candidate.exists():
                split_file = candidate
            else:
                raise FileNotFoundError(f"Split file {split}_split.txt not found in current directory or {candidate.parent}.")
            
        with open(split_file, 'r') as f:
            filenames = f.read().splitlines()
            
        self.files = [Path(data_dir) / f for f in filenames]
        self.dynamic_tier = dynamic_tier
        
        # Original sinusoidal parameters
        self.tiers = {
            "mild": (0.5, 1.5, 0.05, 0.5, 0.05, 0.2),
            "moderate": (1.0, 3.0, 0.1, 1.0, 0.1, 0.5),
            "severe": (2.0, 8.0, 1.0, 4.0, 0.1, 1.0)
        }
        # Polynomial standard deviation limits
        self.poly_stds = {
            "mild": 0.7,
            "moderate": 1.5,
            "severe": 3.0
        }
        
        print(f"Loaded {len(self.files)} samples for {split} split from {data_dir}/")
        if self.dynamic_tier:
            print(f"--> Domain Randomization ENABLED (Tier: {self.dynamic_tier}). 50% Sinusoidal / 50% Polynomial.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        c_patch = data['clean'] # (256, 256) Complex
        
        if self.dynamic_tier is None:
            # Legacy static behavior
            S_patch = data['S_input'].unsqueeze(0)
            c_patch = c_patch.unsqueeze(0)
            p_patch = data['true_phase']
            return S_patch, c_patch, p_patch
            
        # --- On-the-Fly Domain Randomization ---
        clean_np = c_patch.numpy()
        Az, Rg = clean_np.shape
        
        # 50% chance Sinusoidal, 50% chance Polynomial
        if random.random() < 0.5:
            # 1. Sinusoidal Error (from original paper logic)
            br_min, br_max, slope_min, slope_max, amp2_min, amp2_max = self.tiers[self.dynamic_tier]
            t_a = np.linspace(-1, 1, Az)[:, None]
            r = np.linspace(-1, 1, Rg)[None, :]
            freq1 = random.uniform(1.0, 5.0)
            freq2 = random.uniform(3.0, 8.0)
            amp1 = random.uniform(0.5, 2.0) 
            amp2 = random.uniform(amp2_min, amp2_max) 
            base_error = amp1 * np.sin(freq1 * np.pi * t_a) + amp2 * np.cos(freq2 * np.pi * t_a)
            base_range = random.uniform(br_min, br_max)
            slope = random.uniform(slope_min, slope_max)
            range_modifier = base_range + slope * r 
            phi_error = range_modifier * base_error
        else:
            # 2. Polynomial Error (Quadratic or Cubic)
            std_scale = self.poly_stds[self.dynamic_tier]
            x = np.linspace(-1, 1, Az)
            y = np.linspace(-1, 1, Rg)
            X, Y = np.meshgrid(y, x)
            
            if random.random() < 0.5:
                # Quadratic
                c1, c2, c3 = np.random.randn(3)
                phi_error = c1 * (Y**2) + c2 * (X**2) + c3 * (X * Y)
            else:
                # Cubic
                c1, c2, c3, c4 = np.random.randn(4)
                phi_error = c1 * (Y**3) + c2 * (X**3) + c3 * (Y**2)*X + c4 * Y*(X**2)
                
            phi_error = phi_error - np.mean(phi_error)
            phi_error = (phi_error / (np.std(phi_error) + 1e-8)) * (std_scale * np.pi)

        # Apply phase error in the Azimuth-Fourier (Doppler) domain
        # FFT along Azimuth (dim=0 in numpy if shape is Az, Rg)
        S_input_clean = np.fft.fft(clean_np, axis=0)
        S_input_corrupted = S_input_clean * np.exp(1j * phi_error)
        
        S_patch = torch.from_numpy(S_input_corrupted).unsqueeze(0).to(torch.complex64)
        p_patch = torch.from_numpy(phi_error).float()
        c_patch = c_patch.unsqueeze(0)
        
        return S_patch, c_patch, p_patch
