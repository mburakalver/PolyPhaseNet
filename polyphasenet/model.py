import torch
import torch.nn as nn
try:
    from .bspline import cox_de_boor_basis
except ImportError:
    try:
        from bspline import cox_de_boor_basis
    except ImportError:
        from polyphasenet.bspline import cox_de_boor_basis

class PolyPhaseNet(nn.Module):
    def __init__(self, patch_size=256, az_points=64, rg_points=16, degree=3, mode='tensor_grid'):
        super(PolyPhaseNet, self).__init__()
        self.patch_size = patch_size
        self.az_points = az_points
        self.rg_points = rg_points
        self.degree = degree
        self.mode = mode
        
        # Precompute clamped uniform B-Spline basis matrices B_az (A x K) and B_rg (R x L)
        B_az = cox_de_boor_basis(patch_size, az_points, degree=degree)
        B_rg = cox_de_boor_basis(patch_size, rg_points, degree=degree)
        self.register_buffer('B_az', B_az)
        self.register_buffer('B_rg', B_rg)
        
        # Lightweight CNN backbone to extract global context
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        out_dim = (az_points * rg_points) if mode == 'tensor_grid' else (az_points + rg_points)
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
        )
        # Global phase amplitude multiplier
        self.phase_scale = nn.Parameter(torch.tensor(50.0))

    def load_state_dict(self, state_dict, strict=True):
        # Automatically adapt if loading a legacy rank-1 (K+L=80) checkpoint vs full 2D (K*L=1024) checkpoint
        if 'regressor.2.weight' in state_dict:
            ckpt_out = state_dict['regressor.2.weight'].shape[0]
            if ckpt_out == self.az_points + self.rg_points and self.mode != 'rank1_outer':
                self.mode = 'rank1_outer'
                self.regressor[2] = nn.Linear(128, ckpt_out).to(state_dict['regressor.2.weight'].device)
            elif ckpt_out == self.az_points * self.rg_points and self.mode != 'tensor_grid':
                self.mode = 'tensor_grid'
                self.regressor[2] = nn.Linear(128, ckpt_out).to(state_dict['regressor.2.weight'].device)
        return super().load_state_dict(state_dict, strict=False)

    def forward(self, S, oracle_grad=None):
        """
        S: Complex tensor of shape (B, 1, H, W)
        oracle_grad: Optional tuple (grad_az_true, grad_rg_true) for Oracle decoupling experiment
        """
        B, C, H, W = S.shape
        
        S_mag = torch.abs(S)
        
        if oracle_grad is not None:
            phase_grad_az, phase_grad_rg = oracle_grad
        else:
            # Phase gradient in azimuth with boundary masking (no circular wrap-around artifact)
            S_shifted_az = torch.roll(S, shifts=-1, dims=2)
            phase_grad_az = torch.angle(S * torch.conj(S_shifted_az))
            phase_grad_az[:, :, -1, :] = 0.0
            
            # Phase gradient in range with boundary masking
            S_shifted_rg = torch.roll(S, shifts=-1, dims=3)
            phase_grad_rg = torch.angle(S * torch.conj(S_shifted_rg))
            phase_grad_rg[:, :, :, -1] = 0.0
        
        x_input = torch.cat([S_mag, phase_grad_az, phase_grad_rg], dim=1) # (B, 3, H, W)
        
        x = self.features(x_input) # (B, 256, 1, 1)
        x = torch.flatten(x, 1)   # (B, 256)
        
        # Predict control points
        coeffs = self.regressor(x)
        
        if self.mode == 'tensor_grid':
            C_mat = coeffs.view(B, self.az_points, self.rg_points) * self.phase_scale
            if H == self.patch_size and W == self.patch_size:
                phi_pred = torch.einsum('ak,bkl,rl->bar', self.B_az, C_mat, self.B_rg)
            else:
                B_az_dyn = cox_de_boor_basis(H, self.az_points, degree=self.degree).to(S.device)
                B_rg_dyn = cox_de_boor_basis(W, self.rg_points, degree=self.degree).to(S.device)
                phi_pred = torch.einsum('ak,bkl,rl->bar', B_az_dyn, C_mat, B_rg_dyn)
        else:
            az_ctrl = coeffs[:, :self.az_points].unsqueeze(1)
            rg_ctrl = coeffs[:, self.az_points:].unsqueeze(1) + 1.0
            az_full = F.interpolate(az_ctrl, size=H, mode='linear', align_corners=True).view(B, H, 1)
            rg_full = F.interpolate(rg_ctrl, size=W, mode='linear', align_corners=True).view(B, 1, W)
            phi_pred = az_full * rg_full * self.phase_scale
        
        # Add channel dimension to match S: (B, 1, H, W)
        phi_pred = phi_pred.unsqueeze(1)
        
        # Apply phase correction: S_corr = S * exp(-j * phi_pred)
        S_corr = S * torch.exp(-1j * phi_pred)
        
        # Reconstruct image
        X_pred = torch.fft.fft(S_corr, dim=2)

        return X_pred, phi_pred, coeffs

class PolyPhaseNetAblation(nn.Module):
    def __init__(self, patch_size=256, az_points=64, rg_points=16, degree=3, mode='tensor_grid'):
        super(PolyPhaseNetAblation, self).__init__()
        self.patch_size = patch_size
        self.az_points = az_points
        self.rg_points = rg_points
        self.degree = degree
        self.mode = mode
        
        B_az = cox_de_boor_basis(patch_size, az_points, degree=degree)
        B_rg = cox_de_boor_basis(patch_size, rg_points, degree=degree)
        self.register_buffer('B_az', B_az)
        self.register_buffer('B_rg', B_rg)
        
        # Simple feed-forward CNN backbone to extract global context
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1), # 2 channels for Real/Imag
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        out_dim = (az_points * rg_points) if mode == 'tensor_grid' else (az_points + rg_points)
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
        )
        self.phase_scale = nn.Parameter(torch.tensor(50.0))

    def load_state_dict(self, state_dict, strict=True):
        if 'regressor.2.weight' in state_dict:
            ckpt_out = state_dict['regressor.2.weight'].shape[0]
            if ckpt_out == self.az_points + self.rg_points and self.mode != 'rank1_outer':
                self.mode = 'rank1_outer'
                self.regressor[2] = nn.Linear(128, ckpt_out).to(state_dict['regressor.2.weight'].device)
            elif ckpt_out == self.az_points * self.rg_points and self.mode != 'tensor_grid':
                self.mode = 'tensor_grid'
                self.regressor[2] = nn.Linear(128, ckpt_out).to(state_dict['regressor.2.weight'].device)
        return super().load_state_dict(state_dict, strict=False)

    def forward(self, S):
        B, C, H, W = S.shape
        x_input = torch.cat([S.real, S.imag], dim=1) # (B, 2, H, W)
        x = self.features(x_input)
        x = torch.flatten(x, 1)
        coeffs = self.regressor(x)
        
        if self.mode == 'tensor_grid':
            C_mat = coeffs.view(B, self.az_points, self.rg_points) * self.phase_scale
            phi_pred = torch.einsum('ak,bkl,rl->bar', self.B_az, C_mat, self.B_rg)
        else:
            az_ctrl = coeffs[:, :self.az_points].unsqueeze(1)
            rg_ctrl = coeffs[:, self.az_points:].unsqueeze(1) + 1.0
            az_full = F.interpolate(az_ctrl, size=H, mode='linear', align_corners=True).view(B, H, 1)
            rg_full = F.interpolate(rg_ctrl, size=W, mode='linear', align_corners=True).view(B, 1, W)
            phi_pred = az_full * rg_full * self.phase_scale
            
        phi_pred = phi_pred.unsqueeze(1)
        S_corr = S * torch.exp(-1j * phi_pred)
        X_pred = torch.fft.fft(S_corr, dim=2)
        return X_pred, phi_pred, coeffs

class SOTAPhaseNet(nn.Module):
    """
    SOTA Baseline (FocusNet/PAFnet representation)
    Takes Real and Imaginary parts of the SAR signal directly.
    Demonstrates the Phase-Wrapping Trap.
    """
    def __init__(self, patch_size=256, az_points=64, rg_points=16):
        super(SOTAPhaseNet, self).__init__()
        self.patch_size = patch_size
        self.az_points = az_points
        self.rg_points = rg_points
        self.num_control_points = az_points * rg_points
        
        # SOTA models typically take 2 channels: Real and Imaginary parts
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_control_points)
        )
        
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.zero_()
        
    def forward(self, S):
        # S: (B, 1, H, W) Complex
        
        # SOTA Input: Real and Imaginary components
        real_part = S.real
        imag_part = S.imag
        x = torch.cat([real_part, imag_part], dim=1) # (B, 2, H, W)
        
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        
        coeffs = self.fc(feats)
        coeffs = coeffs.view(-1, self.az_points, self.rg_points)
        
        coeffs_az = torch.nn.functional.interpolate(
            coeffs.unsqueeze(1), size=(self.patch_size, self.rg_points), mode='bilinear', align_corners=True
        ).squeeze(1)
        
        phi_surface = torch.nn.functional.interpolate(
            coeffs_az.unsqueeze(1), size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=True
        ).squeeze(1)
        
        S_corr = S.squeeze(1) * torch.exp(-1j * phi_surface)
        S_corr_image = torch.fft.fft(S_corr, dim=1)
        X_pred = S_corr_image.unsqueeze(1)
        
        return X_pred, phi_surface, coeffs

class LEAModule(nn.Module):
    def __init__(self, channels):
        super(LEAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // 8), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(1, channels // 8), channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class LEAPNet(nn.Module):
    """
    LEAP-Net Baseline (2024/2025)
    Incorporates Lightweight Efficient Attention (LEA) module.
    """
    def __init__(self, patch_size=256, az_points=64, rg_points=16):
        super(LEAPNet, self).__init__()
        self.patch_size = patch_size
        self.az_points = az_points
        self.rg_points = rg_points
        self.num_control_points = az_points * rg_points
        
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            LEAModule(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            LEAModule(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            LEAModule(128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_control_points)
        )
        
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.zero_()

    def forward(self, S):
        real_part = S.real
        imag_part = S.imag
        x = torch.cat([real_part, imag_part], dim=1) 
        
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        
        coeffs = self.fc(feats)
        coeffs = coeffs.view(-1, self.az_points, self.rg_points)
        
        coeffs_az = torch.nn.functional.interpolate(
            coeffs.unsqueeze(1), size=(self.patch_size, self.rg_points), mode='bilinear', align_corners=True
        ).squeeze(1)
        
        phi_surface = torch.nn.functional.interpolate(
            coeffs_az.unsqueeze(1), size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=True
        ).squeeze(1)
        
        S_corr = S.squeeze(1) * torch.exp(-1j * phi_surface)
        S_corr_image = torch.fft.fft(S_corr, dim=1)
        X_pred = S_corr_image.unsqueeze(1)
        
        return X_pred, phi_surface, coeffs

class SOTAPhaseNet_Raw(nn.Module):
    """
    SOTA Baseline without B-Spline interpolation head.
    Assumes 1D space-invariant phase error by predicting a 1D vector of size patch_size
    and broadcasting it across the range dimension.
    """
    def __init__(self, patch_size=256):
        super(SOTAPhaseNet_Raw, self).__init__()
        self.patch_size = patch_size
        
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.patch_size)
        )
        
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.zero_()
        
    def forward(self, S):
        real_part = S.real
        imag_part = S.imag
        x = torch.cat([real_part, imag_part], dim=1) 
        
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        
        phi_1d = self.fc(feats) # (B, 256)
        phi_surface = phi_1d.unsqueeze(2).expand(-1, -1, self.patch_size) # (B, 256, 256)
        
        S_corr = S.squeeze(1) * torch.exp(-1j * phi_surface)
        S_corr_image = torch.fft.fft(S_corr, dim=1)
        X_pred = S_corr_image.unsqueeze(1)
        
        return X_pred, phi_surface, phi_1d

class LEAPNet_Raw(nn.Module):
    """
    LEAP-Net Baseline without B-Spline interpolation head.
    Predicts a 1D space-invariant phase error vector.
    """
    def __init__(self, patch_size=256):
        super(LEAPNet_Raw, self).__init__()
        self.patch_size = patch_size
        
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            LEAModule(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            LEAModule(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            LEAModule(128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.patch_size)
        )
        
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.zero_()

    def forward(self, S):
        real_part = S.real
        imag_part = S.imag
        x = torch.cat([real_part, imag_part], dim=1) 
        
        feats = self.features(x)
        feats = feats.view(feats.size(0), -1)
        
        phi_1d = self.fc(feats) # (B, 256)
        phi_surface = phi_1d.unsqueeze(2).expand(-1, -1, self.patch_size) # (B, 256, 256)
        
        S_corr = S.squeeze(1) * torch.exp(-1j * phi_surface)
        S_corr_image = torch.fft.fft(S_corr, dim=1)
        X_pred = S_corr_image.unsqueeze(1)
        
        return X_pred, phi_surface, phi_1d

class SupervisedBlackBox(nn.Module):
    """
    A simple end-to-end supervised network representing black-box architectures like SARAF.
    It takes the defocused complex image (real, imag) and directly outputs the focused magnitude image.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU() # Magnitude is non-negative
        )
        
    def forward(self, S):
        x = torch.cat([S.real, S.imag], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

class DeepUnfoldingAF(nn.Module):
    """
    Properly parameterized Deep Unfolding Network (AF-DCDU-Net style, T=3 stages)
    where each stage estimates a residual 2D phase update from current image magnitude
    and complex phase history residuals.
    """
    def __init__(self, num_stages=3, patch_size=256):
        super().__init__()
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((16, 8)),
                nn.Conv2d(64, 1, kernel_size=3, padding=1)
            )
            for _ in range(num_stages)
        ])

    def forward(self, S):
        B, C, H, W = S.shape
        phi_est = torch.zeros((B, 1, H, W), device=S.device, dtype=torch.float32)
        for stage in self.stages:
            S_corr = S * torch.exp(-1j * phi_est)
            X_curr = torch.fft.fft(S_corr, dim=2)
            X_mag = torch.abs(X_curr)
            X_mag_norm = X_mag / (torch.amax(X_mag, dim=(2, 3), keepdim=True) + 1e-8)
            inp = torch.cat([X_mag_norm, S_corr.real, S_corr.imag], dim=1)
            delta_coarse = stage(inp)
            delta_phi = F.interpolate(
                delta_coarse, size=(H, W), mode='bilinear', align_corners=True
            )
            phi_est = phi_est + delta_phi

        S_corr_final = S * torch.exp(-1j * phi_est)
        X_final = torch.fft.fft(S_corr_final, dim=2)
        return X_final, phi_est.squeeze(1), None

PAFNet = SOTAPhaseNet
PAFNet_Raw = SOTAPhaseNet_Raw
SupervisedCNN = SupervisedBlackBox

def entropy_loss(X):
    """Computes Shannon image entropy."""
    I = torch.abs(X) ** 2
    p = I / (torch.sum(I) + 1e-12)
    return -torch.sum(p * torch.log(p + 1e-12))

def sharpness_metric(X):
    """Computes normalized intensity squared-sum focus concentration (Zhang et al. 2021)."""
    I = torch.abs(X) ** 2
    p = I / (torch.sum(I) + 1e-12)
    return torch.sum(p ** 2)

def subpixel_align_and_compute_metrics(M_gt, M_pred):
    """Sub-pixel phase cross-correlation alignment and metric calculation."""
    import numpy as np
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    if isinstance(M_gt, torch.Tensor):
        M_gt = M_gt.detach().cpu().numpy()
    if isinstance(M_pred, torch.Tensor):
        M_pred = M_pred.detach().cpu().numpy()
        
    M_gt = np.abs(M_gt)
    M_pred = np.abs(M_pred)
    M_max = max(np.max(M_gt), np.max(M_pred), 1e-8)
    
    s = ssim_fn(M_gt / M_max, M_pred / M_max, data_range=1.0)
    p = psnr_fn(M_gt / M_max, M_pred / M_max, data_range=1.0)
    return s, p

