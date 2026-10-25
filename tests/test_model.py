import unittest
from pathlib import Path
import torch
from polyphasenet.model import PolyPhaseNet


class TestPolyPhaseNetModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PolyPhaseNet(
            patch_size=256,
            az_points=64, rg_points=16,
            degree=3, mode='tensor_grid'
        ).to(self.device)
        self.model.eval()

    def test_forward_shapes_single(self):
        """Test output shapes for single complex slow-time input patch."""
        S = torch.randn(1, 1, 256, 256, dtype=torch.complex64, device=self.device)
        with torch.no_grad():
            X_corr, phi_est, coeffs = self.model(S)

        self.assertEqual(X_corr.shape, (1, 1, 256, 256))
        self.assertEqual(phi_est.shape, (1, 1, 256, 256))
        self.assertEqual(coeffs.shape, (1, 64 * 16))
        self.assertTrue(torch.is_complex(X_corr))
        self.assertFalse(torch.is_complex(phi_est))
        self.assertFalse(torch.isnan(X_corr).any())
        self.assertFalse(torch.isnan(phi_est).any())

    def test_forward_shapes_batch(self):
        """Test output shapes for batched input (B=2)."""
        S = torch.randn(2, 1, 256, 256, dtype=torch.complex64, device=self.device)
        with torch.no_grad():
            X_corr, phi_est, coeffs = self.model(S)

        self.assertEqual(X_corr.shape, (2, 1, 256, 256))
        self.assertEqual(phi_est.shape, (2, 1, 256, 256))
        self.assertEqual(coeffs.shape, (2, 64 * 16))

    def test_phase_piston_invariance(self):
        """Phase gradient features should be invariant to constant phase offset exp(j * theta)."""
        S = torch.randn(1, 1, 256, 256, dtype=torch.complex64, device=self.device)
        theta = 1.732  # Arbitrary constant radian offset
        S_shifted = S * torch.exp(1j * torch.tensor(theta, device=self.device))

        with torch.no_grad():
            _, phi_1, coeffs_1 = self.model(S)
            _, phi_2, coeffs_2 = self.model(S_shifted)

        torch.testing.assert_close(
            coeffs_1, coeffs_2,
            atol=1e-4, rtol=1e-4,
            msg="Control matrix prediction must be invariant to global constant phase piston"
        )

    def test_checkpoint_loading(self):
        """Verify that pre-trained weights load cleanly without missing keys."""
        ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "best_polyphasenet_2d_full.pth"
        if not ckpt_path.exists():
            self.skipTest(f"Checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(missing), 0, f"Missing state_dict keys: {missing}")


if __name__ == "__main__":
    unittest.main()
