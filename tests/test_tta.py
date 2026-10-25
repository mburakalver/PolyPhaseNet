import unittest
import torch
from polyphasenet.model import PolyPhaseNet
from polyphasenet.tta import episodic_safeguarded_tta


class TestEpisodicTTA(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PolyPhaseNet(
            patch_size=256,
            az_points=64, rg_points=16,
            degree=3, mode='tensor_grid'
        ).to(self.device)
        self.model.eval()

    def test_tta_output_shapes(self):
        """Test output shapes for TTA."""
        S = torch.randn(1, 1, 256, 256, dtype=torch.complex64, device=self.device)
        best_X, best_phi = episodic_safeguarded_tta(self.model, S, steps=2, lr=1e-2)

        self.assertEqual(best_X.shape, (1, 1, 256, 256))
        self.assertEqual(best_phi.shape, (1, 1, 256, 256))
        self.assertTrue(torch.is_complex(best_X))
        self.assertFalse(torch.is_complex(best_phi))

    def test_backbone_frozen(self):
        """Verify that model backbone weights are strictly frozen during TTA."""
        S = torch.randn(1, 1, 256, 256, dtype=torch.complex64, device=self.device)
        orig_weights = {name: param.clone() for name, param in self.model.named_parameters()}

        _ = episodic_safeguarded_tta(self.model, S, steps=4, lr=1e-2)

        for name, param in self.model.named_parameters():
            torch.testing.assert_close(
                param, orig_weights[name],
                atol=1e-7, rtol=1e-7,
                msg=f"Model parameter '{name}' was modified during TTA"
            )

    def test_entropy_verification_gate(self):
        """Verify that the entropy verification gate prevents entropy increase."""
        # Create a defocused synthetic observation
        torch.manual_seed(42)
        X_clean = torch.randn(1, 1, 256, 256, dtype=torch.complex64, device=self.device)
        # Add high-order phase error
        Az, Rg = 256, 256
        u = torch.linspace(-1, 1, Az, device=self.device).view(1, 1, Az, 1)
        v = torch.linspace(-1, 1, Rg, device=self.device).view(1, 1, 1, Rg)
        phi_err = 3.0 * (u ** 3) * (v ** 2) + 2.0 * (u ** 2)
        S_err = torch.fft.ifft(X_clean, dim=2) * torch.exp(1j * phi_err)

        # Defocused entropy
        X_def = torch.fft.fft(S_err, dim=2)
        I_def = torch.abs(X_def) ** 2
        p_def = I_def / (torch.sum(I_def) + 1e-12)
        ent_def = float(-torch.sum(p_def * torch.log(p_def + 1e-12)).item())

        best_X, _ = episodic_safeguarded_tta(self.model, S_err, steps=5, lr=1.5e-2)
        I_best = torch.abs(best_X) ** 2
        p_best = I_best / (torch.sum(I_best) + 1e-12)
        ent_best = float(-torch.sum(p_best * torch.log(p_best + 1e-12)).item())

        self.assertLessEqual(ent_best, ent_def + 1e-5, "Entropy Verification Gate must ensure non-increasing entropy")


if __name__ == "__main__":
    unittest.main()
