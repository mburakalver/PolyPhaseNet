import unittest
import torch
from polyphasenet.classical import pga_1d_autofocus, pga_2d_autofocus


class TestClassicalAutofocus(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create a synthetic point target scene with a 1D quadratic phase error
        Az, Rg = 128, 128
        X_clean = torch.zeros((Az, Rg), dtype=torch.complex64, device=self.device)
        X_clean[Az // 2, Rg // 2] = 10.0 + 0j
        X_clean[Az // 4, Rg // 4] = 8.0 + 0j

        u = torch.linspace(-1, 1, Az, device=self.device).unsqueeze(1)
        phi_1d = 2.5 * (u ** 2)
        S_clean = torch.fft.ifft(X_clean, dim=0)
        self.S_in = S_clean * torch.exp(1j * phi_1d)

    def test_pga_1d(self):
        """Test classical 1D PGA autofocus."""
        X_corr, phi_est = pga_1d_autofocus(self.S_in, num_iters=4, win_size=16)
        self.assertEqual(X_corr.shape, (128, 128))
        self.assertEqual(phi_est.shape, (128, 128))
        self.assertFalse(torch.isnan(X_corr).any())
        self.assertFalse(torch.isnan(phi_est).any())

    def test_pga_2d(self):
        """Test sub-band cascaded 2D PGA autofocus."""
        X_corr, phi_est = pga_2d_autofocus(self.S_in, num_iters=3, num_subbands=2, win_size=16)
        self.assertEqual(X_corr.shape, (128, 128))
        self.assertEqual(phi_est.shape, (128, 128))
        self.assertFalse(torch.isnan(X_corr).any())
        self.assertFalse(torch.isnan(phi_est).any())


if __name__ == "__main__":
    unittest.main()
