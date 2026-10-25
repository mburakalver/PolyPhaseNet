import unittest
import torch
from polyphasenet.bspline import cox_de_boor_basis


class TestBSplineBasis(unittest.TestCase):
    def test_basis_shapes(self):
        """Test output dimensions for typical SAR patch and knot grid sizes."""
        for n_samples, n_ctrl in [(256, 64), (256, 16), (512, 64), (128, 32)]:
            B = cox_de_boor_basis(n_samples=n_samples, n_ctrl=n_ctrl, degree=3)
            self.assertEqual(B.shape, (n_samples, n_ctrl))
            self.assertEqual(B.dtype, torch.float32)

    def test_partition_of_unity(self):
        """Test that sum of basis functions equals 1 across all evaluation points."""
        for degree in [1, 2, 3]:
            B = cox_de_boor_basis(n_samples=256, n_ctrl=64, degree=degree)
            row_sums = B.sum(dim=1)
            torch.testing.assert_close(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-5,
                rtol=1e-5,
                msg=f"Partition of unity failed for degree {degree}"
            )

    def test_non_negativity(self):
        """Test that basis functions are strictly non-negative."""
        B = cox_de_boor_basis(n_samples=256, n_ctrl=64, degree=3)
        self.assertTrue((B >= -1e-7).all())

    def test_clamped_boundary_conditions(self):
        """Test that clamped knot vectors anchor at boundary endpoints."""
        B = cox_de_boor_basis(n_samples=256, n_ctrl=64, degree=3)
        # At u = 0, first basis should be 1.0, others 0.0
        self.assertAlmostEqual(B[0, 0].item(), 1.0, places=4)
        self.assertAlmostEqual(B[0, 1:].sum().item(), 0.0, places=4)
        # At u = 1, last basis should be 1.0, others 0.0
        self.assertAlmostEqual(B[-1, -1].item(), 1.0, places=4)
        self.assertAlmostEqual(B[-1, :-1].sum().item(), 0.0, places=4)

    def test_tensor_product_surface(self):
        """Test 2D tensor-product surface projection Phi = B_az @ C @ B_rg.T."""
        Az, Rg = 256, 256
        K, L = 64, 16
        B_az = cox_de_boor_basis(Az, K, degree=3)
        B_rg = cox_de_boor_basis(Rg, L, degree=3)
        C = torch.randn(K, L)

        phi_2d = B_az @ C @ B_rg.t()
        self.assertEqual(phi_2d.shape, (Az, Rg))
        self.assertFalse(torch.isnan(phi_2d).any())


if __name__ == "__main__":
    unittest.main()
