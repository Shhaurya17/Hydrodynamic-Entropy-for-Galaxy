"""
Unit test suite for hydro_spectral package.
"""

import unittest
import numpy as np
import pandas as pd

import hydro_spectral as hs

class TestHydroSpectral(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.N = 1000
        self.pos = np.random.uniform(-50, 50, (self.N, 3))
        self.vel = np.random.normal(0, 100, (self.N, 3))
        self.center = np.zeros(3)
        self.Rmax = 100.0

    def test_construct_velocity_grid(self):
        vgrid = hs.construct_velocity_grid(self.pos, self.vel, self.center, self.Rmax, Ng=16)
        self.assertIsNotNone(vgrid)
        self.assertEqual(vgrid.shape, (16, 16, 16, 3))

    def test_compute_energy_spectrum(self):
        vgrid = hs.construct_velocity_grid(self.pos, self.vel, self.center, self.Rmax, Ng=16)
        Ek = hs.compute_energy_spectrum(vgrid, Ng=16)
        self.assertGreater(len(Ek), 0)
        self.assertGreaterEqual(np.min(Ek), 0.0)

    def test_spectral_entropy_bounds(self):
        # Delta function spectrum (all energy at 1 bin) -> Entropy should be 0
        Ek_single = np.zeros(10)
        Ek_single[2] = 100.0
        S_single = hs.spectral_entropy(Ek_single)
        self.assertAlmostEqual(S_single, 0.0)

        # Flat equipartition spectrum -> Entropy should be 1.0
        Ek_flat = np.ones(10)
        S_flat = hs.spectral_entropy(Ek_flat)
        self.assertAlmostEqual(S_flat, 1.0)

    def test_helmholtz_decomposition(self):
        vgrid = hs.construct_velocity_grid(self.pos, self.vel, self.center, self.Rmax, Ng=16)
        helm = hs.helmholtz_decomposition(vgrid, Ng=16)
        
        self.assertIn("v_sol", helm)
        self.assertIn("v_comp", helm)
        self.assertIn("E_sol", helm)
        self.assertIn("E_comp", helm)
        
        tot_ratio = helm["solenoidal_ratio"] + helm["compressive_ratio"]
        self.assertAlmostEqual(tot_ratio, 1.0, places=5)

    def test_powerlaw_fitting(self):
        k = np.arange(1, 10)
        # Perfect power law E(k) = 5 * k^(-2)
        Ek = 5.0 * (k ** -2.0)
        fit = hs.find_best_linear_region_loglog(k, Ek, min_points=4)
        
        self.assertAlmostEqual(fit["alpha"], -2.0, places=4)
        self.assertAlmostEqual(fit["R2"], 1.0, places=4)

    def test_thermo_entropy(self):
        u_code = np.array([100.0, 200.0])
        xe = np.array([1.16, 1.16])
        rho = np.array([1e-5, 1e-4])
        
        T = hs.gas_temperature(u_code, xe)
        self.assertTrue(np.all(T > 0))
        
        s = hs.thermodynamic_entropy_per_particle(T, rho)
        self.assertEqual(len(s), 2)

    def test_caching_roundtrip(self):
        k = np.array([1, 2, 3])
        Ek = np.array([10.0, 5.0, 2.5])
        spectra_data = [{"Snapshot": 99, "SubhaloID": 0, "k": k, "Ek": Ek}]
        
        df = hs.spectra_list_to_df(spectra_data)
        self.assertEqual(len(df), 3)
        
        restored = hs.spectra_df_to_list(df)
        self.assertEqual(len(restored), 1)
        np.testing.assert_array_equal(restored[0]["k"], k)
        np.testing.assert_array_equal(restored[0]["Ek"], Ek)

    def test_profiling(self):
        prof = hs.profile_cpu(self.pos, self.vel, self.center, self.Rmax, Ng=16)
        self.assertIsNotNone(prof)
        self.assertIn("GFLOPs", prof)
        
        report = hs.format_markdown_report(prof)
        self.assertIn("Performance Profile Report", report)

if __name__ == "__main__":
    unittest.main()
