import numpy as np
import pytest

from windtunnel import isentropic as isen


@pytest.mark.parametrize("M, p, T, rho, ar", [
    # NACA Report 1135, gamma = 1.4
    (0.5, 0.84302, 0.95238, 0.88517, 1.3398),
    (2.0, 0.12780, 0.55556, 0.23005, 1.6875),
    (3.0, 0.027224, 0.35714, 0.076226, 4.2346),
])
def test_naca1135_values(M, p, T, rho, ar):
    assert isen.p_p0(M) == pytest.approx(p, rel=2e-4)
    assert isen.T_T0(M) == pytest.approx(T, rel=2e-4)
    assert isen.rho_rho0(M) == pytest.approx(rho, rel=2e-4)
    assert isen.area_ratio(M) == pytest.approx(ar, rel=2e-4)


def test_area_ratio_inversion_both_branches():
    for M in [0.1, 0.5, 0.9, 1.2, 2.0, 3.5, 5.0]:
        ar = isen.area_ratio(M)
        assert isen.mach_from_area_ratio(ar, supersonic=M > 1) == pytest.approx(M, rel=1e-10)


def test_prandtl_meyer():
    assert np.degrees(isen.prandtl_meyer(2.0)) == pytest.approx(26.380, abs=1e-3)
    nu = np.radians([0.5, 10.0, 26.38, 65.78])
    M = isen.mach_from_prandtl_meyer(nu)
    assert np.allclose(isen.prandtl_meyer(M), nu, atol=1e-12)


def test_choked_mass_flow():
    # m_dot = 0.040418 p0 A / sqrt(T0) for air (gamma 1.4, R 287.05)
    md = isen.choked_mass_flow(1e5, 300.0, 0.01)
    assert md == pytest.approx(0.040418 * 1e5 * 0.01 / np.sqrt(300.0), rel=1e-3)
