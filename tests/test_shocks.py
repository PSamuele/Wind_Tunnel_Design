import numpy as np
import pytest

from windtunnel import isentropic as isen
from windtunnel import shocks as sh


def test_normal_shock_m2():
    ns = sh.normal_shock(2.0)
    assert ns.M2 == pytest.approx(0.57735, rel=1e-5)
    assert ns.p2_p1 == pytest.approx(4.5, rel=1e-12)
    assert ns.rho2_rho1 == pytest.approx(2.6667, rel=1e-4)
    assert ns.T2_T1 == pytest.approx(1.6875, rel=1e-4)
    assert ns.p02_p01 == pytest.approx(0.72087, rel=1e-4)


def test_p0_ratio_inverse():
    for M in [1.1, 2.0, 3.7]:
        assert sh.mach_from_p0_ratio(sh.normal_shock_p0_ratio(M)) == pytest.approx(M, rel=1e-9)


def test_oblique_shock():
    o = sh.oblique_shock(2.0, np.radians(10.0))
    assert np.degrees(o.beta) == pytest.approx(39.314, abs=2e-3)
    assert o.M2 == pytest.approx(1.6405, abs=1e-3)
    th_max, _ = sh.max_deflection(2.0)
    assert np.degrees(th_max) == pytest.approx(22.97, abs=0.01)
    assert not sh.oblique_shock(2.0, np.radians(25.0)).attached


def test_cone_shock_bounds():
    M, ang = 2.0, np.radians(10.0)
    c = sh.cone_shock(M, ang)
    w = sh.oblique_shock(M, ang)
    assert c.attached
    assert np.arcsin(1.0 / M) < c.beta < w.beta  # weaker than the wedge shock, stronger than a Mach wave
    assert 1.0 < c.pc_p1 < w.p2_p1
    assert not sh.cone_shock(2.0, np.radians(45.0)).attached


def test_shock_in_nozzle_roundtrip():
    Ae = 2.5
    for As in [1.2, 1.8, 2.3]:
        pb = sh.back_pressure_from_shock_area(Ae, As)
        sol = sh.shock_in_nozzle(Ae, pb)
        assert sol.regime == "normal shock in divergent"
        assert sol.As_At == pytest.approx(As, rel=1e-8)


def test_nozzle_regimes():
    Ae = 2.0
    p_sub, p_exit_shock, p_design = sh.nozzle_regime_limits(Ae)
    assert p_design == pytest.approx(isen.p_p0(isen.mach_from_area_ratio(Ae)), rel=1e-12)
    assert p_design < p_exit_shock < p_sub < 1.0
    assert sh.shock_in_nozzle(Ae, 0.5 * (p_design + p_exit_shock)).regime.startswith("over-expanded")
    assert sh.shock_in_nozzle(Ae, 0.5 * p_design).regime.startswith("under-expanded")
