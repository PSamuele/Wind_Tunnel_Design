import numpy as np
import pytest

from windtunnel.blowdown import simulate_blowdown
from windtunnel.design import design_tunnel


def test_mass_balance_and_run(design_m2):
    b = simulate_blowdown(design_m2, track_shock=False)
    assert b.run_time > 0.0
    assert np.isfinite(b.t_start)
    used = np.trapezoid(b.mdot, b.t)
    assert b.mass_used[-1] == pytest.approx(used, rel=2e-2)
    assert np.all(np.diff(b.p_tank) <= 1e-9)


def test_isothermal_tank(spec_m2):
    s = spec_m2.copy()
    s.tank.polytropic_n = 1.0
    d = design_tunnel(s, fast=True)
    b = simulate_blowdown(d, track_shock=False)
    assert np.allclose(b.T0, s.design.T0)
    b_ad = simulate_blowdown(design_tunnel(spec_m2.copy(), fast=True), track_shock=False)
    assert b.run_time > b_ad.run_time  # isothermal tank stores more usable mass
