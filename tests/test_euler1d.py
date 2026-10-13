import numpy as np
import pytest

from windtunnel import shocks as sh
from windtunnel.euler1d import QuasiOneDEuler, detect_shocks, startup_transient


def test_sod_shock_tube():
    N = 400
    xf = np.linspace(0.0, 1.0, N + 1)
    xc = 0.5 * (xf[1:] + xf[:-1])
    s = QuasiOneDEuler(xf, lambda x: np.ones_like(x), left_bc="transmissive", right_bc="transmissive")
    rho = np.where(xc < 0.5, 1.0, 0.125)
    p = np.where(xc < 0.5, 1.0, 0.1)
    Q, t, n, *_ = s.run(0.2, initial=(rho, np.zeros(N), p), cfl=0.8)
    r, u, p = s.prim(Q)
    m = (xc > 0.72) & (xc < 0.83)  # between contact and shock
    assert p[m].mean() == pytest.approx(0.30313, rel=2e-3)  # exact Riemann solution
    assert u[m].mean() == pytest.approx(0.92745, rel=2e-3)
    x_shock = xc[np.where(r > 0.19)[0].max()]
    assert x_shock == pytest.approx(0.5 + 1.75216 * 0.2, abs=3.0 / N)


def test_nozzle_normal_shock_position():
    N, At = 160, 0.01
    area = lambda x: At * (1.0 + 2.2 * (x - 1.5) ** 2)
    xf = np.linspace(0.0, 3.0, N + 1)
    xc = 0.5 * (xf[1:] + xf[:-1])
    p0, T0, pb = 1e5, 300.0, 0.6784e5
    s = QuasiOneDEuler(xf, area, left_bc="stagnation", right_bc="pressure", p_back=pb, T_back=T0)
    init = (np.full(N, pb / (287.05 * T0)), np.zeros(N), np.full(N, pb))
    Q, *_ = s.run(0.2, lambda t: p0, lambda t: T0, init, cfl=0.8)
    r, u, p = s.prim(Q)
    M = u / np.sqrt(1.4 * p / r)
    xs = detect_shocks(xc, M, p)
    exact = sh.shock_in_nozzle(area(3.0) / At, pb / p0)
    x_exact = 1.5 + np.sqrt((exact.As_At - 1.0) / 2.2)
    assert len(xs) == 1 and abs(xs[0] - x_exact) < 2 * 3.0 / N
    md = r * u * s.Ac
    away = np.abs(xc - x_exact) > 0.1
    from windtunnel.isentropic import choked_mass_flow
    assert np.abs(md[away] / choked_mass_flow(p0, T0, At) - 1.0).max() < 5e-3


def test_startup_transient_starts_and_unstarts(design_m2):
    ts = design_m2.spec.transient
    ts = type(ts)(**{**ts.__dict__, "n_cells": 120, "n_snapshots": 6, "valve_time": 0.02, "hold_time": 0.04,
                     "ramp_down_time": 0.02})
    tr = startup_transient(design_m2, ts)
    assert "t_start" in tr.events and "t_unstart" in tr.events
    assert tr.events["PR_at_start"] >= design_m2.PR_start * 0.98
    assert tr.events["PR_at_unstart"] <= design_m2.PR_run
    assert len(tr.snap) == 6
