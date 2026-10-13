import math

import numpy as np
import pytest

from windtunnel import isentropic as isen
from windtunnel import moc


@pytest.mark.parametrize("M", [1.5, 2.0, 3.0, 4.0])
def test_nozzle_exit_and_wall(M):
    n = moc.design_nozzle(M, 0.05, n_axis=60)
    assert abs(n.area_ratio_error) < 5e-3
    assert n.warnings == []
    th = n.theta_wall
    k = int(np.argmax(th))
    assert n.omega <= th[k] < 1.1 * n.omega  # cone angle, then a little more expansion before turning
    assert np.all(np.diff(th[k:]) <= 1e-6)  # monotone turning after inflection
    assert abs(th[-1]) < 1e-4
    assert np.max(np.abs(np.diff(th))) < math.radians(0.5)  # no kinks
    assert n.M_wall[-1] == pytest.approx(M, rel=1e-3)


def test_lattice_reproduces_source_flow():
    M = 2.0
    om, _ = moc.choose_kernel(M, 2.0)
    r_t = 0.05
    Rs = r_t / math.sqrt(2.0 * (1.0 - math.cos(om)))

    def src(x):
        return isen.mach_from_area_ratio(np.maximum((np.asarray(x) / Rs) ** 2, 1.0), True)

    n = moc.design_nozzle(M, r_t, n_axis=80, axis_override=src)
    ok = np.isfinite(n.lattice_x)
    X = n.lattice_x[ok] + n.x_shift
    R = n.lattice_r[ok]
    inside = R <= np.interp(X - n.x_shift, n.x_wall, n.r_wall)
    Me = np.array([moc.source_flow_mach(x, r, Rs) for x, r in zip(X[inside], R[inside])])
    assert np.max(np.abs(n.lattice_M[ok][inside] - Me)) < 1e-3
    assert np.max(np.abs(n.lattice_theta[ok][inside] - np.arctan2(R[inside], X[inside]))) < math.radians(0.1)


def test_grid_convergence():
    a = moc.design_nozzle(2.0, 0.05, n_axis=60)
    b = moc.design_nozzle(2.0, 0.05, n_axis=120)
    xs = np.linspace(0.0, 0.99 * min(a.length, b.length), 100)
    diff = np.abs(np.interp(xs, a.x_wall, a.r_wall) - np.interp(xs, b.x_wall, b.r_wall)).max()
    assert diff / b.r_exit < 1e-3
