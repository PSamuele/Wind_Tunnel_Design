import math

import numpy as np
import pytest

from windtunnel import fanno as fn
from windtunnel import shocks as sh
from windtunnel.design import design_tunnel, model_drag
from windtunnel.euler1d import QuasiOneDEuler
from windtunnel.second_throat import REGIMES, classify, study


@pytest.fixture(scope="module")
def model_spec(spec_m2):
    s = spec_m2.copy()
    s.model.frontal_area = 0.0015
    s.second_throat.n_points = 6
    return s


@pytest.fixture(scope="module")
def design_model(model_spec):
    return design_tunnel(model_spec.copy())


@pytest.fixture(scope="module")
def st(design_model):
    return study(design_model)


def test_classify():
    assert [classify(a, 1.0, 1.2, 1.4) for a in (0.9, 1.1, 1.3, 1.5)] == ["R0", "R1", "R2", "R3"]
    assert set(REGIMES) == {"R0", "R1", "R2", "R3"}


def test_critical_area_order(st):
    # two sonic throats need A2/A1 = p01/p02 > 1: equal areas cannot both be sonic
    assert st.A_t < st.A_sub < st.A_run < st.A_start <= st.A_design
    assert st.equal_area_regime == "R0"
    assert st.design_regime == "R3"


def test_model_drag_is_a_fanno_step(design_model, model_spec):
    d = design_model
    A_ts = math.pi * (0.5 * model_spec.design.test_section_diameter) ** 2
    assert d.drag.fLD == pytest.approx(d.drag.cd * model_spec.model.frontal_area / A_ts, rel=1e-12)
    sol = d.sol_operating
    ev = [e for e in sol.events if e[0].startswith("model drag")][0]
    # Mach just upstream of the model: two records at the model station (before/after)
    ts = d.geometry.segment("test section")
    x_m = ts.x0 + model_spec.model.position * ts.length
    i = int(np.where(np.abs(sol.x - x_m) < 1e-12)[0][0])
    M1, M2 = sol.M[i], sol.M[i + 1]
    assert fn.fanno_fL_D(M1) - fn.fanno_fL_D(M2) == pytest.approx(d.drag.fLD, rel=1e-6)
    assert ev[2] / ev[1] == pytest.approx(fn.fanno_p0_p0star(M2) / fn.fanno_p0_p0star(M1), rel=1e-9)


def test_model_raises_ratios(design_m2, design_model):
    assert design_model.PR_start > design_m2.PR_start
    assert design_model.PR_run > design_m2.PR_run
    assert design_model.A_second_throat_min > design_m2.A_second_throat_min


def test_automatic_cd_cone(model_spec):
    d = model_drag(model_spec)
    M = model_spec.design.mach
    cone = sh.cone_shock(M, math.radians(model_spec.model.half_angle_deg))
    expected = (cone.pc_p1 - 1.0) / (0.7 * M * M) + 1.0 / M ** 2
    assert d.cd == pytest.approx(expected, rel=1e-9)


def test_variable_geometry_gain(st):
    v = st.variable
    assert v["applicable"]
    assert v["A_closed/A_t [-]"] < v["A_open/A_t [-]"]
    assert v["PR_run closed [-]"] < v["PR_run fixed [-]"]
    assert v["run time variable [s]"] > v["run time fixed [s]"]


def test_sensitivity_monotone_in_R3(st):
    rows = [r for r in st.sensitivity if r["regime"] == "R3" and np.isfinite(r["PR_run [-]"])]
    pr = [r["PR_run [-]"] for r in rows]
    assert len(pr) >= 2 and np.all(np.diff(pr) > 0)  # wider throat -> stronger running shock


def test_shock_pair():
    weak = sh.shock_pair(2.0, math.radians(0.5))
    assert weak.regime == "regular reflection" and weak.p0_ratio > 0.9999
    th = sh.max_regular_pair_angle(2.0)
    assert sh.shock_pair(2.0, 0.98 * th).regime == "regular reflection"
    assert sh.shock_pair(2.0, 1.02 * th).regime != "regular reflection"
    # a pair of oblique shocks loses less than one normal shock at the same Mach
    assert sh.shock_pair(2.0, 0.9 * th).p0_ratio > sh.normal_shock(2.0).p02_p01


def test_moving_walls_isentropic_compression():
    """Uniform gas at rest, area shrinking uniformly in time: the -p dA/dt
    term must give p ~ rho^gamma with rho ~ 1/A."""
    N = 20
    xf = np.linspace(0.0, 1.0, N + 1)
    eps = 0.1
    area_xt = lambda x, t: np.ones_like(x) * (1.0 - eps * t)
    s = QuasiOneDEuler(xf, lambda x: np.ones_like(x), left_bc="transmissive", right_bc="transmissive",
                       area_xt=area_xt)
    Q, t, *_ = s.run(1.0, initial=(np.ones(N), np.zeros(N), np.ones(N)), cfl=0.5)
    s.set_time(t)
    r, u, p = s.prim(Q)
    assert np.allclose(r, 1.0 / (1.0 - eps), rtol=1e-6)
    assert np.allclose(p, (1.0 / (1.0 - eps)) ** 1.4, rtol=1e-3)
    assert np.allclose(u, 0.0, atol=1e-9)
