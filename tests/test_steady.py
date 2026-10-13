import math

import numpy as np
import pytest

from windtunnel import isentropic as isen
from windtunnel import shocks as sh
from windtunnel.design import design_tunnel


def test_mass_conservation(design_m2):
    for sol in (design_m2.sol_start, design_m2.sol_operating):
        A = math.pi * sol.r ** 2
        md = sol.rho * sol.V * A
        assert np.ptp(md) / md.mean() < 1e-6


def test_ratios_and_sizing(design_m2):
    d = design_m2
    assert 1.0 < d.PR_run < d.PR_start
    assert d.A_second_throat >= d.A_second_throat_min
    assert d.A_test / d.A_throat == pytest.approx(isen.area_ratio(2.0), rel=1e-12)
    assert d.sol_operating.exit_p == pytest.approx(d.spec.design.p_ambient, rel=1e-6)
    assert d.sol_start.exit_p == pytest.approx(d.spec.design.p_ambient, rel=1e-4)
    # quasi-1D test-section Mach at the nozzle exit equals the design Mach
    x_ne = d.geometry.segment("nozzle").x1
    i = np.searchsorted(d.sol_operating.x, x_ne)
    assert d.sol_operating.M[i] == pytest.approx(2.0, rel=1e-6)


def test_frictionless_starting_ratio_matches_analytic(spec_m2):
    s = spec_m2.copy()
    s.settling.K_honeycomb = s.settling.K_screens = 0.0
    s.diffuser.K_override = 0.0
    d = design_tunnel(s, fast=True)
    solver = d.flow.s
    solver.friction = False
    x_s = d.geometry.segment("nozzle").x1 + 1e-6
    pr, sol = d.flow.pressure_ratio_for_shock(x_s)
    # analytic: normal shock at M = 2, isentropic subsonic flow to the exit area
    p0r = sh.normal_shock_p0_ratio(2.0)
    Ae_At = d.geometry.r_exit ** 2 / d.geometry.r_throat ** 2
    Me = isen.mach_from_area_ratio(Ae_At * p0r, supersonic=False)
    assert pr == pytest.approx(1.0 / (p0r * isen.p_p0(Me)), rel=1e-6)
    # minimum A_d2 without friction is A* p01/p02
    assert sol.p0[-1] / sol.p0_supply == pytest.approx(p0r, rel=1e-9)


def test_hysteresis_branches(design_m2):
    d = design_m2
    pr_mid = 0.5 * (d.PR_start + d.PR_run)
    up = d.flow.operating_point(pr_mid * d.spec.design.p_ambient, started=False)
    down = d.flow.operating_point(pr_mid * d.spec.design.p_ambient, started=True)
    assert up[0].startswith("unstarted")
    assert down[0] == "started"
    assert up[1] < d.geometry.segment("test section").x1 < down[1]


def test_kantrowitz_and_liquefaction(design_m2):
    d = design_m2
    M2 = sh.normal_shock_M2(2.0)
    assert d.max_model_area == pytest.approx(d.A_test * (1 - 1 / isen.area_ratio(M2)), rel=1e-12)
    assert max(d.liquefaction.values()) < 1.0
    assert d.T0_min_no_liquefaction < d.spec.design.T0
