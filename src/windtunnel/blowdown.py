"""Quasi-steady blowdown run: tank depletion with a pressure regulator and
start/unstart hysteresis.

Model:
* tank of volume V with polytropic expansion p V^n = const per unit mass
  (n = gamma: adiabatic, n = 1: isothermal, e.g. with a heat-storage matrix);
* regulator holding the supply total pressure at the set point while the
  tank pressure exceeds p0_set (1 + regulator_margin); below that, p0 follows
  the tank, p0 = p_tank / (1 + regulator_margin);
* throttling is isenthalpic: T0 = T_tank;
* choked throat: m_dot = p0 A_t Gamma / sqrt(R T0);
* hysteresis: the tunnel starts when p0/p_a >= PR_start and stays started
  until p0/p_a < PR_run.
The integration uses classic RK4 on the tank density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import isentropic as isen
from . import losses as ls


@dataclass
class BlowdownResult:
    t: np.ndarray
    p_tank: np.ndarray
    T_tank: np.ndarray
    p0: np.ndarray
    T0: np.ndarray
    mdot: np.ndarray
    started: np.ndarray
    Re_unit: np.ndarray
    T_test: np.ndarray
    liquefaction_ratio: np.ndarray
    mass_used: np.ndarray
    shock_t: np.ndarray = field(default_factory=lambda: np.array([]))
    shock_x: np.ndarray = field(default_factory=lambda: np.array([]))
    shock_regime: list = field(default_factory=list)
    t_start: float = float("nan")
    t_unstart: float = float("nan")
    run_time: float = 0.0
    stop_reason: str = ""

    def as_columns(self):
        return {"time [s]": self.t, "tank pressure [Pa]": self.p_tank, "tank temperature [K]": self.T_tank,
                "supply p0 [Pa]": self.p0, "supply T0 [K]": self.T0, "mass flow [kg/s]": self.mdot,
                "started [1/0]": self.started.astype(int), "unit Reynolds [1/m]": self.Re_unit,
                "test static T [K]": self.T_test, "liquefaction saturation ratio [-]": self.liquefaction_ratio,
                "air used [kg]": self.mass_used}


def simulate_blowdown(design, n_shock_samples=30, track_shock=True) -> BlowdownResult:
    spec = design.spec
    gas = design.gas
    g, R = gas.gamma, gas.R
    tank, run = spec.tank, spec.run
    p_a = spec.design.p_ambient
    M = spec.design.mach
    V = tank.volume
    n = tank.polytropic_n
    p_i, T_i = tank.pressure, spec.design.T0
    rho_i = p_i / (R * T_i)
    A_t = design.A_throat
    gam = float(isen.mass_flow_function(1.0, g))
    choke_pr = design.flow.pressure_ratio_for_shock(design.flow.x_throat + 1e-6)[0]
    margin = tank.regulator_margin

    def tank_state(rho):
        p = p_i * (rho / rho_i) ** n
        T = T_i * (rho / rho_i) ** (n - 1.0)
        return p, T

    def supply(rho, started_flag, was_started):
        p_t, T_t = tank_state(rho)
        if spec.run.p0_schedule == "reduce_after_start" and was_started:
            p_set = design.p0_run_set
        else:
            p_set = design.p0_start_set
        p0 = min(p_set, p_t / (1.0 + margin))
        return p0, T_t, p_t

    def mdot(p0, T0):
        if p0 / p_a <= 1.0:
            return 0.0
        if p0 / p_a < choke_pr:
            # unchoked: scale the choked value by the isentropic throat flow
            # function at the throat Mach giving roughly p_a downstream
            Mt = float(isen.mach_from_p_p0(min(p_a / p0, 0.999999), g))
            return p0 * A_t / math.sqrt(R * T0) * float(isen.mass_flow_function(min(Mt, 1.0), g))
        return p0 * A_t * gam / math.sqrt(R * T0)

    dt = run.dt
    t, rho = 0.0, rho_i
    started = False
    ever_started = False
    rows = []
    t_start = t_unstart = float("nan")
    stop = "t_max reached"
    while t <= run.t_max:
        p0, T0, p_t = supply(rho, started, ever_started)
        pr = p0 / p_a
        if not started and not ever_started and pr >= design.PR_start:
            started, ever_started, t_start = True, True, t
        elif started and pr < design.PR_run:
            started, t_unstart = False, t
        md = mdot(p0, T0)
        p_s, T_s, _, _ = isen.static_state(M, p0, T0, g, R)
        Re = float(gas.unit_reynolds(p_s, T_s, M)) if started else 0.0
        liq = max(ls.liquefaction_margin(p_s, T_s).values()) if started else 0.0
        rows.append((t, p_t, tank_state(rho)[1], p0, T0, md, started, Re, T_s if started else np.nan, liq,
                     (rho_i - rho) * V))
        if ever_started and not started:
            stop = "unstart (p0/p_a below the running ratio)"
            break
        if pr <= 1.01 * min(choke_pr, design.PR_run):
            stop = "tank exhausted before start" if not ever_started else "tank exhausted"
            break

        def f(r):
            p0_, T0_, _ = supply(r, started, ever_started)
            return -mdot(p0_, T0_) / V

        k1 = f(rho)
        k2 = f(rho + 0.5 * dt * k1)
        k3 = f(rho + 0.5 * dt * k2)
        k4 = f(rho + dt * k3)
        rho = max(rho + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 1e-9)
        t += dt

    cols = list(zip(*rows))
    res = BlowdownResult(*(np.array(c) for c in cols[:6]), np.array(cols[6], bool), *(np.array(c) for c in cols[7:]))
    res.t_start, res.t_unstart, res.stop_reason = t_start, t_unstart, stop
    if np.isfinite(t_start):
        t_end = t_unstart if np.isfinite(t_unstart) else res.t[-1]
        res.run_time = float(t_end - t_start)
    if track_shock and np.any(res.started):
        idx = np.unique(np.linspace(0, res.t.size - 1, n_shock_samples).astype(int))
        st, sx, sr = [], [], []
        for k in idx:
            regime, x_s, _ = design.flow.operating_point(res.p0[k], bool(res.started[k]))
            st.append(res.t[k])
            sx.append(np.nan if x_s is None else x_s)
            sr.append(regime)
        res.shock_t, res.shock_x, res.shock_regime = np.array(st), np.array(sx), sr
    return res
