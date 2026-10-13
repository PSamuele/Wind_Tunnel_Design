"""Second throat: sonic compatibility of two throats in series, operating
regimes, variable geometry, sensitivity to the throat area and ramp
(oblique-shock) compression in the convergent.

Two sonic throats in series
---------------------------
Through a sonic throat m_dot = Gamma p0 A* / sqrt(R T0). Between the nozzle
throat (1) and the second throat (2) the flow is adiabatic (T0 constant)
but loses total pressure (friction, model drag, shocks), so both throats can
be sonic only if

    A*_2 / A*_1 = p0_1 / p0_2 > 1.

With equal areas the second throat chokes first and the nozzle throat cannot
be sonic. Three critical areas follow (all include friction and model drag):

    A_sub    smallest A_d2 that lets the nozzle throat be sonic (subsonic
             flow downstream, no shock): below it the second throat controls
             the mass flow and the whole tunnel is subsonic;
    A_run    smallest A_d2 through which the started (supersonic) flow passes
             the whole second throat (friction inside the throat included):
             the running limit;
    A_start  smallest A_d2 that swallows the starting normal shock.

Regimes for a given A_d2:
    R0  A_d2 < A_sub            second throat controls, tunnel subsonic
    R1  A_sub <= A_d2 < A_run   nozzle throat sonic, normal shock locked in the
                                nozzle or test section: never started
    R2  A_run <= A_d2 < A_start can run started but cannot start with fixed
                                walls: needs a variable second throat
    R3  A_d2 >= A_start         starts and runs with fixed walls
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import shocks as sh
from .blowdown import simulate_blowdown
from .design import TunnelDesign, _build, _solver
from .geometry import with_second_throat
from .steady import TunnelFlow

REGIMES = {
    "R0": "second throat controls the mass flow: nozzle throat not sonic, tunnel subsonic",
    "R1": "nozzle throat sonic, normal shock locked in the nozzle/test section: never started",
    "R2": "runs started but cannot start with fixed walls: variable second throat needed",
    "R3": "starts and runs with fixed walls",
}


@dataclass
class SecondThroatStudy:
    A_t: float
    A_sub: float
    A_run: float
    A_start: float
    A_design: float
    design_regime: str
    equal_area_regime: str
    p0_ratio_running: float  # p0_1 / p0_2 between the throats, started flow
    p0_ratio_starting: float  # same with the starting shock
    explanation: list
    solutions: list
    variable: dict | None
    sensitivity: list
    ramp: list
    flows: dict = field(default_factory=dict)  # name -> (geometry, TunnelFlow), for plots and exports

    def critical_table(self):
        A_t = self.A_t
        return [
            {"quantity": "nozzle throat A_t", "A/A_t [-]": 1.0, "meaning": "first sonic throat"},
            {"quantity": "A_sub", "A/A_t [-]": self.A_sub / A_t,
             "meaning": "below: second throat controls, nozzle throat not sonic (R0)"},
            {"quantity": "A_run", "A/A_t [-]": self.A_run / A_t,
             "meaning": "below: started flow impossible, shock locked upstream (R1); running limit"},
            {"quantity": "A_start", "A/A_t [-]": self.A_start / A_t,
             "meaning": "below: fixed walls cannot start (R2); above: starts (R3)"},
            {"quantity": "design A_d2", "A/A_t [-]": self.A_design / A_t, "meaning": REGIMES[self.design_regime]},
        ]


def classify(A, A_sub, A_run, A_start):
    if A < A_sub:
        return "R0"
    if A < A_run:
        return "R1"
    if A < A_start:
        return "R2"
    return "R3"


def _flow_for(design: TunnelDesign, A_d2, keep_layout=False, convergent_model=None):
    spec = design.spec
    g = design.geometry
    r = math.sqrt(A_d2 / math.pi)
    geom = with_second_throat(g, r) if keep_layout else _build(spec, design.nozzle, g.r_throat, g.r_test, r)
    return geom, TunnelFlow(_solver(spec, geom, design.gas, drag=design.drag, convergent_model=convergent_model),
                            spec.design.T0, spec.design.p_ambient)


def running_critical_area(design: TunnelDesign, n_iter=30):
    """Running limit: smallest second-throat area through which the started
    (supersonic) flow passes the whole second throat. It includes friction
    inside the throat (a near-sonic flow chokes in a few diameters) and the
    convergent model. Found by bisection at the running set point."""
    spec = design.spec
    T0 = spec.design.T0
    p0 = design.p0_run_set

    def passes(A):
        geom, flow = _flow_for(design, A, keep_layout=True)
        sol = flow.s.march(p0, T0)
        return sol.ok or sol.x[-1] > geom.segment("second throat").x1 + 1e-9

    hi = min(design.A_test * 0.999, max(design.A_second_throat, design.A_second_throat_min) * 1.2)
    if not passes(hi):
        raise RuntimeError("started flow cannot pass the second throat even when it is wide open")
    # lower bound: local A* at the entrance (sonic, no friction in the throat)
    geom, flow = _flow_for(design, hi, keep_layout=True)
    sol = flow.s.march(p0, T0)
    x2 = geom.segment("second throat").x0
    i = int(np.searchsorted(sol.x, x2, side="right") - 1)
    lo = design.A_throat * sol.p0_supply / sol.p0[i] * (1.0 - 1e-9)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if passes(mid):
            hi = mid
        else:
            lo = mid
    return hi


def subsonic_critical_area(design: TunnelDesign):
    """A*_loc at the second-throat entrance with a sonic nozzle throat and
    subsonic flow downstream (zero-strength shock at the throat)."""
    f = design.flow
    return f.critical_area_after_shock(f.x_throat + 1e-7, design.p0_run_set)


def locked_shock(flow: TunnelFlow, A_d2, p0):
    """R1: position and Mach of the normal shock fixed by a choked second
    throat (A*_loc behind the shock = A_d2)."""
    def resid(x):
        return flow.critical_area_after_shock(x, p0) - A_d2

    a, b = flow.x_throat + 1e-6, flow.x_ts_end - 1e-6
    try:
        if resid(a) * resid(b) > 0:
            return None, None
        x = brentq(resid, a, b, xtol=1e-6)
    except ValueError:
        return None, None
    sol = flow.s.march(p0, flow.T0, shocks=[x])
    return x, (sol.shocks[0][1] if sol.shocks else None)


def _run_time(design: TunnelDesign, flow: TunnelFlow, PR_start, PR_run, A_throat=None):
    spec = design.spec
    p_a = spec.design.p_ambient
    d = copy.copy(design)
    d.flow = flow
    d.PR_start, d.PR_run = PR_start, PR_run
    d.p0_start_set = PR_start * p_a * (1.0 + spec.run.start_margin)
    d.p0_run_set = (PR_run * p_a * (1.0 + spec.run.run_margin) if spec.run.p0_schedule == "reduce_after_start"
                    else d.p0_start_set)
    b = simulate_blowdown(d, track_shock=False)
    return b.run_time, float(b.mass_used[-1]), d.p0_run_set


def _entrance_mach(sol, geom):
    x = geom.segment("second-throat convergent").x0
    i = int(np.searchsorted(sol.x, x, side="right") - 1)
    return float(sol.M[i])


def _throat_mach(sol, geom):
    x = geom.segment("second throat").x0
    i = int(np.searchsorted(sol.x, x, side="right") - 1)
    return float(sol.M[i])


def study(design: TunnelDesign, verbose=False) -> SecondThroatStudy:
    spec = design.spec
    st = spec.second_throat
    A_t = design.A_throat
    A_start = design.A_second_throat_min
    A_run = running_critical_area(design)
    A_sub = subsonic_critical_area(design)
    A_des = design.A_second_throat
    reg = classify(A_des, A_sub, A_run, A_start)
    reg_eq = classify(A_t, A_sub, A_run, A_start)

    explanation = [
        "Mass flow through a sonic throat: m = Gamma p0 A* / sqrt(R T0). T0 is constant between the throats, "
        "p0 falls, so two sonic throats need A*_2/A*_1 = p0_1/p0_2 > 1.",
        f"Started flow: p0_1/p0_2 = {A_run / A_t:.4f} (friction, model drag, convergent) -> the second throat must "
        f"be at least {A_run / A_t:.4f} A_t to stay sonic.",
        f"Starting: with the normal shock in the test section p0_1/p0_2 = {A_start / A_t:.4f} -> fixed walls need "
        f"A_d2 >= {A_start / A_t:.4f} A_t.",
        f"Equal areas (A_d2 = A_t): regime {reg_eq}, {REGIMES[reg_eq]}.",
        f"Without a sonic nozzle throat the flow is subsonic everywhere when A_d2 < {A_sub / A_t:.4f} A_t "
        "(subsonic losses only).",
    ]

    # ---------------- variable geometry
    variable = None
    flows = {}
    g_des = design.geometry
    flows["design"] = (g_des, design.flow)
    if st.variable_geometry:
        A_closed = A_run * (1.0 + st.run_margin)
        if A_closed >= A_des:
            variable = {"applicable": False,
                        "reason": "closed area not smaller than the fixed design: no gain from variable walls"}
        else:
            g_cl, f_cl = _flow_for(design, A_closed, keep_layout=True)
            pr_run_cl, x_cl, sol_cl = f_cl.running()
            if np.isfinite(pr_run_cl):
                t_fix, m_fix, p0_fix = _run_time(design, design.flow, design.PR_start, design.PR_run)
                t_vg, m_vg, p0_vg = _run_time(design, f_cl, design.PR_start, pr_run_cl)
                variable = {
                    "applicable": True,
                    "A_open/A_t [-]": A_des / A_t,
                    "A_closed/A_t [-]": A_closed / A_t,
                    "PR_start (open) [-]": design.PR_start,
                    "PR_run fixed [-]": design.PR_run,
                    "PR_run closed [-]": pr_run_cl,
                    "running p0 fixed [bar]": p0_fix / 1e5,
                    "running p0 closed [bar]": p0_vg / 1e5,
                    "mass flow running, fixed [kg/s]": design.mdot_per_p0 * p0_fix,
                    "mass flow running, closed [kg/s]": design.mdot_per_p0 * p0_vg,
                    "run time fixed [s]": t_fix,
                    "run time variable [s]": t_vg,
                    "run time gain [%]": 100 * (t_vg / t_fix - 1.0) if t_fix > 0 else float("nan"),
                    "Mach at closed throat entrance [-]": _throat_mach(sol_cl, g_cl),
                    "running shock Mach, closed [-]": sol_cl.shocks[0][1] if sol_cl.shocks else float("nan"),
                    "running shock Mach, fixed [-]": (design.sol_run_min.shocks[0][1]
                                                      if design.sol_run_min.shocks else float("nan")),
                    "closed-throat geometry": g_cl,
                    "solution closed": sol_cl,
                }
                flows["closed"] = (g_cl, f_cl)
            else:
                variable = {"applicable": False, "reason": "no running solution with the closed throat"}

    # ---------------- sensitivity
    sens = []
    A_lo = st.area_min_factor * min(A_run, A_t)
    A_hi = min(st.area_max_factor * A_start, 0.999 * design.A_test)
    edges = [A_sub * 1.0005, A_run * 1.0005, A_start * 0.9995, A_start * 1.0005]
    areas = np.unique(np.concatenate([np.linspace(A_lo, A_hi, st.n_points), [A_t, A_des],
                                      [a for a in edges if A_lo < a < A_hi]]))
    for A in areas:
        r = classify(A, A_sub, A_run, A_start)
        row = {"A_d2/A_t [-]": A / A_t, "regime": r, "PR_start [-]": float("nan"), "PR_run [-]": float("nan"),
               "throat Mach running [-]": float("nan"), "running shock Mach [-]": float("nan"),
               "locked shock x [m]": float("nan"), "locked shock Mach [-]": float("nan"),
               "run time [s]": float("nan"), "start": ""}
        try:
            geom, flow = _flow_for(design, A, keep_layout=True)
            if r == "R1":
                x, Ms = locked_shock(flow, A, design.p0_start_set)
                if x is not None:
                    row["locked shock x [m]"], row["locked shock Mach [-]"] = x, Ms
            if r in ("R2", "R3"):
                pr_run, x_run, sol = flow.running(n=10)
                row["PR_run [-]"] = pr_run
                if sol is not None and np.isfinite(pr_run):
                    row["throat Mach running [-]"] = _throat_mach(sol, geom)
                    row["running shock Mach [-]"] = sol.shocks[0][1] if sol.shocks else float("nan")
                if r == "R3":
                    pr_st, _, _ = flow.starting(n=8)
                    row["PR_start [-]"] = pr_st
                    row["start"] = "fixed walls"
                else:
                    pr_st = design.PR_start
                    row["PR_start [-]"] = pr_st
                    row["start"] = "open to design area first"
                if np.isfinite(pr_run) and np.isfinite(pr_st):
                    row["run time [s]"] = _run_time(design, flow, pr_st, pr_run)[0]
        except (RuntimeError, ValueError):
            pass
        sens.append(row)

    # ---------------- ramp (oblique shocks) vs isentropic convergent
    ramp = []
    configs = [("fixed design", A_des)]
    if variable and variable.get("applicable"):
        configs.append(("variable, closed", variable["A_closed/A_t [-]"] * A_t))
    for name, A in configs:
        M_in = float("nan")
        for model in ("isentropic", "oblique_shocks"):
            geom, flow = _flow_for(design, A, keep_layout=True, convergent_model=model)
            pr_run, _, sol = flow.running(n=10)
            row = {"configuration": name, "convergent model": model, "A_d2/A_t [-]": A / A_t,
                   "convergent half-angle [deg]": math.degrees(flow.s.convergent_half_angle),
                   "PR_run [-]": pr_run, "Mach at convergent entrance [-]": float("nan"),
                   "shock pair": "", "pair p0 ratio [-]": float("nan"),
                   "max angle for regular reflection [deg]": float("nan"),
                   "running shock Mach [-]": float("nan"), "result": ""}
            if model == "isentropic" and sol is not None and np.isfinite(pr_run):
                M_in = _entrance_mach(sol, geom)
            if model == "oblique_shocks" and (sol is None or not np.isfinite(pr_run)):
                # the shock pair ends subsonic: try the state with the normal
                # shock at the convergent entrance
                pr_run, sol = flow.pressure_ratio_without_shock()
                row["PR_run [-]"] = pr_run
            if sol is not None and np.isfinite(pr_run):
                row["running shock Mach [-]"] = sol.shocks[-1][1] if sol.shocks else float("nan")
                row["result"] = "runs"
            else:
                row["result"] = "unstart: second throat chokes behind the convergent shocks"
            row["Mach at convergent entrance [-]"] = M_in
            if model == "oblique_shocks" and np.isfinite(M_in):
                pair = flow.s.shock_pair(M_in)
                row["shock pair"] = pair.regime
                row["pair p0 ratio [-]"] = pair.p0_ratio
                row["max angle for regular reflection [deg]"] = math.degrees(sh.max_regular_pair_angle(M_in))
            ramp.append(row)

    # ---------------- solutions
    sol_txt = []
    if reg == "R3":
        sol_txt.append(f"Fixed second throat A_d2 = {A_des / A_t:.4f} A_t (>= A_start = {A_start / A_t:.4f} A_t): "
                       "the tunnel starts and runs.")
    else:
        sol_txt.append(f"Design second throat in regime {reg} ({REGIMES[reg]}): enlarge it to at least "
                       f"{A_start / A_t:.4f} A_t or use variable walls.")
    sol_txt.append(f"Never size the second throat equal to the nozzle throat or below A_run = {A_run / A_t:.4f} A_t: "
                   "the second throat would choke and the test section cannot be supersonic.")
    if variable and variable.get("applicable"):
        sol_txt.append(
            f"Variable second throat: start at {variable['A_open/A_t [-]']:.4f} A_t, then close to "
            f"{variable['A_closed/A_t [-]']:.4f} A_t ({100 * st.run_margin:.0f} % above A_run): PR_run "
            f"{variable['PR_run fixed [-]']:.4f} -> {variable['PR_run closed [-]']:.4f}, run time "
            f"{variable['run time fixed [s]']:.2f} s -> {variable['run time variable [s]']:.2f} s. "
            "Close slowly and keep a margin above A_run: closing below it unstarts the tunnel.")
    if spec.model.frontal_area > 0:
        sol_txt.append(f"Model drag accounts for p0_1/p0_2 growth: equivalent friction length "
                       f"Cd A_m/A_ts = {design.drag.fLD:.4f}; a smaller or more slender model lowers A_start "
                       "and PR_run.")
    sol_txt.append("Shorter test section and second throat, and smooth walls, reduce the friction part of "
                   "p0_1/p0_2.")
    if len(ramp) >= 2 and all(np.isfinite(r["PR_run [-]"]) for r in ramp[:2]):
        d_pr = ramp[1]["PR_run [-]"] - ramp[0]["PR_run [-]"]
        sol_txt.append(f"Ramp convergent (oblique shock pair) on the fixed design changes PR_run by {d_pr:+.4f} "
                       "compared with the isentropic convergent.")
    for r in ramp:
        if r["convergent model"] == "oblique_shocks" and r["shock pair"] not in ("", "regular reflection"):
            sol_txt.append(
                f"{r['configuration']}: a {r['convergent half-angle [deg]']:.1f} deg convergent at M = "
                f"{r['Mach at convergent entrance [-]']:.3f} gives a {r['shock pair']} ({r['result']}). Keep the "
                f"convergent half-angle below {r['max angle for regular reflection [deg]']:.1f} deg, i.e. make the "
                "movable walls longer.")

    return SecondThroatStudy(A_t, A_sub, A_run, A_start, A_des, reg, reg_eq, A_run / A_t, A_start / A_t,
                             explanation, sol_txt, variable, sens, ramp, flows)
