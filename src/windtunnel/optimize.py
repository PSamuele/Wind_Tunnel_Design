"""Diffuser optimisation and Mach-number trade study.

Optimisation problem (the nozzle is fixed by the test-section Mach number
and diameter, so only the downstream components are free):

    minimise    PR_run = p0/p_a needed to keep the tunnel started
    variables   second-throat margin, second-throat length / diameter,
                subsonic diffuser half-angle, subsonic diffuser area ratio
    subject to  total length <= max_total_length (if set)
                PR_start p_a (1 + start_margin)(1 + regulator_margin) <= p_tank
                bounds on every variable

Lower PR_run means a lower supply pressure while running, hence lower mass
flow per unit time and a longer run for the same tank. The quasi-1D model
contains the trade-offs that make the problem non-trivial:
* a smaller second throat lowers the running shock Mach (less shock loss)
  but cannot go below the starting limit;
* a longer second throat adds friction loss but slows the supersonic flow
  before the shock;
* the diffuser half-angle balances wall friction (small angles) against
  expansion/separation loss (large angles);
* the diffuser area ratio balances diffuser loss against the exit
  dynamic-pressure loss, within the length limit.

Method: Sobol sampling of the box (global exploration), then SLSQP from the
best feasible sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from .blowdown import simulate_blowdown
from .design import _build, _solver, design_tunnel
from .steady import TunnelFlow

VARIABLES = [
    # (spec attribute in [diffuser], label)
    ("second_throat_margin", "second-throat margin [-]"),
    ("second_throat_length_factor", "second-throat length / diameter [-]"),
    ("subsonic_half_angle_deg", "subsonic diffuser half-angle [deg]"),
    ("area_ratio", "subsonic diffuser area ratio [-]"),
]


@dataclass
class OptimisationResult:
    names: list
    bounds: list
    baseline_x: np.ndarray
    baseline: dict
    x_opt: np.ndarray
    optimum: dict
    history: list = field(default_factory=list)  # dicts with x and metrics
    message: str = ""

    def table(self):
        rows = []
        for i, (name, label) in enumerate(VARIABLES):
            rows.append({"variable": label, "baseline": float(self.baseline_x[i]), "optimum": float(self.x_opt[i]),
                         "lower": self.bounds[i][0], "upper": self.bounds[i][1]})
        return rows


def _bounds(spec):
    o = spec.optimisation
    return [
        (o.min_second_throat_margin, o.max_second_throat_margin),
        (0.5, 4.0),
        (1.5, o.max_subsonic_half_angle_deg),
        (1.5, 5.0),
    ]


class _Evaluator:
    def __init__(self, design):
        self.d = design
        self.spec = design.spec
        self.cache = {}
        self.history = []

    def __call__(self, x):
        key = tuple(np.round(np.asarray(x, dtype=float), 10))
        if key in self.cache:
            return self.cache[key]
        spec = self.spec.copy()
        for (name, _), v in zip(VARIABLES, x):
            setattr(spec.diffuser, name, float(v))
        d = self.d
        p_a = spec.design.p_ambient
        try:
            # second throat from the (design-independent) starting limit
            A_d2 = d.A_second_throat_min * (1.0 + spec.diffuser.second_throat_margin)
            A_ts = d.A_test
            if A_d2 >= A_ts:
                raise ValueError("second throat larger than test section")
            r_d2 = math.sqrt(A_d2 / math.pi)
            geom = _build(spec, d.nozzle, d.geometry.r_throat, d.geometry.r_test, r_d2)
            flow = TunnelFlow(_solver(spec, geom, d.gas, drag=d.drag), spec.design.T0, p_a)
            pr_run, x_run, _ = flow.running(n=8)
            pr_start, _ = flow.pressure_ratio_for_shock(flow.x_nozzle_exit + 1e-6)
            length = geom.length
            ok = np.isfinite(pr_run) and np.isfinite(pr_start)
        except (ValueError, RuntimeError):
            pr_run = pr_start = length = float("nan")
            ok = False
        p_need = pr_start * p_a * (1.0 + spec.run.start_margin) * (1.0 + spec.tank.regulator_margin) if ok else np.inf
        res = {"x": np.asarray(x, dtype=float).copy(), "PR_run": pr_run, "PR_start": pr_start, "length": length,
               "start_pressure_margin": (spec.tank.pressure - p_need) / spec.tank.pressure if ok else -1.0, "ok": ok}
        self.cache[key] = res
        self.history.append(res)
        return res


def optimise(design, n_samples=16, maxiter=40, seed=1) -> OptimisationResult:
    spec = design.spec
    bounds = _bounds(spec)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.array([getattr(spec.diffuser, n) for n, _ in VARIABLES], dtype=float)
    x0 = np.clip(x0, lo, hi)
    ev = _Evaluator(design)
    L_max = spec.optimisation.max_total_length

    def feasible(r):
        if not r["ok"]:
            return False
        if L_max > 0 and r["length"] > L_max:
            return False
        return r["start_pressure_margin"] >= 0.0

    base = ev(x0)
    sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=seed)
    samples = qmc.scale(sampler.random(n_samples), lo, hi)
    cands = [base] + [ev(s) for s in samples]
    feas = [c for c in cands if feasible(c)]
    start = min(feas, key=lambda c: c["PR_run"]) if feas else base

    scale = hi - lo

    def to_u(x):
        return (x - lo) / scale

    def to_x(u):
        return lo + np.clip(u, 0.0, 1.0) * scale

    def obj(u):
        r = ev(to_x(u))
        return r["PR_run"] if r["ok"] else 1e3

    cons = [{"type": "ineq", "fun": lambda u: ev(to_x(u))["start_pressure_margin"]}]
    if L_max > 0:
        cons.append({"type": "ineq", "fun": lambda u: (L_max - ev(to_x(u))["length"]) / L_max
                     if ev(to_x(u))["ok"] else -1.0})
    res = minimize(obj, to_u(start["x"]), method="SLSQP", bounds=[(0.0, 1.0)] * len(bounds), constraints=cons,
                   options={"maxiter": maxiter, "ftol": 1e-7, "eps": 1e-3})
    x_opt = to_x(res.x)
    best = ev(x_opt)
    if not feasible(best) or (feasible(start) and start["PR_run"] < best["PR_run"]):
        best = start
        x_opt = start["x"]
    return OptimisationResult([n for n, _ in VARIABLES], bounds, x0, base, np.asarray(x_opt), best, ev.history,
                              str(res.message))


def optimised_spec(design, result: OptimisationResult):
    spec = design.spec.copy()
    for (name, _), v in zip(VARIABLES, result.x_opt):
        setattr(spec.diffuser, name, float(v))
    return spec


# --------------------------------------------------------------------------
# Mach trade study
# --------------------------------------------------------------------------
def mach_sweep(spec, n_axis=60):
    """Key figures versus test-section Mach number with everything else fixed."""
    sw = spec.sweep
    rows = []
    for M in np.linspace(sw.mach_min, sw.mach_max, sw.n):
        s = spec.copy()
        s.design.mach = float(M)
        s.nozzle.n_axis = n_axis
        s.nozzle.inflection_angle_deg = 0.0
        try:
            d = design_tunnel(s, fast=True)
            b = simulate_blowdown(d, track_shock=False)
            rows.append({
                "mach": float(M), "PR_start": d.PR_start, "PR_run": d.PR_run,
                "p0_start_Pa": d.p0_start_set, "p0_run_Pa": d.p0_run_set,
                "mdot_run_kg_s": d.mdot_per_p0 * d.p0_run_set, "run_time_s": b.run_time,
                "T_test_K": d.test_static["T"], "T0_min_K": d.T0_min_no_liquefaction,
                "liquefaction_ratio": max(d.liquefaction.values()), "nozzle_length_m": d.nozzle.length,
                "total_length_m": d.geometry.length, "A_d2_over_At": d.A_second_throat / d.A_throat,
                "Re_unit_per_m": d.test_static["Re_unit"], "ok": True,
            })
        except (RuntimeError, ValueError) as exc:
            rows.append({"mach": float(M), "ok": False, "error": str(exc)})
    return rows
