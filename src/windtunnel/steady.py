"""Steady quasi-1D flow through the complete tunnel.

The solver marches along x on the geometry samples:

* variable-area segments: isentropic, A/A*_loc on the current branch;
* constant-area segments: Fanno flow (Darcy factor from the local Reynolds
  number at the segment inlet, Haaland formula);
* settling chamber: honeycomb + screens as a total-pressure drop K q;
* subsonic diffuser: isentropic area change plus a total-pressure loss
  K q_in distributed linearly along its length;
* normal shocks at prescribed positions;
* test model: its drag D = Cd q A_m is a momentum sink equivalent to a
  Fanno step with f L/D = Cd A_m / A_ts (fully mixed-out loss), applied at
  the model station;
* optional ramp compression: an oblique shock pair at the entrance of the
  second-throat convergent (planar estimate) when the flow is supersonic.

Total-pressure losses are tracked through the local critical area
A*_loc = A*_ref p0_ref / p0_loc (mass conservation in adiabatic flow).
Friction inside variable-area segments is neglected; the boundary-layer
displacement thickness is not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from . import fanno as fn
from . import losses as ls
from . import shocks as sh
from .gas import Gas
from .geometry import TunnelGeometry


# --------------------------------------------------------------------------
# Fast scalar inversions (safeguarded Newton)
# --------------------------------------------------------------------------
def _ar(M, g):
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** (0.5 * (g + 1.0) / (g - 1.0))


def mach_from_ar(ar, supersonic, g, guess=None):
    if ar <= 1.0:
        return 1.0
    if supersonic:
        lo, hi = 1.0, 2.0
        while _ar(hi, g) < ar:
            hi *= 2.0
    else:
        lo, hi = 1e-10, 1.0
    M = guess if (guess is not None and lo < guess < hi) else (0.5 * (lo + hi) if supersonic else min(0.5, 0.6 / ar))
    for _ in range(80):
        f = _ar(M, g) - ar
        if f > 0.0:
            if supersonic:
                hi = M
            else:
                lo = M
        else:
            if supersonic:
                lo = M
            else:
                hi = M
        d = _ar(M, g) * (M * M - 1.0) / (M * (1.0 + 0.5 * (g - 1.0) * M * M))
        Mn = M - f / d if d != 0.0 else 0.5 * (lo + hi)
        if not (lo < Mn < hi):
            Mn = 0.5 * (lo + hi)
        if abs(Mn - M) < 1e-13 * max(M, 1e-3):
            return Mn
        M = Mn
    return M


def _fanno(M, g):
    m2 = M * M
    return (1.0 - m2) / (g * m2) + (g + 1.0) / (2.0 * g) * math.log((g + 1.0) * m2 / (2.0 + (g - 1.0) * m2))


def mach_from_fanno(val, supersonic, g, guess):
    if val <= 0.0:
        return 1.0
    if supersonic:
        lo, hi = 1.0, max(guess, 1.0 + 1e-9) * 1.0000001
        while _fanno(hi, g) < val:
            hi *= 1.5
    else:
        lo, hi = 1e-10, 1.0
    M = guess if lo < guess < hi else 0.5 * (lo + hi)
    for _ in range(80):
        f = _fanno(M, g) - val
        # fanno(M) decreases towards 1 on the subsonic branch, increases on the supersonic one
        if (f > 0.0) == supersonic:
            hi = M
        else:
            lo = M
        d = 4.0 * (M * M - 1.0) / (g * M ** 3 * (2.0 + (g - 1.0) * M * M))
        Mn = M - f / d if d != 0.0 else 0.5 * (lo + hi)
        if not (lo < Mn < hi):
            Mn = 0.5 * (lo + hi)
        if abs(Mn - M) < 1e-13 * M:
            return Mn
        M = Mn
    return M


def _fanno_p0(M, g):
    return 1.0 / M * ((2.0 + (g - 1.0) * M * M) / (g + 1.0)) ** (0.5 * (g + 1.0) / (g - 1.0))


def _ns(M1, g):
    m2 = M1 * M1
    M2 = math.sqrt((1.0 + 0.5 * (g - 1.0) * m2) / (g * m2 - 0.5 * (g - 1.0)))
    a = ((g + 1.0) * m2 / ((g - 1.0) * m2 + 2.0)) ** (g / (g - 1.0))
    b = ((g + 1.0) / (2.0 * g * m2 - (g - 1.0))) ** (1.0 / (g - 1.0))
    return M2, a * b


def _pp0(M, g):
    return (1.0 + 0.5 * (g - 1.0) * M * M) ** (-g / (g - 1.0))


def _tt0(M, g):
    return 1.0 / (1.0 + 0.5 * (g - 1.0) * M * M)


# --------------------------------------------------------------------------
# Solution container
# --------------------------------------------------------------------------
@dataclass
class AxialSolution:
    x: np.ndarray
    r: np.ndarray
    M: np.ndarray
    p0: np.ndarray  # local total pressure [Pa]
    p: np.ndarray
    T: np.ndarray
    rho: np.ndarray
    V: np.ndarray
    p0_supply: float  # nozzle supply total pressure [Pa]
    p0_upstream: float  # total pressure upstream of honeycomb/screens [Pa]
    T0: float
    shocks: list  # [(x, M1)]
    exit_p: float
    exit_M: float
    exit_p0: float
    ok: bool
    message: str
    events: list = field(default_factory=list)  # (component, p0_in, p0_out)

    @property
    def p_over_p0(self):
        return self.p / self.p0_supply

    def as_columns(self):
        return {
            "x [m]": self.x, "r [m]": self.r, "area [m^2]": np.pi * self.r ** 2, "Mach [-]": self.M,
            "static pressure [Pa]": self.p, "total pressure [Pa]": self.p0,
            "p / p0 supply [-]": self.p_over_p0, "p0 / p0 supply [-]": self.p0 / self.p0_supply,
            "static temperature [K]": self.T, "density [kg/m^3]": self.rho, "velocity [m/s]": self.V,
        }


# --------------------------------------------------------------------------
# Quasi-1D marcher
# --------------------------------------------------------------------------
class QuasiOneD:
    def __init__(self, geom: TunnelGeometry, gas: Gas, K_settling: float, diffuser_half_angle: float,
                 diffuser_area_ratio: float, K_diffuser_override: float = -1.0, wall_roughness: float = 0.0,
                 friction: bool = True, model_station: float | None = None, model_fLD: float = 0.0,
                 model_fLD_subsonic: float | None = None, convergent_model: str = "isentropic",
                 convergent_half_angle: float = 0.0):
        self.geom = geom
        self.gas = gas
        self.K_settling = K_settling
        self.diffuser_half_angle = diffuser_half_angle
        self.diffuser_area_ratio = diffuser_area_ratio
        self.K_override = K_diffuser_override
        self.eps = wall_roughness
        self.friction = friction
        self.A_t = math.pi * geom.r_throat ** 2
        self.x_throat = geom.segment("contraction").x1
        self.model_station = model_station if model_fLD > 0.0 else None
        self.model_fLD = model_fLD
        self.model_fLD_sub = model_fLD if model_fLD_subsonic is None else model_fLD_subsonic
        self.convergent_model = convergent_model
        self.convergent_half_angle = convergent_half_angle
        self._pair_cache = {}

    def shock_pair(self, M):
        key = round(M, 9)
        if key not in self._pair_cache:
            self._pair_cache[key] = sh.shock_pair(M, self.convergent_half_angle, self.gas.gamma)
        return self._pair_cache[key]

    def diffuser_K(self, f_darcy, area_ratio=None):
        if self.K_override >= 0.0:
            return self.K_override
        n = self.diffuser_area_ratio if area_ratio is None else area_ratio
        return ls.conical_diffuser_K(self.diffuser_half_angle, n, f_darcy)

    def march(self, p0, T0, shocks=(), throat_mach=1.0) -> AxialSolution:
        """March the full tunnel for supply total pressure p0 [Pa].

        shocks: x positions of normal shocks (applied only where the local
        flow is supersonic). throat_mach < 1 gives the unchoked solution.
        """
        g, R = self.gas.gamma, self.gas.R
        geom = self.geom
        shocks = sorted(float(s) for s in shocks)
        extra = shocks + ([self.model_station] if self.model_station is not None else [])
        xs_all = np.unique(np.concatenate([geom.x, np.array(extra, dtype=float)]))

        A_ref = self.A_t / _ar(throat_mach, g) if throat_mach < 1.0 else self.A_t
        p0_ref = p0
        X, Rr, MM, P0 = [], [], [], []
        shock_list, events = [], []
        ok, msg = True, "ok"

        branch_sup = False
        p0_loc = p0
        # Settling chamber: devices located at mid-length
        sc = geom.segment("settling chamber")
        A_sc = math.pi * geom.r_settling ** 2
        M_sc = mach_from_ar(A_sc / A_ref, False, g)
        ratio_dev = ls.device_p0_ratio(self.K_settling, M_sc, g)
        p0_up = p0 / ratio_dev
        events.append(("honeycomb + screens", p0_up, p0))
        x_dev = 0.5 * (sc.x0 + sc.x1)

        seg_idx = 0
        segs = geom.segments
        f_seg = None
        seg_drop = 0.0  # non-shock total-pressure drop in the current segment [Pa]
        M = M_sc
        diff_state = None
        pending = list(shocks)
        conv_done = False

        def record(x, r, M_, p0_):
            X.append(x)
            Rr.append(r)
            MM.append(M_)
            P0.append(p0_)

        prev_x = None
        r_all = geom.radius(xs_all)
        for k_x, x in enumerate(xs_all):
            while seg_idx < len(segs) - 1 and x > segs[seg_idx].x1 + 1e-12:
                # leaving a segment
                if segs[seg_idx].name != "settling chamber":
                    events.append((segs[seg_idx].name, p0_loc + seg_drop, p0_loc))
                seg_idx += 1
                seg_drop = 0.0
                f_seg = None
                diff_state = None
            seg = segs[seg_idx]
            r = float(r_all[k_x])
            A = math.pi * r * r
            if seg.name == "settling chamber":
                p0_loc = p0_up if x < x_dev else p0
                Astar = A_ref * p0_ref / p0_loc
                M = mach_from_ar(A / Astar, False, g, M)
                record(x, r, M, p0_loc)
                prev_x = x
                continue
            Astar = A_ref * p0_ref / p0_loc
            if seg.kind == "constant" and self.friction and prev_x is not None and x > seg.x0 + 1e-12:
                if f_seg is None:
                    # Reynolds number at segment inlet
                    T = T0 * _tt0(M, g)
                    p = p0_loc * _pp0(M, g)
                    rho = p / (R * T)
                    V = M * math.sqrt(g * R * T)
                    D = 2.0 * r
                    Re = rho * V * D / float(self.gas.viscosity(T))
                    f_seg = fn.darcy_friction_factor(Re, self.eps / D)
                D = 2.0 * r
                dx = x - max(prev_x, seg.x0)
                val = _fanno(M, g) - f_seg * dx / D
                if val <= 0.0:
                    ok, msg = False, f"friction choking in {seg.name} at x = {x:.4f} m"
                    record(x, r, 1.0, p0_loc)
                    break
                M_new = mach_from_fanno(val, branch_sup, g, M)
                step_ratio = _fanno_p0(M_new, g) / _fanno_p0(M, g)
                seg_drop += p0_loc * (1.0 - step_ratio)
                p0_loc *= step_ratio
                M = M_new
            elif seg.kind == "diffuser" and not branch_sup:
                if diff_state is None:
                    T = T0 * _tt0(M, g)
                    p = p0_loc * _pp0(M, g)
                    rho = p / (R * T)
                    V = M * math.sqrt(g * R * T)
                    D = 2.0 * geom.r_second_throat
                    Re = rho * V * D / float(self.gas.viscosity(T))
                    f = fn.darcy_friction_factor(Re, self.eps / D)
                    q_in = 0.5 * g * M * M * p
                    # The diffuser may start mid-segment (after a shock)
                    x0d = max(seg.x0, prev_x)
                    n_eff = (geom.r_exit / float(geom.radius(x0d))) ** 2
                    diff_state = (self.diffuser_K(f, n_eff) * q_in, x0d, seg.x1 - x0d, p0_loc)
                dp_tot, x0d, Ld, p0_in = diff_state
                p0_new = p0_in - dp_tot * (x - x0d) / Ld
                seg_drop += p0_loc - p0_new
                p0_loc = p0_new
                Astar = A_ref * p0_ref / p0_loc
                if A / Astar < 1.0 - 1e-9:
                    ok, msg = False, f"choking in {seg.name} at x = {x:.4f} m"
                    record(x, r, 1.0, p0_loc)
                    break
                M = mach_from_ar(A / Astar, False, g, M)
            elif seg.kind != "constant":
                if seg.kind == "convergent" and not conv_done:
                    conv_done = True
                    if branch_sup and self.convergent_model == "oblique_shocks":
                        pair = self.shock_pair(M)
                        events.append((f"oblique shock pair ({pair.regime}, M = {M:.3f})", p0_loc,
                                       p0_loc * pair.p0_ratio))
                        p0_loc *= pair.p0_ratio
                        Astar = A_ref * p0_ref / p0_loc
                        if pair.regime != "regular reflection":
                            branch_sup = False
                            shock_list.append((x, M))
                if A / Astar < 1.0 - 1e-9:
                    ok, msg = False, f"choking in {seg.name} at x = {x:.4f} m (A/A* = {A / Astar:.4f})"
                    record(x, r, 1.0, p0_loc)
                    break
                if seg.kind == "contraction" and abs(x - self.x_throat) < 1e-12:
                    M = throat_mach
                    branch_sup = throat_mach >= 1.0
                else:
                    M = mach_from_ar(A / Astar, branch_sup, g, M)
            record(x, r, M, p0_loc)
            # Shock at this station
            while pending and abs(pending[0] - x) < 1e-12:
                pending.pop(0)
                if branch_sup and M > 1.0:
                    M2, pr = _ns(M, g)
                    shock_list.append((x, M))
                    events.append((f"normal shock (M = {M:.3f})", p0_loc, p0_loc * pr))
                    p0_loc *= pr
                    M = M2
                    branch_sup = False
                    record(x, r, M, p0_loc)
            # Test model at this station
            if self.model_station is not None and abs(x - self.model_station) < 1e-12:
                fld = self.model_fLD if branch_sup else self.model_fLD_sub
                val = _fanno(M, g) - fld
                if val <= 0.0:
                    ok, msg = False, f"model drag chokes the test section at x = {x:.4f} m"
                    break
                M_new = mach_from_fanno(val, branch_sup, g, M)
                ratio = _fanno_p0(M_new, g) / _fanno_p0(M, g)
                events.append((f"model drag ({'supersonic' if branch_sup else 'subsonic'})", p0_loc,
                               p0_loc * ratio))
                p0_loc *= ratio
                M = M_new
                record(x, r, M, p0_loc)
            prev_x = x
        else:
            events.append((segs[seg_idx].name, p0_loc + seg_drop, p0_loc))

        X, Rr, MM, P0 = map(np.array, (X, Rr, MM, P0))
        T = T0 * _tt0(MM, g)
        P = P0 * _pp0(MM, g)
        rho = P / (R * T)
        V = MM * np.sqrt(g * R * T)
        exit_M = float(MM[-1])
        return AxialSolution(X, Rr, MM, P0, P, T, rho, V, p0, p0_up, T0, shock_list, float(P[-1]), exit_M,
                             float(P0[-1]), ok, msg, events)


# --------------------------------------------------------------------------
# Operating-point analyses
# --------------------------------------------------------------------------
class TunnelFlow:
    """Started / unstarted operating points of a given tunnel."""

    def __init__(self, solver: QuasiOneD, T0: float, p_a: float):
        self.s = solver
        self.T0 = T0
        self.p_a = p_a
        g = solver.geom
        self.x_throat = g.segment("contraction").x1
        self.x_nozzle_exit = g.segment("nozzle").x1
        self.x_ts_end = g.segment("test section").x1
        self.x_d2_start = g.segment("second throat").x0
        self.x_d2_end = g.segment("second throat").x1
        self.x_exit = g.segment("subsonic diffuser").x1
        self._pr_guess = 3.0

    # ---- basic: p0 needed for a shock at x_s
    def pressure_ratio_for_shock(self, x_s, n_iter=6, rtol=1e-7):
        """Return (p0/p_a, solution) such that a normal shock sits at x_s and
        the exit static pressure equals p_a. nan if impossible.

        p0/p_a depends on p0 only through the Reynolds number (friction), so
        the fixed-point iteration converges in two or three marches."""
        p0 = self._pr_guess * self.p_a
        sol = None
        for _ in range(n_iter):
            sol = self.s.march(p0, self.T0, shocks=[x_s])
            if not sol.ok or sol.exit_M >= 1.0 or not sol.shocks:
                return float("nan"), sol
            p0_new = p0 * self.p_a / sol.exit_p
            if abs(p0_new - p0) < rtol * p0:
                break
            p0 = p0_new
        self._pr_guess = p0 / self.p_a
        return p0 / self.p_a, sol

    def pressure_ratio_without_shock(self, n_iter=6, rtol=1e-7):
        """p0/p_a with no imposed normal shock and subsonic exit at p_a (used
        when shocks generated inside the solver, e.g. a detached ramp shock,
        already make the flow subsonic)."""
        p0 = self._pr_guess * self.p_a
        sol = None
        for _ in range(n_iter):
            sol = self.s.march(p0, self.T0)
            if not sol.ok or sol.exit_M >= 1.0:
                return float("nan"), sol
            p0_new = p0 * self.p_a / sol.exit_p
            if abs(p0_new - p0) < rtol * p0:
                break
            p0 = p0_new
        return p0 / self.p_a, sol

    def critical_area_after_shock(self, x_s, p0):
        """Local A* at the end of the test section with a shock at x_s."""
        sol = self.s.march(p0, self.T0, shocks=[x_s])
        i = np.searchsorted(sol.x, self.x_ts_end, side="right") - 1
        return self.s.A_t * sol.p0_supply / sol.p0[i]

    def _extremum(self, x_lo, x_hi, sign, n):
        """Coarse scan + bounded Brent refinement of sign * PR(x_s)."""
        xs = np.linspace(x_lo, x_hi, n)
        prs = np.array([self.pressure_ratio_for_shock(x)[0] for x in xs])
        if not np.any(np.isfinite(prs)):
            return float("nan"), None
        vals = np.where(np.isfinite(prs), sign * prs, np.inf)
        k = int(np.argmin(vals))
        a, b = xs[max(k - 1, 0)], xs[min(k + 1, n - 1)]

        def f(x):
            pr = self.pressure_ratio_for_shock(x)[0]
            return sign * pr if np.isfinite(pr) else np.inf

        res = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-5 * (x_hi - x_lo)})
        if np.isfinite(res.fun) and res.fun < vals[k]:
            return sign * float(res.fun), float(res.x)
        return float(prs[k]), float(xs[k])

    def starting(self, n=12):
        """Starting pressure ratio: max p0/p_a over shock positions between
        the throat and the end of the test section."""
        pr, x_s = self._extremum(self.x_throat + 1e-6, self.x_ts_end - 1e-6, -1.0, n)
        if x_s is None:
            return float("nan"), None, None
        pr, sol = self.pressure_ratio_for_shock(x_s)
        return pr, x_s, sol

    def running(self, n=14):
        """Minimum p0/p_a of a started solution (shock downstream of the
        second-throat entrance)."""
        pr, x_s = self._extremum(self.x_d2_start + 1e-6, self.x_exit - 1e-6, 1.0, n)
        if x_s is None:
            return float("nan"), None, None
        pr, sol = self.pressure_ratio_for_shock(x_s)
        return pr, x_s, sol

    def shock_position(self, p0, x_lo, x_hi, n=40):
        """Shock position in [x_lo, x_hi] matching exit p = p_a at supply p0.
        Returns the most downstream root (stable branch) or None."""
        xs = np.linspace(x_lo + 1e-6, x_hi - 1e-6, n)

        def resid(x_s):
            sol = self.s.march(p0, self.T0, shocks=[x_s])
            if not sol.ok or not sol.shocks or sol.exit_M >= 1.0:
                return float("nan")
            return sol.exit_p / self.p_a - 1.0

        vals = np.array([resid(x) for x in xs])
        roots = []
        for i in range(n - 1):
            a, b = vals[i], vals[i + 1]
            if np.isfinite(a) and np.isfinite(b) and a * b <= 0.0:
                roots.append(brentq(resid, xs[i], xs[i + 1], xtol=1e-9))
        return roots[-1] if roots else None

    def unchoked(self, p0):
        """Fully subsonic solution (throat not choked) for supply p0."""
        def resid(Mt):
            sol = self.s.march(p0, self.T0, throat_mach=Mt)
            return sol.exit_p / self.p_a - 1.0

        try:
            Mt = brentq(resid, 1e-4, 0.999999, xtol=1e-10)
        except ValueError:
            return None
        return self.s.march(p0, self.T0, throat_mach=Mt)

    def operating_point(self, p0, started):
        """Steady solution at supply p0 for the given history state.

        Returns (regime, x_shock or None, solution)."""
        pr = p0 / self.p_a
        if started:
            # started branch: stable root downstream of the second throat
            x_s = self.shock_position(p0, self.x_d2_start, self.x_exit)
            if x_s is not None:
                return "started", x_s, self.s.march(p0, self.T0, shocks=[x_s])
            sol = self.s.march(p0, self.T0)
            if sol.ok and sol.exit_M > 1.0 and sol.exit_p <= self.p_a * 1.0:
                return "started (shock expelled, supersonic exit)", None, sol
            if sol.ok and sol.exit_M > 1.0:
                return "started (shock expelled, over-expanded exit)", None, sol
        # unstarted
        choke_pr, _ = self.pressure_ratio_for_shock(self.x_throat + 1e-6)
        if not np.isfinite(choke_pr) or pr < choke_pr:
            sol = self.unchoked(p0)
            return "subsonic (unchoked)", None, sol
        x_s = self.shock_position(p0, self.x_throat, self.x_ts_end)
        if x_s is not None:
            return "unstarted (shock in nozzle / test section)", x_s, self.s.march(p0, self.T0, shocks=[x_s])
        return "unresolved", None, None

    def back_pressure_sweep(self, pr_start, pr_run, n=24, pr_max_factor=1.3):
        """Steady hysteresis map: shock position vs p0/p_a on the
        increasing (unstarted) and decreasing (started) branches."""
        out = {"up": [], "down": []}
        pr_lo = 1.02
        for pr in np.linspace(pr_lo, pr_max_factor * pr_start, n):
            started = pr >= pr_start
            regime, x_s, sol = self.operating_point(pr * self.p_a, started)
            out["up"].append((float(pr), regime, x_s, sol))
        for pr in np.linspace(pr_max_factor * pr_start, pr_lo, n):
            started = pr >= pr_run
            regime, x_s, sol = self.operating_point(pr * self.p_a, started)
            out["down"].append((float(pr), regime, x_s, sol))
        return out
