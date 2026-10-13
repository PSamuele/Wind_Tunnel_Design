"""Unsteady quasi-1D Euler solver for start-up / unstart transients.

Equations (per unit length, A = A(x)):

    d(rho A)/dt   + d(rho u A)/dx         = 0
    d(rho u A)/dt + d((rho u^2 + p) A)/dx = p dA/dx + S_f
    d(E A)/dt     + d(u (E + p) A)/dx     = 0

With moving walls (variable second throat) A = A(x, t): the energy
equation gains the wall work -p dA/dt; the other equations keep their form.

S_f collects momentum sinks: wall friction in constant-area ducts
(-f_D/(2D) rho u|u| A), honeycomb/screens and subsonic-diffuser losses
(-K q A / L over the device length). Adiabatic stationary walls do no work,
so the energy equation has no source.

Discretisation: cell-centred finite volumes, HLLC flux, MUSCL reconstruction
of primitive variables, SSP-RK3 time stepping. The default limiter is
minmod: with van Leer a nearly stationary normal shock in a nozzle keeps
emitting low-amplitude acoustic waves (a few % mass-flux oscillation behind
the shock); minmod damps them.

Boundaries: "stagnation" inlet (p0(t), T0(t), outgoing Riemann invariant
from the interior), "pressure" outlet (p_back when subsonic, extrapolation
when supersonic), or "transmissive".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class TransientResult:
    t: np.ndarray  # diagnostic times
    p0_inlet: np.ndarray
    x_shock: np.ndarray  # nan when no shock
    M_test: np.ndarray
    started: np.ndarray
    mdot_throat: np.ndarray
    mdot_exit: np.ndarray
    snap_t: list
    snap: list  # dicts with x, M, p, T, rho, u, p0
    x_cells: np.ndarray
    area_cells: np.ndarray
    n_steps: int
    events: dict = field(default_factory=dict)
    throat_area_ratio: np.ndarray | None = None  # A_d2(t)/A_t at the diagnostic times

    def mach_xt(self):
        """(t, x, M) arrays for an x-t diagram from the snapshots."""
        return np.array(self.snap_t), self.x_cells, np.array([s["M"] for s in self.snap])


def _haaland(Re, rel_rough):
    Re = np.maximum(Re, 1.0)
    turb = (-1.8 * np.log10((rel_rough / 3.7) ** 1.11 + 6.9 / Re)) ** -2
    return np.where(Re < 2300.0, 64.0 / Re, turb)


def _van_leer(a, b):
    ab = a * b
    s = a + b
    return np.where(ab > 0.0, 2.0 * ab / np.where(s == 0.0, 1.0, s), 0.0)


def _minmod(a, b):
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


LIMITERS = {"minmod": _minmod, "van_leer": _van_leer}


class QuasiOneDEuler:
    def __init__(self, x_faces, area_fn: Callable, gamma=1.4, R=287.05, mu_fn: Callable | None = None,
                 friction_mask=None, roughness=0.0, loss_regions=(), left_bc="stagnation", right_bc="pressure",
                 p_back=101325.0, T_back=288.15, limiter="minmod", area_xt: Callable | None = None):
        """area_xt(x, t): optional time-dependent area (moving walls); area_fn
        is then the area at t = 0."""
        self.g = gamma
        self.R = R
        self.xf = np.asarray(x_faces, dtype=float)
        self.N = self.xf.size - 1
        self.dx = np.diff(self.xf)
        if not np.allclose(self.dx, self.dx[0], rtol=1e-9):
            raise ValueError("uniform grid required")
        self.h = float(self.dx[0])
        self.xc = 0.5 * (self.xf[:-1] + self.xf[1:])
        self.area_xt = area_xt
        self._t_geom = None
        self.dAdt = np.zeros(self.N)
        self._set_area(lambda x: area_fn(x))
        self.mu_fn = mu_fn
        self.fmask = np.zeros(self.N, bool) if friction_mask is None else np.asarray(friction_mask, bool)
        self.rough = roughness
        # loss regions: list of (mask, K, mode) with mode "uniform" or "diffuser" (K scaled by (A/A_in)^2)
        self.loss = []
        for mask, K, mode in loss_regions:
            mask = np.asarray(mask, bool)
            if not np.any(mask) or K <= 0.0:
                continue
            L = mask.sum() * self.h
            scale = np.ones(self.N)
            if mode == "diffuser":
                A_in = self.Ac[mask][0]
                scale = (self.Ac / A_in) ** 2
            self.loss.append((mask, K * scale / L, mode))
        self.limiter = LIMITERS[limiter]
        self.left_bc = left_bc
        self.right_bc = right_bc
        self.p_back = p_back
        self.T_back = T_back

    def _cell_areas(self, fn):
        Af = np.asarray(fn(self.xf), dtype=float)
        # cell area: Simpson average over the cell
        Ac = (Af[:-1] + 4.0 * np.asarray(fn(self.xc), dtype=float) + Af[1:]) / 6.0
        return Af, Ac

    def _set_area(self, fn):
        self.Af, self.Ac = self._cell_areas(fn)
        self.dA = (self.Af[1:] - self.Af[:-1]) / self.h
        self.Dc = np.sqrt(4.0 * self.Ac / np.pi)

    def set_time(self, t):
        """Update the geometry for moving walls (no-op for fixed walls)."""
        if self.area_xt is None or t == self._t_geom:
            return
        self._set_area(lambda x: self.area_xt(x, t))
        dt = 1e-6
        _, Ap = self._cell_areas(lambda x: self.area_xt(x, t + dt))
        _, Am = self._cell_areas(lambda x: self.area_xt(x, max(t - dt, 0.0)))
        self.dAdt = (Ap - Am) / (dt + min(dt, t))
        self._t_geom = t

    # ------------------------------------------------------------------
    def prim(self, Q):
        A = self.Ac
        rho = Q[0] / A
        u = Q[1] / Q[0]
        E = Q[2] / A
        p = (self.g - 1.0) * (E - 0.5 * rho * u * u)
        return rho, u, p

    def cons(self, rho, u, p):
        E = p / (self.g - 1.0) + 0.5 * rho * u * u
        return np.array([rho * self.Ac, rho * u * self.Ac, E * self.Ac])

    def _ghosts(self, rho, u, p, t):
        g, R = self.g, self.R
        # left
        if self.left_bc == "stagnation":
            p0, T0 = self.p0_fn(t), self.T0_fn(t)
            a1 = math.sqrt(g * p[0] / rho[0])
            Rm = u[0] - 2.0 * a1 / (g - 1.0)
            a0sq = g * R * T0
            A2 = (g + 1.0) / (g - 1.0)
            disc = 4.0 * Rm * Rm - 4.0 * A2 * (0.5 * (g - 1.0) * Rm * Rm - a0sq)
            a = (-2.0 * Rm + math.sqrt(max(disc, 0.0))) / (2.0 * A2)
            ub = Rm + 2.0 * a / (g - 1.0)
            if ub < 0.0:
                ub, a = 0.0, math.sqrt(a0sq)
            Tb = a * a / (g * R)
            pb = p0 * (Tb / T0) ** (g / (g - 1.0))
            rb = pb / (R * Tb)
            L = (rb, ub, pb)
        else:
            L = (rho[0], u[0], p[0])
        # right
        if self.right_bc == "pressure":
            aN = math.sqrt(g * p[-1] / rho[-1])
            if u[-1] >= aN:
                Rr = (rho[-1], u[-1], p[-1])
            elif u[-1] > 0.0:
                pb = self.p_back
                rb = rho[-1] * (pb / p[-1]) ** (1.0 / g)
                Rp = u[-1] + 2.0 * aN / (g - 1.0)
                ab = math.sqrt(g * pb / rb)
                Rr = (rb, max(Rp - 2.0 * ab / (g - 1.0), 0.0), pb)
            else:
                Rr = (self.p_back / (R * self.T_back), u[-1], self.p_back)
        else:
            Rr = (rho[-1], u[-1], p[-1])
        return L, Rr

    def _hllc(self, rL, uL, pL, rR, uR, pR):
        g = self.g
        aL = np.sqrt(g * pL / rL)
        aR = np.sqrt(g * pR / rR)
        EL = pL / (g - 1.0) + 0.5 * rL * uL * uL
        ER = pR / (g - 1.0) + 0.5 * rR * uR * uR
        SL = np.minimum(uL - aL, uR - aR)
        SR = np.maximum(uL + aL, uR + aR)
        num = pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)
        den = rL * (SL - uL) - rR * (SR - uR)
        Ss = num / den
        FL = np.array([rL * uL, rL * uL * uL + pL, uL * (EL + pL)])
        FR = np.array([rR * uR, rR * uR * uR + pR, uR * (ER + pR)])

        def star(r, u, p, E, S):
            c = r * (S - u) / (S - Ss)
            return np.array([c, c * Ss, c * (E / r + (Ss - u) * (Ss + p / (r * (S - u))))])

        UL = np.array([rL, rL * uL, EL])
        UR = np.array([rR, rR * uR, ER])
        FsL = FL + SL * (star(rL, uL, pL, EL, SL) - UL)
        FsR = FR + SR * (star(rR, uR, pR, ER, SR) - UR)
        F = np.where(SL >= 0.0, FL, np.where(Ss >= 0.0, FsL, np.where(SR > 0.0, FsR, FR)))
        return F

    def rhs(self, Q, t):
        g = self.g
        self.set_time(t)
        rho, u, p = self.prim(Q)
        (rl, ul, pl), (rr, ur, pr) = self._ghosts(rho, u, p, t)
        # padded primitive arrays with two ghost cells per side
        W = np.empty((3, self.N + 4))
        W[:, 2:-2] = (rho, u, p)
        W[:, 0] = W[:, 1] = (rl, ul, pl)
        W[:, -1] = W[:, -2] = (rr, ur, pr)
        d = np.diff(W, axis=1)  # N + 3 differences
        slope = self.limiter(d[:, :-1], d[:, 1:])  # N + 2 cells (indices 1 .. N + 2)
        WL = W[:, 1:-2] + 0.5 * slope[:, :-1]  # left state at faces 0..N
        WR = W[:, 2:-1] - 0.5 * slope[:, 1:]  # right state at faces 0..N
        # positivity fallback to first order
        bad = (WL[0] <= 0) | (WL[2] <= 0) | (WR[0] <= 0) | (WR[2] <= 0)
        if np.any(bad):
            WL[:, bad] = W[:, 1:-2][:, bad]
            WR[:, bad] = W[:, 2:-1][:, bad]
        F = self._hllc(WL[0], WL[1], WL[2], WR[0], WR[1], WR[2]) * self.Af
        dQ = -(F[:, 1:] - F[:, :-1]) / self.h
        dQ[1] += p * self.dA
        if self.area_xt is not None:
            dQ[2] -= p * self.dAdt  # work of the moving walls
        # momentum sinks
        if np.any(self.fmask) and self.mu_fn is not None:
            m = self.fmask
            T = p[m] / (rho[m] * self.R)
            Re = rho[m] * np.abs(u[m]) * self.Dc[m] / self.mu_fn(T)
            f = _haaland(Re, self.rough / self.Dc[m])
            dQ[1, m] -= f / (2.0 * self.Dc[m]) * rho[m] * u[m] * np.abs(u[m]) * self.Ac[m]
        for mask, k_per_len, mode in self.loss:
            q = 0.5 * rho * u * np.abs(u)
            sink = k_per_len * q * self.Ac
            if mode == "diffuser":
                a = np.sqrt(g * p / rho)
                sink = np.where(np.abs(u) < a, sink, 0.0)
            dQ[1] -= np.where(mask, sink, 0.0)
        return dQ

    # ------------------------------------------------------------------
    def run(self, t_end, p0_fn=None, T0_fn=None, initial=None, cfl=0.7, snapshot_times=(), diag_dt=None,
            diagnostics: Callable | None = None, max_steps=5_000_000):
        """Integrate to t_end. initial = (rho, u, p) arrays at cell centres."""
        self.p0_fn = p0_fn
        self.T0_fn = T0_fn
        rho, u, p = (np.asarray(a, dtype=float).copy() for a in initial)
        self.set_time(0.0)
        Q = self.cons(rho, u, p)
        t = 0.0
        snaps_t, snaps = [], []
        st = sorted(snapshot_times)
        k_snap = 0
        diag = []
        next_diag = 0.0
        n = 0
        g = self.g
        while k_snap < len(st) and st[k_snap] <= t:
            snaps_t.append(t)
            snaps.append(self.state_dict(rho, u, p))
            k_snap += 1
        while t < t_end - 1e-15 and n < max_steps:
            self.set_time(t)
            rho, u, p = self.prim(Q)
            a = np.sqrt(g * p / rho)
            dt = cfl * self.h / np.max(np.abs(u) + a)
            dt = min(dt, t_end - t)
            if k_snap < len(st) and st[k_snap] > t:
                dt = min(dt, st[k_snap] - t)
            if diagnostics is not None and t >= next_diag:
                diag.append(diagnostics(t, rho, u, p))
                next_diag = t + (diag_dt or 0.0)
            Q1 = Q + dt * self.rhs(Q, t)
            Q2 = 0.75 * Q + 0.25 * (Q1 + dt * self.rhs(Q1, t + dt))
            Q = Q / 3.0 + 2.0 / 3.0 * (Q2 + dt * self.rhs(Q2, t + 0.5 * dt))
            t += dt
            n += 1
            if not np.all(np.isfinite(Q)):
                raise FloatingPointError(f"Euler solver diverged at t = {t:.6g} s")
            self.set_time(t)
            while k_snap < len(st) and t >= st[k_snap] - 1e-14:
                r_, u_, p_ = self.prim(Q)
                snaps_t.append(t)
                snaps.append(self.state_dict(r_, u_, p_))
                k_snap += 1
        self.set_time(t)
        rho, u, p = self.prim(Q)
        if diagnostics is not None:
            diag.append(diagnostics(t, rho, u, p))
        return Q, t, n, snaps_t, snaps, diag

    def state_dict(self, rho, u, p):
        g, R = self.g, self.R
        T = p / (rho * R)
        a = np.sqrt(g * p / rho)
        M = u / a
        p0 = p * (1.0 + 0.5 * (g - 1.0) * M * M) ** (g / (g - 1.0))
        return {"rho": rho.copy(), "u": u.copy(), "p": p.copy(), "T": T, "M": M, "p0": p0, "A": self.Ac.copy()}


# --------------------------------------------------------------------------
# Shock detection
# --------------------------------------------------------------------------
def detect_shocks(x, M, p, x_min=-np.inf, jump=1.15, width=4):
    """Positions of compression shocks: supersonic-to-subsonic transitions
    with a static-pressure jump > `jump` over `width` cells."""
    out = []
    n = M.size
    i = 0
    while i < n - width:
        if x[i] >= x_min and M[i] > 1.0 and M[i + width] < 1.0 and p[i + width] / p[i] > jump:
            seg = slice(i, i + width + 1)
            k = i + int(np.argmax(np.diff(p[seg])))
            out.append(0.5 * (x[k] + x[k + 1]))
            i += width
        i += 1
    return out


# --------------------------------------------------------------------------
# Tunnel start-up scenario
# --------------------------------------------------------------------------
def p0_schedule(design, ts):
    """Piecewise-linear supply pressure: open to the starting set point,
    hold, reduce to the running set point, hold, then (optionally) reduce
    below the running ratio to show unstart."""
    p_a = design.spec.design.p_ambient
    pts = [(0.0, p_a), (ts.valve_time, design.p0_start_set)]
    t = ts.valve_time + ts.hold_time
    pts.append((t, design.p0_start_set))
    t += ts.ramp_down_time
    pts.append((t, design.p0_run_set))
    t += ts.hold_time
    pts.append((t, design.p0_run_set))
    if ts.unstart_demo:
        t += ts.ramp_down_time
        p_low = 0.85 * design.PR_run * p_a
        pts.append((t, p_low))
        t += ts.hold_time
        pts.append((t, p_low))
    tt, pp = np.array(pts).T
    return tt, pp


def _tunnel_transient(design, ts, tt, pp, area_xt=None, throat_ratio=None) -> TransientResult:
    """Run the Euler solver on the tunnel with the supply schedule (tt, pp)
    and, optionally, moving walls area_xt(x, t)."""
    spec = design.spec
    geom = design.geometry
    gas = design.gas
    p_a, T_a, T0 = spec.design.p_ambient, spec.design.T_ambient, spec.design.T0
    xf = np.linspace(geom.x[0], geom.x[-1], ts.n_cells + 1)
    area = lambda x: np.pi * geom.radius(x) ** 2
    xc = 0.5 * (xf[:-1] + xf[1:])

    def mask(name):
        s = geom.segment(name)
        return (xc >= s.x0) & (xc <= s.x1)

    fmask = (mask("test section") | mask("second throat")) if ts.friction else None
    sc = geom.segment("settling chamber")
    dev = (xc >= sc.x0 + 0.4 * sc.length) & (xc <= sc.x0 + 0.6 * sc.length)
    if not np.any(dev):
        dev = np.abs(xc - 0.5 * (sc.x0 + sc.x1)) == np.min(np.abs(xc - 0.5 * (sc.x0 + sc.x1)))
    K_dev = spec.settling.K_honeycomb + spec.settling.K_screens
    dif = mask("subsonic diffuser")
    K_dif = design.flow.s.diffuser_K(0.015)
    losses = [(dev, K_dev, "uniform"), (dif, K_dif, "diffuser")]
    # model drag: K = Cd A_m / A_ts over the cell of the model station
    if design.drag.fLD > 0.0:
        ts_seg = geom.segment("test section")
        x_m = ts_seg.x0 + spec.model.position * ts_seg.length
        m_cell = np.abs(xc - x_m) == np.min(np.abs(xc - x_m))
        losses.append((m_cell, design.drag.cd * spec.model.frontal_area / design.A_test, "uniform"))
    solver = QuasiOneDEuler(xf, area, gas.gamma, gas.R, gas.viscosity, fmask, spec.test_section.wall_roughness,
                            losses, "stagnation", "pressure", p_a, T_a, ts.limiter, area_xt=area_xt)
    t_end = float(tt[-1])
    p0_fn = lambda t: float(np.interp(t, tt, pp))
    T0_fn = lambda t: T0
    rho0 = np.full(xc.size, p_a / (gas.R * T_a))
    init = (rho0, np.zeros(xc.size), np.full(xc.size, p_a))

    ts_seg = geom.segment("test section")
    in_ts = (xc >= ts_seg.x0) & (xc <= ts_seg.x1)
    i_mid = int(np.argmin(np.abs(xc - (ts_seg.x0 + 0.25 * ts_seg.length))))  # upstream of the model
    i_thr = int(np.argmin(np.abs(xc - geom.segment("contraction").x1)))
    x_throat = geom.segment("contraction").x1
    x_d2 = geom.segment("second throat").x0
    M_design = spec.design.mach

    def diagnostics(t, rho, u, p):
        a = np.sqrt(gas.gamma * p / rho)
        M = u / a
        shocks = detect_shocks(xc, M, p, x_min=x_throat)
        xs = shocks[0] if shocks else np.nan
        started = bool(np.all(M[in_ts] > 1.0) and (not shocks or shocks[0] >= x_d2 - 1e-9)
                       and M[i_mid] > 0.9 * M_design)
        return (t, p0_fn(t), xs, M[i_mid], started, rho[i_thr] * u[i_thr] * solver.Ac[i_thr],
                rho[-1] * u[-1] * solver.Ac[-1], throat_ratio(t) if throat_ratio else np.nan)

    snap_times = np.linspace(0.0, t_end, max(ts.n_snapshots, 2))
    diag_dt = t_end / 1500.0
    _, _, nsteps, s_t, s_list, diag = solver.run(t_end, p0_fn, T0_fn, init, ts.cfl, snap_times, diag_dt,
                                                  diagnostics)
    diag = list(zip(*diag))
    res = TransientResult(np.array(diag[0]), np.array(diag[1]), np.array(diag[2]), np.array(diag[3]),
                          np.array(diag[4], bool), np.array(diag[5]), np.array(diag[6]), s_t, s_list, xc,
                          solver.Ac, nsteps)
    res.throat_area_ratio = np.array(diag[7])
    # key events
    st = res.started
    ev = {"schedule_t": tt.tolist(), "schedule_p0": pp.tolist()}
    if np.any(st):
        k = int(np.argmax(st))
        ev["t_start"] = float(res.t[k])
        ev["p0_at_start"] = float(res.p0_inlet[k])
        ev["PR_at_start"] = float(res.p0_inlet[k] / p_a)
        after = np.where(~st[k:])[0]
        if after.size:
            k2 = k + int(after[0])
            ev["t_unstart"] = float(res.t[k2])
            ev["PR_at_unstart"] = float(res.p0_inlet[k2] / p_a)
            if throat_ratio is not None:
                ev["A_d2/A_t_at_unstart"] = float(throat_ratio(res.t[k2]))
    res.events = ev
    return res


def startup_transient(design, ts=None) -> TransientResult:
    """Fixed second throat: valve opening, start, running and (optionally)
    unstart."""
    ts = ts or design.spec.transient
    tt, pp = p0_schedule(design, ts)
    return _tunnel_transient(design, ts, tt, pp)


def variable_throat_transient(design, r_closed_geom, PR_run_closed, close_time, ts=None) -> TransientResult:
    """Variable second throat: open at start, closed after the start while
    p0 is reduced to the closed-throat running set point, then (optionally)
    p0 below the closed running ratio to show unstart.

    r_closed_geom: geometry of the closed throat with the same x-layout."""
    spec = design.spec
    ts = ts or spec.transient
    p_a = spec.design.p_ambient
    g_open = design.geometry
    p0_run_cl = PR_run_closed * p_a * (1.0 + spec.run.run_margin)
    t1 = ts.valve_time
    t2 = t1 + ts.hold_time  # started, throat still open
    t3 = t2 + close_time  # throat closed, p0 at the closed running set point
    t4 = t3 + ts.hold_time
    pts = [(0.0, p_a), (t1, design.p0_start_set), (t2, design.p0_start_set), (t3, p0_run_cl), (t4, p0_run_cl)]
    if ts.unstart_demo:
        t5 = t4 + ts.ramp_down_time
        p_low = 0.85 * PR_run_closed * p_a
        pts += [(t5, p_low), (t5 + ts.hold_time, p_low)]
    tt, pp = np.array(pts).T

    def s_of_t(t):
        return float(np.clip((t - t2) / close_time, 0.0, 1.0))

    def area_xt(x, t):
        s = s_of_t(t)
        r = g_open.radius(x) + s * (r_closed_geom.radius(x) - g_open.radius(x))
        return np.pi * r ** 2

    A_t = design.A_throat

    def throat_ratio(t):
        s = s_of_t(t)
        r = g_open.r_second_throat + s * (r_closed_geom.r_second_throat - g_open.r_second_throat)
        return np.pi * r * r / A_t

    res = _tunnel_transient(design, ts, tt, pp, area_xt=area_xt, throat_ratio=throat_ratio)
    res.events["close_start_s"] = t2
    res.events["close_end_s"] = t3
    return res
