"""Normal shocks, oblique shocks, conical (Taylor-Maccoll) shocks and the
classic quasi-1D shock-in-nozzle problem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

from . import isentropic as isen

G = 1.4


# --------------------------------------------------------------------------
# Normal shock
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalShock:
    M1: float
    M2: float
    p2_p1: float
    rho2_rho1: float
    T2_T1: float
    p02_p01: float


def normal_shock(M1, g=G) -> NormalShock:
    M1 = float(M1)
    if M1 < 1.0:
        raise ValueError("normal shock requires M1 >= 1")
    m2 = M1 * M1
    M2 = np.sqrt((1.0 + 0.5 * (g - 1.0) * m2) / (g * m2 - 0.5 * (g - 1.0)))
    p21 = 1.0 + 2.0 * g / (g + 1.0) * (m2 - 1.0)
    r21 = (g + 1.0) * m2 / ((g - 1.0) * m2 + 2.0)
    T21 = p21 / r21
    p0r = normal_shock_p0_ratio(M1, g)
    return NormalShock(M1, float(M2), float(p21), float(r21), float(T21), float(p0r))


def normal_shock_p0_ratio(M1, g=G):
    """Total-pressure ratio p02/p01 across a normal shock (vectorised)."""
    M1 = np.asarray(M1, dtype=float)
    m2 = M1 * M1
    a = ((g + 1.0) * m2 / ((g - 1.0) * m2 + 2.0)) ** (g / (g - 1.0))
    b = ((g + 1.0) / (2.0 * g * m2 - (g - 1.0))) ** (1.0 / (g - 1.0))
    return np.where(M1 <= 1.0, 1.0, a * b)


def normal_shock_M2(M1, g=G):
    M1 = np.asarray(M1, dtype=float)
    m2 = M1 * M1
    return np.sqrt((1.0 + 0.5 * (g - 1.0) * m2) / (g * m2 - 0.5 * (g - 1.0)))


def mach_from_p0_ratio(p0r, g=G):
    """Upstream Mach of a normal shock with given p02/p01 (0 < p0r <= 1)."""
    if p0r >= 1.0:
        return 1.0
    hi = 2.0
    while normal_shock_p0_ratio(hi, g) > p0r:
        hi *= 2.0
    return brentq(lambda M: normal_shock_p0_ratio(M, g) - p0r, 1.0, hi, xtol=1e-14)


# --------------------------------------------------------------------------
# Oblique shock (wedge / planar)
# --------------------------------------------------------------------------
def theta_from_beta(M1, beta, g=G):
    """Flow deflection for a given shock angle (theta-beta-M relation)."""
    sb = np.sin(beta)
    num = 2.0 / np.tan(beta) * (M1 * M1 * sb * sb - 1.0)
    den = M1 * M1 * (g + np.cos(2.0 * beta)) + 2.0
    return np.arctan(num / den)


def max_deflection(M1, g=G):
    """Maximum attached deflection theta_max and the shock angle where it occurs."""
    mu = np.arcsin(1.0 / M1)
    res = minimize_scalar(lambda b: -theta_from_beta(M1, b, g), bounds=(mu, 0.5 * np.pi), method="bounded",
                          options={"xatol": 1e-12})
    return float(-res.fun), float(res.x)


@dataclass(frozen=True)
class ObliqueShock:
    M1: float
    theta: float
    beta: float
    M2: float
    p2_p1: float
    T2_T1: float
    p02_p01: float
    attached: bool


def oblique_shock(M1, theta, weak=True, g=G) -> ObliqueShock:
    """Oblique shock for deflection theta (rad). Returns attached=False if
    theta exceeds the maximum deflection (detached bow shock)."""
    theta_max, beta_max = max_deflection(M1, g)
    if theta > theta_max:
        return ObliqueShock(M1, theta, np.nan, np.nan, np.nan, np.nan, np.nan, False)
    mu = np.arcsin(1.0 / M1)
    f = lambda b: theta_from_beta(M1, b, g) - theta
    if theta <= 0.0:
        beta = mu
    elif weak:
        beta = brentq(f, mu + 1e-12, beta_max, xtol=1e-14)
    else:
        beta = brentq(f, beta_max, 0.5 * np.pi - 1e-12, xtol=1e-14)
    Mn1 = M1 * np.sin(beta)
    ns = normal_shock(max(Mn1, 1.0), g)
    M2 = ns.M2 / np.sin(beta - theta)
    return ObliqueShock(M1, theta, float(beta), float(M2), ns.p2_p1, ns.T2_T1, ns.p02_p01, True)


# --------------------------------------------------------------------------
# Conical shock (Taylor-Maccoll)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ConeShock:
    M1: float
    cone_angle: float
    beta: float
    Mc: float  # Mach on the cone surface
    pc_p1: float  # cone-surface pressure / free-stream static pressure
    attached: bool


def _taylor_maccoll_rhs(theta, y, g):
    Vr, Vt = y
    A = 0.5 * (g - 1.0) * (1.0 - Vr * Vr - Vt * Vt)
    dVt = (Vr * Vt * Vt - A * (2.0 * Vr + Vt / np.tan(theta))) / (A - Vt * Vt)
    return [Vt, dVt]


def _cone_angle_for_shock(M1, beta, g):
    """Integrate Taylor-Maccoll inward from the shock; return cone angle and
    the non-dimensional surface speed V/V_max."""
    delta = theta_from_beta(M1, beta, g)
    Mn1 = M1 * np.sin(beta)
    M2 = normal_shock_M2(Mn1, g) / np.sin(beta - delta)
    V = (2.0 / ((g - 1.0) * M2 * M2) + 1.0) ** -0.5
    y0 = [V * np.cos(beta - delta), -V * np.sin(beta - delta)]

    def ev(theta, y, g_):
        return y[1]

    ev.terminal = True
    ev.direction = 1
    sol = solve_ivp(_taylor_maccoll_rhs, (beta, 1e-6), y0, args=(g,), events=ev, rtol=1e-10, atol=1e-12)
    if sol.t_events[0].size == 0:
        return np.nan, np.nan
    theta_c = float(sol.t_events[0][0])
    Vc = float(sol.y_events[0][0][0])
    return theta_c, Vc


def cone_shock(M1, cone_angle, g=G) -> ConeShock:
    """Attached conical shock on a cone of half-angle cone_angle (rad) at zero
    incidence, from the Taylor-Maccoll equation."""
    mu = np.arcsin(1.0 / M1)
    betas = np.linspace(mu + 1e-4, 0.5 * np.pi - 1e-3, 60)
    thetas = np.array([_cone_angle_for_shock(M1, b, g)[0] for b in betas])
    valid = np.isfinite(thetas)
    i_max = int(np.nanargmax(np.where(valid, thetas, -1.0)))
    if cone_angle >= thetas[i_max]:
        return ConeShock(M1, cone_angle, np.nan, np.nan, np.nan, False)
    i_lo = int(np.argmax(valid))
    f = lambda b: _cone_angle_for_shock(M1, b, g)[0] - cone_angle
    lo = betas[i_lo]
    if f(lo) > 0.0:
        lo = mu + 1e-7
    beta = brentq(f, lo, betas[i_max], xtol=1e-12)
    _, Vc = _cone_angle_for_shock(M1, beta, g)
    Mc = np.sqrt(2.0 / (g - 1.0) * Vc * Vc / (1.0 - Vc * Vc))
    Mn1 = M1 * np.sin(beta)
    ns = normal_shock(Mn1, g)
    delta = theta_from_beta(M1, beta, g)
    M2 = ns.M2 / np.sin(beta - delta)
    # Isentropic compression between shock and cone surface
    pc_p1 = ns.p2_p1 * isen.p_p0(Mc, g) / isen.p_p0(M2, g)
    return ConeShock(M1, cone_angle, float(beta), float(Mc), float(pc_p1), True)


# --------------------------------------------------------------------------
# Quasi-1D shock in a converging-diverging nozzle
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NozzleShockSolution:
    regime: str
    pb_p0: float
    As_At: float  # shock station area ratio (nan if no shock in nozzle)
    Ms: float  # Mach upstream of the shock
    Me: float  # exit Mach


def nozzle_regime_limits(Ae_At, g=G):
    """Back-pressure ratios bounding the classic nozzle regimes."""
    Me_sub = isen.mach_from_area_ratio(Ae_At, supersonic=False, g=g)
    Me_sup = isen.mach_from_area_ratio(Ae_At, supersonic=True, g=g)
    p_sub = float(isen.p_p0(Me_sub, g))  # choked, fully subsonic exit
    p_design = float(isen.p_p0(Me_sup, g))
    p_exit_shock = p_design * normal_shock(Me_sup, g).p2_p1
    return p_sub, p_exit_shock, p_design


def exit_mach_from_pressure_area(pe_p01, Ae_At, g=G):
    """Subsonic exit Mach from mass conservation when the exit static
    pressure and the throat (choked, upstream total pressure) are known."""
    gam = ((g + 1.0) / 2.0) ** (-0.5 * (g + 1.0) / (g - 1.0))
    C = gam / (pe_p01 * Ae_At)
    X = (-1.0 + np.sqrt(1.0 + 2.0 * (g - 1.0) * C * C)) / (g - 1.0)
    return float(np.sqrt(X))


def shock_in_nozzle(Ae_At, pb_p0, g=G) -> NozzleShockSolution:
    """Normal-shock position in the divergent for back pressure pb/p0."""
    p_sub, p_exit_shock, p_design = nozzle_regime_limits(Ae_At, g)
    if pb_p0 >= 1.0:
        return NozzleShockSolution("no flow", pb_p0, np.nan, np.nan, 0.0)
    if pb_p0 > p_sub:
        Me = float(isen.mach_from_p_p0(pb_p0, g))
        return NozzleShockSolution("subsonic (unchoked)", pb_p0, np.nan, np.nan, Me)
    if pb_p0 >= p_exit_shock:
        Me = exit_mach_from_pressure_area(pb_p0, Ae_At, g)
        p02_p01 = pb_p0 / float(isen.p_p0(Me, g))
        Ms = mach_from_p0_ratio(min(p02_p01, 1.0), g)
        As = float(isen.area_ratio(Ms, g))
        return NozzleShockSolution("normal shock in divergent", pb_p0, As, Ms, Me)
    Me = isen.mach_from_area_ratio(Ae_At, True, g)
    if pb_p0 > p_design * (1.0 + 1e-9):
        regime = "over-expanded (oblique shocks outside)"
    elif pb_p0 < p_design * (1.0 - 1e-9):
        regime = "under-expanded (expansion fans outside)"
    else:
        regime = "design (adapted)"
    return NozzleShockSolution(regime, pb_p0, np.nan, np.nan, float(Me))


def back_pressure_from_shock_area(Ae_At, As_At, g=G):
    """Inverse of shock_in_nozzle: back pressure that puts the shock at As/At."""
    Ms = isen.mach_from_area_ratio(As_At, True, g)
    p0r = normal_shock_p0_ratio(Ms, g)
    Me = isen.mach_from_area_ratio(Ae_At * p0r, False, g)
    return float(p0r * isen.p_p0(Me, g))


# --------------------------------------------------------------------------
# Ramp (shock-pair) compression in a convergent
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ShockPair:
    M_in: float
    theta: float
    regime: str  # "regular reflection", "Mach reflection", "detached"
    beta1: float
    M_mid: float
    beta2: float
    M_out: float
    p0_ratio: float


def shock_pair(M, theta, g=G) -> ShockPair:
    """Planar estimate of the compression at a convergent entrance of
    half-angle theta: an oblique shock turns the flow by theta towards the
    axis and its reflection turns it back parallel. If either deflection
    exceeds the attached limit, a normal shock is used for that step
    (detached shock / Mach reflection)."""
    s1 = oblique_shock(M, theta, True, g)
    if not s1.attached:
        ns = normal_shock(M, g)
        return ShockPair(M, theta, "detached", np.nan, ns.M2, np.nan, ns.M2, ns.p02_p01)
    s2 = oblique_shock(s1.M2, theta, True, g) if s1.M2 > 1.0 else None
    if s2 is None or not s2.attached:
        ns = normal_shock(max(s1.M2, 1.0), g)
        return ShockPair(M, theta, "Mach reflection", s1.beta, s1.M2, np.nan, ns.M2, s1.p02_p01 * ns.p02_p01)
    return ShockPair(M, theta, "regular reflection", s1.beta, s1.M2, s2.beta, s2.M2, s1.p02_p01 * s2.p02_p01)


def max_regular_pair_angle(M, g=G):
    """Largest convergent half-angle giving a regular shock-pair reflection
    at Mach M (planar estimate)."""
    th_max, _ = max_deflection(M, g)
    lo, hi = 1e-6, th_max * (1.0 - 1e-9)
    if shock_pair(M, hi, g).regime == "regular reflection":
        return hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if shock_pair(M, mid, g).regime == "regular reflection":
            lo = mid
        else:
            hi = mid
    return lo

