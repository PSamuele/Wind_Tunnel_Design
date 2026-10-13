"""Axisymmetric supersonic nozzle design by the method of characteristics.

Method (source-flow kernel + prescribed centreline, in the spirit of the
Sivells procedure):

1. Kernel: downstream of the throat arc the flow is a source (radial) flow
   from a virtual origin, bounded by a conical wall of half-angle omega
   (the inflection angle). On a sphere of radius R the Mach number satisfies
   A/A* = (R/R*)^2 with A* = 2 pi R*^2 (1 - cos omega) = pi r_t^2.
2. Inflection point I: the cone ends at the wall point I (wall Mach M_I).
   The source-flow C- characteristic through I is integrated analytically
   down to the axis, where it lands at x_B with Mach M_B. Everything left of
   this C- is exact source flow; it is the initial data line of the lattice.
   Near-sonic characteristics are almost normal to the flow, so M_B is much
   larger than M_I; omega and M_I are chosen so that
   nu(M_B) = nu_fraction * nu(M_E) and I lies downstream of the throat arc.
3. Centreline: downstream of x_B the axis Mach follows a quintic matching
   M, M', M'' of the source flow at x_B and reaching M_E at x_E with zero
   first and second derivatives (uniform flow afterwards). The length is the
   shortest one keeping the distribution concave.
4. Field: C+ lines start on the initial C- and on the axis points, C- lines
   end on the axis points. Each new point comes from a predictor-corrector
   unit process with the axisymmetric compatibility relations

       along C+ (dr/dx = tan(theta + mu)):  d(theta - nu) = -Q+ dx
       along C- (dr/dx = tan(theta - mu)):  d(theta + nu) =  Q- dx
       Q+- = sin(mu) sin(theta) / (r cos(theta +- mu)).

5. Wall: along each C- line the mass flow is integrated from the axis; the
   wall is where it equals the throat mass flow.

Upstream of I the wall is the cone, joined to the throat by a circular arc
of radius rc_ratio * r_t (the throat region is approximated by the source
flow, as in the classic kernel methods).

Checks: with a source-flow centreline the lattice must reproduce the
analytic source flow, and the exit radius must satisfy the isentropic area
ratio A_e/A* of the design Mach number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import isentropic as isen


# --------------------------------------------------------------------------
# Scalar helpers (math module: much faster than numpy for scalars)
# --------------------------------------------------------------------------
def _pm(M, g):
    k = math.sqrt((g + 1.0) / (g - 1.0))
    s = math.sqrt(max(M * M - 1.0, 0.0))
    return k * math.atan(s / k) - math.atan(s)


def _inv_pm(nu, g, M0=1.5):
    if nu <= 0.0:
        return 1.0
    M = max(M0, 1.0 + 1e-6)
    for _ in range(50):
        f = _pm(M, g) - nu
        d = math.sqrt(M * M - 1.0) / (M * (1.0 + 0.5 * (g - 1.0) * M * M))
        Mn = M - f / d
        if Mn <= 1.0:
            Mn = 0.5 * (M + 1.0)
        if abs(Mn - M) < 1e-13 * M:
            return Mn
        M = Mn
    return M


def _flux(M, g):
    """(rho/rho0) (V/a0): mass flux density normalised by rho0 a0."""
    t = 1.0 / (1.0 + 0.5 * (g - 1.0) * M * M)
    return t ** (1.0 / (g - 1.0)) * M * math.sqrt(t)


@dataclass
class _Pt:
    x: float
    r: float
    th: float
    nu: float
    M: float

    @property
    def mu(self):
        return math.asin(1.0 / self.M)


def _qplus(p: _Pt):
    mu = p.mu
    return math.sin(mu) * math.sin(p.th) / (p.r * math.cos(p.th + mu))


def _qminus(p: _Pt):
    mu = p.mu
    return math.sin(mu) * math.sin(p.th) / (p.r * math.cos(p.th - mu))


def unit_process(p1: _Pt, p2: _Pt, g: float, n_iter: int = 4, r_tiny: float = 1e-12) -> _Pt:
    """Interior point: p1 lies on the C+ through the new point, p2 on the C-."""
    P = None
    for it in range(n_iter):
        if P is None:
            ap, am = p1.th + p1.mu, p2.th - p2.mu
        else:
            ap = 0.5 * (p1.th + p1.mu + P.th + P.mu)
            am = 0.5 * (p2.th - p2.mu + P.th - P.mu)
        sp, sm = math.tan(ap), math.tan(am)
        x = (p2.r - p1.r + sp * p1.x - sm * p2.x) / (sp - sm)
        r = p1.r + sp * (x - p1.x)
        # Source terms; a parent on the axis (r = 0) takes the value at P
        if P is None:
            q1 = _qplus(p1) if p1.r > r_tiny else 0.0
            q2 = _qminus(p2) if p2.r > r_tiny else 0.0
            if p1.r <= r_tiny and p2.r > r_tiny:
                q1 = _qplus(_Pt(p2.x, p2.r, p2.th, p2.nu, p2.M))
            if p2.r <= r_tiny and p1.r > r_tiny:
                q2 = _qminus(_Pt(p1.x, p1.r, p1.th, p1.nu, p1.M))
            qp, qm = q1, q2
        else:
            qPp = _qplus(P) if P.r > r_tiny else 0.0
            qPm = _qminus(P) if P.r > r_tiny else 0.0
            q1 = _qplus(p1) if p1.r > r_tiny else qPp
            q2 = _qminus(p2) if p2.r > r_tiny else qPm
            qp, qm = 0.5 * (q1 + qPp), 0.5 * (q2 + qPm)
        a = (p1.th - p1.nu) - qp * (x - p1.x)  # theta - nu at P
        b = (p2.th + p2.nu) + qm * (x - p2.x)  # theta + nu at P
        th = 0.5 * (a + b)
        nu = 0.5 * (b - a)
        M = _inv_pm(nu, g, p1.M)
        P = _Pt(x, r, th, nu, M)
    return P


# --------------------------------------------------------------------------
# Centreline distribution
# --------------------------------------------------------------------------
def default_inflection_angle(M_E, rc_ratio=2.0, g=1.4):
    """Automatic inflection angle (see choose_kernel)."""
    return choose_kernel(M_E, rc_ratio, None, 0.75, g)[0]


@dataclass
class AxisDistribution:
    """Centreline Mach number: source flow up to x_B, quintic up to x_E,
    uniform M_E downstream."""

    R_star: float
    x_B: float
    x_E: float
    M_B: float
    M_E: float
    coeffs: np.ndarray  # quintic in xi = (x - x_B)/(x_E - x_B)
    g: float

    def mach(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty_like(x)
        src = x <= self.x_B
        if np.any(src):
            out[src] = isen.mach_from_area_ratio(np.maximum((x[src] / self.R_star) ** 2, 1.0), True, self.g)
        pol = (x > self.x_B) & (x < self.x_E)
        if np.any(pol):
            xi = (x[pol] - self.x_B) / (self.x_E - self.x_B)
            out[pol] = np.polyval(self.coeffs[::-1], xi)
        out[x >= self.x_E] = self.M_E
        return out


def _source_axis_derivs(x, R_star, g):
    """M, dM/dx, d2M/dx2 of source flow on the axis."""
    def M_of(xx):
        return isen.mach_from_area_ratio((xx / R_star) ** 2, True, g)

    h = 1e-4 * x
    m0, mp, mm = M_of(x), M_of(x + h), M_of(x - h)
    return m0, (mp - mm) / (2 * h), (mp - 2 * m0 + mm) / (h * h)


def _quintic(M_B, d1, d2, M_E, L):
    """Coefficients c0..c5 in xi for the end conditions."""
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 5],
        [0, 0, 2, 6, 12, 20],
    ], dtype=float)
    b = np.array([M_B, d1 * L, d2 * L * L, M_E, 0.0, 0.0])
    return np.linalg.solve(A, b)


def _mach_sup(ar, g, guess=2.0):
    """Supersonic Mach from A/A* (safeguarded Newton, scalar)."""
    if ar <= 1.0:
        return 1.0
    lo, hi = 1.0, 2.0
    while float(isen.area_ratio(hi, g)) < ar:
        hi *= 2.0
    M = guess if lo < guess < hi else 0.5 * (lo + hi)
    e = 0.5 * (g + 1.0) / (g - 1.0)
    for _ in range(60):
        A = (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** e
        f = A - ar
        if f > 0.0:
            hi = M
        else:
            lo = M
        d = A * (M * M - 1.0) / (M * (1.0 + 0.5 * (g - 1.0) * M * M))
        Mn = M - f / d if d > 0.0 else 0.5 * (lo + hi)
        if not lo < Mn < hi:
            Mn = 0.5 * (lo + hi)
        if abs(Mn - M) < 1e-13 * M:
            return Mn
        M = Mn
    return M


def source_c_minus(R_I, omega, R_star, g, n=400, overshoot=1.15):
    """Source-flow C- characteristic through the wall point at spherical
    radius R_I on the cone, from the axis up to overshoot * r_I (slightly
    outside the wall, for lattice coverage). RK4 with the radius r as
    parameter, n + 1 points equally spaced in r, ordered from the axis up."""
    def M_at(x, r, guess):
        return _mach_sup(max((math.hypot(x, r) / R_star) ** 2, 1.0 + 1e-14), g, guess)

    state = {"M": 1.5}

    def dxdr(x, r):
        M = M_at(x, r, state["M"])
        state["M"] = M
        return 1.0 / math.tan(math.atan2(r, x) - math.asin(1.0 / M))

    def rk4(x, r, h):
        k1 = dxdr(x, r)
        k2 = dxdr(x + 0.5 * h * k1, r + 0.5 * h)
        k3 = dxdr(x + 0.5 * h * k2, r + 0.5 * h)
        k4 = dxdr(x + h * k3, r + h)
        return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), r + h

    x_I, r_I = R_I * math.cos(omega), R_I * math.sin(omega)
    # down to the axis
    sub = 8
    rs = np.linspace(0.0, r_I, n + 1)
    xs = np.empty_like(rs)
    x, r = x_I, r_I
    xs[-1] = x
    for k in range(n, 0, -1):
        h = (rs[k - 1] - rs[k]) / sub
        for _ in range(sub):
            x, r = rk4(x, r, h)
        xs[k - 1] = x
    # overshoot above the wall
    n_up = max(3, int(n * (overshoot - 1.0)))
    ru = np.linspace(r_I, overshoot * r_I, n_up + 1)[1:]
    xu = np.empty_like(ru)
    x, r = x_I, r_I
    prev = r_I
    for k, rt in enumerate(ru):
        h = (rt - prev) / sub
        for _ in range(sub):
            x, r = rk4(x, r, h)
        xu[k] = x
        prev = rt
    pts = []
    for x, r in zip(np.concatenate([xs, xu]), np.concatenate([rs, ru])):
        M = M_at(x, r, 1.5)
        pts.append(_Pt(float(x), float(r), math.atan2(r, x), _pm(M, g), M))
    pts[0] = _Pt(pts[0].x, 0.0, 0.0, pts[0].nu, pts[0].M)
    return pts, n  # index n is the wall point


def _throat_tangent_radius(omega, rc_ratio):
    """Spherical radius (in units of R*) where the throat arc meets the cone."""
    rt = math.sqrt(2.0 * (1.0 - math.cos(omega)))  # r_t / R*
    rc = rc_ratio * rt
    x_tan = (rt + rc * (1.0 - math.cos(omega))) / math.tan(omega)
    return x_tan / math.cos(omega)


def _axis_mach_from_inflection(M_I, omega, g, n=40):
    """Centreline Mach where the source C- through the wall point M_I lands."""
    R_I = math.sqrt(float(isen.area_ratio(M_I, g)))
    pts, _ = source_c_minus(R_I, omega, 1.0, g, n, overshoot=1.0001)
    return pts[0].M


def choose_kernel(M_E, rc_ratio, omega=None, nu_fraction=0.75, g=1.4):
    """Inflection angle omega and inflection wall Mach M_I.

    The source C- through the inflection point reaches the axis at M_B with
    nu(M_B) = nu_fraction * nu(M_E) (the rest of the expansion is done by
    the centreline polynomial). The inflection point must lie downstream of
    the throat arc (M_I > M_tan(omega)). Without a user omega, omega is 80 %
    of the largest angle satisfying both conditions."""
    nu_target = nu_fraction * _pm(M_E, g)
    M_B_target = _inv_pm(nu_target, g)

    def M_tan(om):
        return _mach_sup(_throat_tangent_radius(om, rc_ratio) ** 2, g, 1.2)

    def MB_at_tan(om):
        M_I = M_tan(om) * 1.005
        M_B = _axis_mach_from_inflection(M_I, om, g)
        return M_B if M_B > M_I else math.inf  # invalid: C- runs into the sonic region

    if omega is None:
        lo, hi = math.radians(0.5), math.radians(22.0)
        if MB_at_tan(lo) > M_B_target:
            raise RuntimeError("no admissible inflection angle: reduce throat_curvature_ratio")
        if MB_at_tan(hi) < M_B_target:
            om_max = hi
        else:
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if MB_at_tan(mid) < M_B_target:
                    lo = mid
                else:
                    hi = mid
            om_max = lo
        omega = 0.8 * om_max
    m_lo = M_tan(omega) * 1.005
    if _axis_mach_from_inflection(m_lo, omega, g) >= M_B_target:
        if _axis_mach_from_inflection(m_lo, omega, g) >= M_E:
            raise RuntimeError(f"inflection angle {math.degrees(omega):.2f} deg too large for M = {M_E:g} "
                               "with this throat curvature")
        return omega, m_lo
    lo, hi = m_lo, M_B_target
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _axis_mach_from_inflection(mid, omega, g) < M_B_target:
            lo = mid
        else:
            hi = mid
    return omega, 0.5 * (lo + hi)


def build_axis_distribution(M_E, R_star, x_B, length_factor=1.0, g=1.4) -> AxisDistribution:
    m0, d1, d2 = _source_axis_derivs(x_B, R_star, g)
    if not m0 < M_E:
        raise ValueError("end-of-source Mach must be below the exit Mach")
    xi = np.linspace(0.0, 1.0, 401)

    def measures(L):
        c = _quintic(m0, d1, d2, M_E, L)
        dd = np.polyval(np.polyder(c[::-1], 2), xi) / (L * L)
        d = np.polyval(np.polyder(c[::-1], 1), xi) / L
        return dd.max(), d.min()

    # Shortest length with a concave distribution (M'' <= 0); if no length
    # gives a concave curve, the monotone one with the smallest positive M''.
    L_lo = max((M_E - m0) / d1, 1e-6)  # linear extrapolation reaches M_E
    best_L, best_val = None, np.inf
    for L in L_lo * np.geomspace(1.0, 30.0, 400):
        dd_max, d_min = measures(L)
        if d_min < -1e-9 * d1:
            continue
        if dd_max <= 1e-9 * abs(d2):
            best_L = L
            break
        if dd_max < best_val:
            best_L, best_val = L, dd_max
    if best_L is None:
        raise RuntimeError("could not build a monotone centreline distribution")
    L = best_L * length_factor
    c = _quintic(m0, d1, d2, M_E, L)
    return AxisDistribution(R_star, x_B, x_B + L, m0, M_E, c, g)


# --------------------------------------------------------------------------
# Nozzle design
# --------------------------------------------------------------------------
@dataclass
class MOCNozzle:
    """Result of the MOC design. Coordinates have the throat at x = 0."""

    M_exit: float
    r_throat: float
    r_exit: float
    omega: float
    x_wall: np.ndarray
    r_wall: np.ndarray
    theta_wall: np.ndarray
    M_wall: np.ndarray
    lattice_x: np.ndarray  # (n_lines, n_cols) arrays, NaN where not computed
    lattice_r: np.ndarray
    lattice_M: np.ndarray
    lattice_theta: np.ndarray
    axis: AxisDistribution
    x_shift: float  # local (source-origin) x of the throat
    area_ratio_error: float
    length: float
    x_inflection: float  # end of the conical (source-flow) wall, from the throat
    warnings: list = field(default_factory=list)

    @property
    def max_wall_angle(self):
        """Largest wall angle (the true inflection point of the contour; it can
        lie slightly downstream of the end of the cone)."""
        return float(np.max(self.theta_wall))

    def wall_function(self):
        """Return r(x) for 0 <= x <= length by linear interpolation."""
        return lambda x: np.interp(x, self.x_wall, self.r_wall)


def design_nozzle(M_E, r_t, omega=None, rc_ratio=2.0, n_axis=160, M_I=None, nu_fraction=0.75, length_factor=1.0,
                  g=1.4, axis_override=None) -> MOCNozzle:
    """Design an axisymmetric nozzle with uniform exit flow at M_E.

    omega: wall inflection angle [rad] (None: automatic, see choose_kernel).
    M_I: wall Mach number at the inflection point (None: automatic).
    nu_fraction: share of the exit Prandtl-Meyer angle reached on the axis
    at the end of the source-flow region.
    axis_override: optional callable x -> M replacing the centreline
    distribution downstream of x_B (used by the tests to check that the
    lattice reproduces pure source flow)."""
    if M_I is None:
        omega, M_I = choose_kernel(M_E, rc_ratio, omega, nu_fraction, g)
    elif omega is None:
        raise ValueError("M_I requires an explicit inflection angle")
    R_star = r_t / math.sqrt(2.0 * (1.0 - math.cos(omega)))
    R_I = R_star * math.sqrt(float(isen.area_ratio(M_I, g)))
    if R_I <= _throat_tangent_radius(omega, rc_ratio) * R_star:
        raise ValueError("inflection point upstream of the throat arc")
    warnings = []

    # Initial characteristic: source-flow C- through the inflection point,
    # from the axis (x_B) up to slightly outside the wall (analytic).
    n_q = max(30, int(n_axis) // 2)
    q, i_wall = source_c_minus(R_I, omega, R_star, g, n_q)
    ax = build_axis_distribution(M_E, R_star, q[0].x, length_factor, g)
    axis_mach = axis_override or ax.mach

    # Axis points downstream of x_B: up to x_E, then far enough for the
    # C- lines from the uniform region to reach the wall.
    mu_E = math.asin(1.0 / M_E)
    r_e_ideal = r_t * math.sqrt(float(isen.area_ratio(M_E, g)))
    x_F = ax.x_E + 2.4 * r_e_ideal / math.tan(mu_E)
    n_core = int(n_axis)
    x_core = np.linspace(ax.x_B, ax.x_E, n_core + 1)
    dx = x_core[1] - x_core[0]
    x_ext = np.arange(ax.x_E + dx, x_F + dx, dx)
    x_axis = np.concatenate([x_core, x_ext])
    M_axis = np.asarray(axis_mach(x_axis), dtype=float)
    M_axis[0] = q[0].M
    N = x_axis.size

    # Line ids: C+ lines start at q[m], ..., q[1] (ids 0..m-1) and at the
    # axis points (id m + j). cols[j][id] = point on C-_j and C+_id.
    m = len(q) - 1
    cols = [dict() for _ in range(N)]
    for k, p in enumerate(q):
        cols[0][m - k] = p
    target = math.pi * r_t * r_t * _flux(1.0, g)

    def mass_and_wall(col_pts):
        """Integrate the mass flow along a C- (ordered from the axis up)."""
        cum = 0.0
        for a, b in zip(col_pts[:-1], col_pts[1:]):
            ds = math.hypot(b.x - a.x, b.r - a.r)
            f0 = 2.0 * math.pi * a.r * _flux(a.M, g) * math.sin(a.mu)
            f1 = 2.0 * math.pi * b.r * _flux(b.M, g) * math.sin(b.mu)
            dm = 0.5 * (f0 + f1) * ds
            if cum + dm >= target:
                s = (target - cum) / dm
                return (a.x + s * (b.x - a.x), a.r + s * (b.r - a.r), a.th + s * (b.th - a.th),
                        a.M + s * (b.M - a.M)), cum + dm
            cum += dm
        return None, cum

    w0, _ = mass_and_wall(q)
    if w0 is None:
        raise RuntimeError("initial characteristic does not carry the throat mass flow")
    # consistency of the source kernel: the mass-flow wall point on the
    # initial C- must be the inflection point on the cone
    kernel_err = math.hypot(w0[0] - q[i_wall].x, w0[1] - q[i_wall].r) / q[i_wall].r
    if kernel_err > 2e-3:
        warnings.append(f"source-flow kernel: mass-flow wall point {100 * kernel_err:.2f} % off the cone")
    wall = [(q[i_wall].x, q[i_wall].r, omega, q[i_wall].M)]

    top = 0  # smallest line id still available on the previous column
    exit_reached = False
    for j in range(1, N):
        axis_id = m + j
        prev = _Pt(float(x_axis[j]), 0.0, 0.0, _pm(float(M_axis[j]), g), float(M_axis[j]))
        cols[j][axis_id] = prev
        new_top = axis_id
        cum = 0.0
        found = None
        for lid in range(axis_id - 1, top - 1, -1):
            if lid not in cols[j - 1]:
                break
            P = unit_process(cols[j - 1][lid], prev, g)
            if not (P.r > prev.r) or not math.isfinite(P.x):
                warnings.append(f"characteristic breakdown at C-_{j}, C+_{lid}")
                break
            cols[j][lid] = P
            new_top = lid
            if found is None:
                ds = math.hypot(P.x - prev.x, P.r - prev.r)
                f0 = 2.0 * math.pi * prev.r * _flux(prev.M, g) * math.sin(prev.mu)
                f1 = 2.0 * math.pi * P.r * _flux(P.M, g) * math.sin(P.mu)
                dm = 0.5 * (f0 + f1) * ds
                if cum + dm >= target:
                    sfr = (target - cum) / dm
                    found = (prev.x + sfr * (P.x - prev.x), prev.r + sfr * (P.r - prev.r),
                             prev.th + sfr * (P.th - prev.th), prev.M + sfr * (P.M - prev.M))
                cum += dm
            elif P.r > 1.15 * found[1]:
                break  # stop shortly above the wall
            prev = P
        top = new_top
        if found is not None:
            wall.append(found)
            if x_axis[j] > ax.x_E and abs(found[2]) < 1e-5 and abs(found[3] - M_E) < 1e-3 * M_E:
                exit_reached = True
                break
        else:
            warnings.append(f"C-_{j} did not reach the wall")
    if not exit_reached:
        warnings.append("uniform exit not reached on the wall; increase n_axis")

    n_cols = len(cols)
    ids = m + n_cols
    LX = np.full((ids, n_cols), np.nan)
    LR = np.full_like(LX, np.nan)
    LM = np.full_like(LX, np.nan)
    LT = np.full_like(LX, np.nan)
    for j, col in enumerate(cols):
        for lid, p in col.items():
            LX[lid, j], LR[lid, j], LM[lid, j], LT[lid, j] = p.x, p.r, p.M, p.th

    wall = np.array(wall)
    keep = np.concatenate([[True], np.diff(wall[:, 0]) > 0])
    wall = wall[keep]

    # Throat arc tangent to the source-flow cone r = x tan(omega)
    rc = rc_ratio * r_t
    x_t = (r_t + rc * (1.0 - math.cos(omega))) / math.tan(omega) - rc * math.sin(omega)
    x_tan = x_t + rc * math.sin(omega)
    phi = np.linspace(0.0, omega, 40)
    arc_x = x_t + rc * np.sin(phi)
    arc_r = r_t + rc * (1.0 - np.cos(phi))
    arc_M = np.array([isen.mach_from_area_ratio(max((R / ax.R_star) ** 2, 1.0), True, g)
                      for R in np.hypot(arc_x, arc_r)])
    arc_M[0] = 1.0
    x_I = float(wall[0, 0])
    if x_I <= x_tan:
        raise RuntimeError("inflection point upstream of the throat arc: reduce throat_curvature_ratio "
                           "or increase the end-of-source Mach number")
    cone_x = np.linspace(x_tan, x_I, 30)[1:-1]
    cone_r = cone_x * math.tan(omega)
    cone_M = np.array([isen.mach_from_area_ratio((x / math.cos(omega) / ax.R_star) ** 2, True, g) for x in cone_x])
    xw = np.concatenate([arc_x, cone_x, wall[:, 0]])
    rw = np.concatenate([arc_r, cone_r, wall[:, 1]])
    tw = np.concatenate([phi, np.full(cone_x.size, omega), wall[:, 2]])
    Mw = np.concatenate([arc_M, cone_M, wall[:, 3]])

    r_exit = float(rw[-1])
    err = (r_exit / r_t) ** 2 / float(isen.area_ratio(M_E, g)) - 1.0
    if abs(err) > 5e-3:
        warnings.append(f"exit area ratio error {100 * err:.2f} % exceeds 0.5 %")
    d_th = np.diff(tw[np.argmax(tw):])
    if np.any(d_th > 1e-6):
        warnings.append("wall angle not monotone downstream of the inflection point")

    return MOCNozzle(M_E, r_t, r_exit, omega, xw - x_t, rw, tw, Mw, LX - x_t, LR, LM, LT, ax, x_t, float(err),
                     float(xw[-1] - x_t), x_I - x_t, warnings)


def source_flow_mach(x, r, R_star, g=1.4):
    """Analytic source-flow Mach number at (x, r) (local coordinates)."""
    R = math.hypot(x, r)
    return isen.mach_from_area_ratio((R / R_star) ** 2, True, g)
