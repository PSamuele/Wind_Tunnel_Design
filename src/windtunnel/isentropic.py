"""Isentropic quasi-1D relations for a calorically perfect gas.

All functions accept scalars or numpy arrays. Angles are in radians.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

G = 1.4


def T_T0(M, g=G):
    M = np.asarray(M, dtype=float)
    return 1.0 / (1.0 + 0.5 * (g - 1.0) * M * M)


def p_p0(M, g=G):
    return T_T0(M, g) ** (g / (g - 1.0))


def rho_rho0(M, g=G):
    return T_T0(M, g) ** (1.0 / (g - 1.0))


def area_ratio(M, g=G):
    """A/A* as a function of Mach number."""
    M = np.asarray(M, dtype=float)
    e = 0.5 * (g + 1.0) / (g - 1.0)
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** e


def _scalar_mach_from_area_ratio(ar, supersonic, g):
    if ar < 1.0:
        raise ValueError(f"A/A* = {ar:.6f} < 1: flow would be choked upstream")
    if abs(ar - 1.0) < 1e-12:
        return 1.0
    f = lambda M: area_ratio(M, g) - ar
    if supersonic:
        hi = 2.0
        while f(hi) < 0.0:
            hi *= 2.0
        return brentq(f, 1.0, hi, xtol=1e-14, rtol=1e-13)
    return brentq(f, 1e-12, 1.0, xtol=1e-14, rtol=1e-13)


def mach_from_area_ratio(ar, supersonic=True, g=G):
    """Invert A/A* on the chosen branch."""
    ar_arr = np.asarray(ar, dtype=float)
    if ar_arr.ndim == 0:
        return _scalar_mach_from_area_ratio(float(ar_arr), supersonic, g)
    return np.array([_scalar_mach_from_area_ratio(a, supersonic, g) for a in ar_arr.ravel()]).reshape(ar_arr.shape)


def mach_from_p_p0(pr, g=G):
    """Mach number from static-to-total pressure ratio."""
    pr = np.asarray(pr, dtype=float)
    return np.sqrt(2.0 / (g - 1.0) * (pr ** (-(g - 1.0) / g) - 1.0))


def mach_from_T_T0(tr, g=G):
    tr = np.asarray(tr, dtype=float)
    return np.sqrt(2.0 / (g - 1.0) * (1.0 / tr - 1.0))


def prandtl_meyer(M, g=G):
    """Prandtl-Meyer function nu(M) in radians (M >= 1)."""
    M = np.asarray(M, dtype=float)
    k = np.sqrt((g + 1.0) / (g - 1.0))
    s = np.sqrt(np.maximum(M * M - 1.0, 0.0))
    return k * np.arctan(s / k) - np.arctan(s)


def mach_from_prandtl_meyer(nu, g=G):
    """Inverse Prandtl-Meyer function (Newton iterations, vectorised)."""
    nu = np.asarray(nu, dtype=float)
    nu_max = 0.5 * np.pi * (np.sqrt((g + 1.0) / (g - 1.0)) - 1.0)
    if np.any(nu < -1e-12) or np.any(nu >= nu_max):
        raise ValueError("Prandtl-Meyer angle out of range")
    nu_c = np.maximum(nu, 0.0)
    # Starting guess: small-nu expansion for M near 1, blended with a
    # large-M estimate; Newton then converges in a few steps.
    M = 1.0 + (1.5 * nu_c * (g + 1.0) / 2.0) ** (2.0 / 3.0) * 0.5
    M = np.maximum(M, 1.0 + 1e-9)
    for _ in range(60):
        f = prandtl_meyer(M, g) - nu_c
        dnu = np.sqrt(M * M - 1.0) / (M * (1.0 + 0.5 * (g - 1.0) * M * M))
        step = f / dnu
        M_new = M - step
        M_new = np.where(M_new <= 1.0, 0.5 * (M + 1.0), M_new)
        if np.all(np.abs(M_new - M) < 1e-13 * M):
            M = M_new
            break
        M = M_new
    M = np.where(nu_c < 1e-14, 1.0, M)
    return M if M.ndim else float(M)


def mach_angle(M):
    return np.arcsin(1.0 / np.asarray(M, dtype=float))


def mass_flow_function(M, g=G):
    """m_dot sqrt(R T0) / (p0 A) as a function of local Mach number."""
    M = np.asarray(M, dtype=float)
    return np.sqrt(g) * M * (1.0 + 0.5 * (g - 1.0) * M * M) ** (-0.5 * (g + 1.0) / (g - 1.0))


def choked_mass_flow(p0, T0, A_star, g=G, R=287.05):
    """Mass flow through a sonic throat of area A_star."""
    return p0 * A_star / np.sqrt(R * T0) * mass_flow_function(1.0, g)


def static_state(M, p0, T0, g=G, R=287.05):
    """Static p, T, rho, V for a given Mach and stagnation state."""
    T = T0 * T_T0(M, g)
    p = p0 * p_p0(M, g)
    rho = p / (R * T)
    V = M * np.sqrt(g * R * T)
    return p, T, rho, V
