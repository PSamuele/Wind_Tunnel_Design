"""Loss models: settling-chamber devices, conical subsonic diffuser,
liquefaction check."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from . import isentropic as isen

R_UNIVERSAL = 8.314462618  # J/(mol K)
P_ATM = 101325.0


def q_over_p0(M, g=1.4):
    """Dynamic pressure over local total pressure: (g/2) M^2 p/p0."""
    return 0.5 * g * M * M * isen.p_p0(M, g)


def device_p0_ratio(K, M, g=1.4):
    """Total-pressure ratio across a device with loss dp0 = K q (low Mach)."""
    return 1.0 - K * q_over_p0(M, g)


def conical_diffuser_K(half_angle, area_ratio, f_darcy=0.015):
    """Loss coefficient of a conical diffuser referred to the inlet dynamic
    pressure, dp0 = K q_in.

    Idelchik-type correlation (Handbook of Hydraulic Resistance, conical
    diffusers with uniform inlet profile):
        K = K_friction + K_expansion
        K_friction  = f_D / (8 sin a) (1 - 1/n^2)
        K_expansion = 3.2 tan(a)^1.25 (1 - 1/n)^2,  a = half-angle
    The expansion part is capped at the Borda-Carnot value (1 - 1/n)^2.
    It is an incompressible correlation: for the post-shock subsonic flow
    (M ~ 0.5-0.7) it is an engineering estimate. Check against the handbook
    or replace it with diffuser.K_override for a final design.
    """
    n = float(area_ratio)
    if n <= 1.0:
        return 0.0
    a = float(half_angle)
    k_fr = f_darcy / (8.0 * math.sin(a)) * (1.0 - 1.0 / n ** 2)
    bc = (1.0 - 1.0 / n) ** 2
    k_exp = min(3.2 * math.tan(a) ** 1.25, 1.0) * bc
    return k_fr + k_exp


# --------------------------------------------------------------------------
# Liquefaction (Clausius-Clapeyron, constant latent heat)
# --------------------------------------------------------------------------
SPECIES = {
    # normal boiling point [K], latent heat of vaporisation [J/mol], mole fraction in air
    "N2": (77.355, 5.57e3, 0.7808),
    "O2": (90.188, 6.82e3, 0.2095),
}


def saturation_pressure(T, species):
    Tb, dH, _ = SPECIES[species]
    return P_ATM * np.exp(-dH / R_UNIVERSAL * (1.0 / np.asarray(T, dtype=float) - 1.0 / Tb))


def liquefaction_margin(p_static, T_static):
    """For each species: partial pressure / saturation pressure.

    A value >= 1 means the gas is saturated or supersaturated. Real flows
    tolerate some supersaturation (short residence time), so this is a
    conservative criterion.
    """
    out = {}
    for sp, (_, _, x) in SPECIES.items():
        out[sp] = float(x * p_static / saturation_pressure(T_static, sp))
    return out


def minimum_T0_no_liquefaction(M, p0, g=1.4):
    """Smallest stagnation temperature keeping every species unsaturated in
    the test section (isentropic expansion to Mach M)."""
    pr = float(isen.p_p0(M, g))
    tr = float(isen.T_T0(M, g))

    def worst(T0):
        return max(liquefaction_margin(p0 * pr, T0 * tr).values()) - 1.0

    if worst(5000.0) > 0:
        return float("inf")
    if worst(1.0 / tr * 30.0) < 0:  # already safe at a 30 K static temperature
        return 30.0 / tr
    return float(brentq(worst, 30.0 / tr, 5000.0, xtol=1e-6))
