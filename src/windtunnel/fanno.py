"""Fanno flow (adiabatic, constant-area duct with wall friction).

The friction factor used here is the Darcy factor f_D (= 4 x Fanning).
The classic tables use 4 f_F L*/D, which equals f_D L*/D.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

G = 1.4


def fanno_fL_D(M, g=G):
    """f_D L*/D: duct length needed to reach M = 1 from Mach M."""
    M = np.asarray(M, dtype=float)
    m2 = M * M
    return (1.0 - m2) / (g * m2) + (g + 1.0) / (2.0 * g) * np.log((g + 1.0) * m2 / (2.0 + (g - 1.0) * m2))


def fanno_p_pstar(M, g=G):
    M = np.asarray(M, dtype=float)
    return 1.0 / M * np.sqrt((g + 1.0) / (2.0 + (g - 1.0) * M * M))


def fanno_T_Tstar(M, g=G):
    M = np.asarray(M, dtype=float)
    return (g + 1.0) / (2.0 + (g - 1.0) * M * M)


def fanno_p0_p0star(M, g=G):
    M = np.asarray(M, dtype=float)
    return 1.0 / M * ((2.0 + (g - 1.0) * M * M) / (g + 1.0)) ** (0.5 * (g + 1.0) / (g - 1.0))


def darcy_friction_factor(Re, rel_roughness=0.0):
    """Haaland explicit approximation of the Colebrook equation (turbulent).
    For Re < 2300 the laminar value 64/Re is returned."""
    Re = float(Re)
    if Re < 2300.0:
        return 64.0 / max(Re, 1.0)
    term = (rel_roughness / 3.7) ** 1.11 + 6.9 / Re
    return (-1.8 * np.log10(term)) ** -2


def fanno_exit_mach(M_in, fL_D, g=G):
    """Mach at the end of a duct of friction length fL_D starting at M_in.

    Returns (M_out, choked). If the duct is longer than L*, the flow is
    choked: M_out = 1 is returned with choked = True.
    """
    if fL_D <= 0.0:
        return float(M_in), False
    if abs(M_in - 1.0) < 1e-12:
        return 1.0, True
    star = float(fanno_fL_D(M_in, g))
    rem = star - fL_D
    if rem <= 0.0:
        return 1.0, True
    if M_in > 1.0:
        hi = M_in
        return float(brentq(lambda M: fanno_fL_D(M, g) - rem, 1.0, hi, xtol=1e-13)), False
    return float(brentq(lambda M: fanno_fL_D(M, g) - rem, M_in, 1.0, xtol=1e-13)), False


def fanno_length_to_mach(M_in, M_out, g=G):
    """Friction length f_D L/D between two Mach numbers on the same branch."""
    return float(fanno_fL_D(M_in, g) - fanno_fL_D(M_out, g))
