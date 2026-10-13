"""Calorically perfect gas model and transport properties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gas:
    """Calorically perfect gas.

    Viscosity follows Sutherland's law, valid for air roughly between
    100 K and 1900 K. Below 100 K (high-Mach test sections) the error grows,
    which mainly affects the Reynolds number estimate, not the inviscid sizing.
    """

    gamma: float = 1.4
    R: float = 287.05  # J/(kg K)
    mu_ref: float = 1.716e-5  # Pa s at T_ref
    T_ref: float = 273.15  # K
    sutherland: float = 110.4  # K

    @property
    def cp(self) -> float:
        return self.gamma * self.R / (self.gamma - 1.0)

    @property
    def cv(self) -> float:
        return self.R / (self.gamma - 1.0)

    def viscosity(self, T):
        T = np.asarray(T, dtype=float)
        return self.mu_ref * (T / self.T_ref) ** 1.5 * (self.T_ref + self.sutherland) / (T + self.sutherland)

    def sound_speed(self, T):
        return np.sqrt(self.gamma * self.R * np.asarray(T, dtype=float))

    def density(self, p, T):
        return np.asarray(p, dtype=float) / (self.R * np.asarray(T, dtype=float))

    def unit_reynolds(self, p, T, M):
        """Reynolds number per metre, rho V / mu."""
        rho = self.density(p, T)
        V = np.asarray(M, dtype=float) * self.sound_speed(T)
        return rho * V / self.viscosity(T)


AIR = Gas()
