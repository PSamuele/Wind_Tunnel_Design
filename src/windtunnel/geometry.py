"""Axisymmetric tunnel geometry r(x): settling chamber, contraction, MOC
nozzle, test section, second-throat convergent, second throat and subsonic
diffuser."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .moc import MOCNozzle


@dataclass
class Segment:
    name: str
    kind: str  # "constant", "contraction", "nozzle", "convergent", "diffuser"
    x0: float
    x1: float

    @property
    def length(self):
        return self.x1 - self.x0


@dataclass
class TunnelGeometry:
    x: np.ndarray
    r: np.ndarray
    segments: list[Segment]
    r_settling: float
    r_throat: float
    r_test: float
    r_second_throat: float
    r_exit: float
    extra: dict = field(default_factory=dict)

    def radius(self, x):
        return np.interp(x, self.x, self.r)

    def area(self, x):
        return math.pi * self.radius(x) ** 2

    def segment(self, name) -> Segment:
        for s in self.segments:
            if s.name == name:
                return s
        raise KeyError(name)

    def segment_at(self, x) -> Segment:
        for s in self.segments:
            if s.x0 <= x <= s.x1:
                return s
        return self.segments[-1]

    @property
    def length(self):
        return float(self.x[-1] - self.x[0])

    def to_csv(self, path: str | Path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x [m]", "r [m]", "area [m^2]", "component"])
            for xi, ri in zip(self.x, self.r):
                w.writerow([f"{xi:.6e}", f"{ri:.6e}", f"{math.pi * ri * ri:.6e}", self.segment_at(xi).name])


def smooth_step(xi):
    """Quintic smooth step with zero 1st and 2nd derivatives at both ends."""
    xi = np.clip(xi, 0.0, 1.0)
    return 10 * xi ** 3 - 15 * xi ** 4 + 6 * xi ** 5


def build_geometry(r_settling, r_throat, r_test, r_second_throat, nozzle: MOCNozzle, *, settling_length,
                   contraction_length, test_length, convergent_half_angle, second_throat_length,
                   diffuser_half_angle, diffuser_area_ratio, n_per_segment=120) -> TunnelGeometry:
    xs, rs, segs = [], [], []
    x = 0.0

    def add(name, kind, xx, rr):
        nonlocal x
        segs.append(Segment(name, kind, float(xx[0]), float(xx[-1])))
        if xs:
            xx, rr = xx[1:], rr[1:]
        xs.append(np.asarray(xx, dtype=float))
        rs.append(np.asarray(rr, dtype=float))
        x = float(segs[-1].x1)

    # Settling chamber
    xx = np.linspace(x, x + settling_length, n_per_segment // 3)
    add("settling chamber", "constant", xx, np.full_like(xx, r_settling))
    # Contraction (quintic smooth step)
    xi = np.linspace(0.0, 1.0, n_per_segment)
    add("contraction", "contraction", x + contraction_length * xi,
        r_settling - (r_settling - r_throat) * smooth_step(xi))
    # Supersonic nozzle (MOC wall, throat at local x = 0)
    xw = nozzle.x_wall.copy()
    rw = nozzle.r_wall.copy()
    # Force exact continuity with the specified throat and test-section radii
    rw = r_throat + (rw - rw[0]) * (r_test - r_throat) / (rw[-1] - rw[0])
    add("nozzle", "nozzle", x + xw, rw)
    # Test section
    xx = np.linspace(x, x + test_length, n_per_segment // 2)
    add("test section", "constant", xx, np.full_like(xx, r_test))
    # Second-throat convergent
    Lc = (r_test - r_second_throat) / math.tan(convergent_half_angle)
    xx = np.linspace(x, x + Lc, n_per_segment // 2)
    add("second-throat convergent", "convergent", xx, np.linspace(r_test, r_second_throat, xx.size))
    # Second throat
    xx = np.linspace(x, x + second_throat_length, n_per_segment // 2)
    add("second throat", "constant", xx, np.full_like(xx, r_second_throat))
    # Subsonic diffuser
    r_exit = r_second_throat * math.sqrt(diffuser_area_ratio)
    Ld = (r_exit - r_second_throat) / math.tan(diffuser_half_angle)
    xx = np.linspace(x, x + Ld, n_per_segment)
    add("subsonic diffuser", "diffuser", xx, np.linspace(r_second_throat, r_exit, xx.size))

    return TunnelGeometry(np.concatenate(xs), np.concatenate(rs), segs, r_settling, r_throat, r_test,
                          r_second_throat, r_exit)


def with_second_throat(geom: TunnelGeometry, r_d2: float) -> TunnelGeometry:
    """Same tunnel with a different second-throat radius and unchanged
    x-layout (variable-geometry second throat: the walls of the convergent,
    of the throat and of the diffuser move, their lengths do not)."""
    import copy

    cv = geom.segment("second-throat convergent")
    st = geom.segment("second throat")
    df = geom.segment("subsonic diffuser")
    r = geom.r.copy()
    x = geom.x
    m = (x > cv.x0) & (x <= cv.x1)
    r[m] = geom.r_test + (r_d2 - geom.r_test) * (x[m] - cv.x0) / cv.length
    m = (x > st.x0) & (x <= st.x1)
    r[m] = r_d2
    m = (x > df.x0) & (x <= df.x1)
    r[m] = r_d2 + (geom.r_exit - r_d2) * (x[m] - df.x0) / df.length
    out = copy.copy(geom)
    out.r = r
    out.r_second_throat = r_d2
    out.segments = list(geom.segments)
    out.extra = dict(geom.extra)
    return out

