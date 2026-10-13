"""ParaView exports of the steady, MOC and transient solutions."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from . import isentropic as isen
from . import vtk_io


def _steady_fields(sol, x):
    def f(a):
        return np.interp(x, sol.x, a)

    return {
        "mach": f(sol.M), "pressure_Pa": f(sol.p), "temperature_K": f(sol.T), "density_kg_m3": f(sol.rho),
        "total_pressure_Pa": f(sol.p0), "p_over_p0_supply": f(sol.p) / sol.p0_supply,
    }, f(sol.V)


def moc_mach_on_grid(design, x_tunnel, n_r):
    """Mach number from the MOC lattice on (x, eta) points of the nozzle.
    Returns an (nx, n_r) array, NaN outside the nozzle or the lattice."""
    n = design.nozzle
    g = design.geometry
    x0 = g.segment("nozzle").x0
    X, R, M = n.lattice_x, n.lattice_r, n.lattice_M
    ok = np.isfinite(X)
    pts = np.column_stack([np.concatenate([X[ok], n.x_wall]), np.concatenate([R[ok], n.r_wall])])
    vals = np.concatenate([M[ok], n.M_wall])
    interp = LinearNDInterpolator(pts, vals)
    out = np.full((x_tunnel.size, n_r), np.nan)
    eta = np.linspace(0.0, 1.0, n_r)
    sel = (x_tunnel >= x0) & (x_tunnel <= g.segment("nozzle").x1)
    xl = x_tunnel[sel] - x0
    Rl = g.radius(x_tunnel[sel])
    XX, EE = np.meshgrid(xl, eta, indexing="ij")
    out[sel] = interp(XX, EE * Rl[:, None])
    return out


def export_steady(design, path, sol=None, n_r=8, n_theta=48, compress=True, use_moc=True, nx=400, geom=None):
    sol = sol or design.sol_operating
    g = geom or design.geometry
    x = np.unique(np.concatenate([np.linspace(g.x[0], g.x[-1], nx), [s.x1 for s in g.segments]]))
    fields, V = _steady_fields(sol, x)
    field2d = None
    u = V
    if use_moc:
        Mm = moc_mach_on_grid(design, x, n_r)
        gam = design.gas.gamma
        M2 = np.where(np.isfinite(Mm), Mm, fields["mach"][:, None])
        p0 = np.interp(x, sol.x, sol.p0)[:, None]
        T0 = sol.T0
        T2 = T0 * isen.T_T0(M2, gam)
        p2 = p0 * isen.p_p0(M2, gam)
        field2d = {"mach": M2, "pressure_Pa": p2, "temperature_K": T2,
                   "density_kg_m3": p2 / (design.gas.R * T2), "p_over_p0_supply": p2 / sol.p0_supply}
        for k in field2d:
            fields.pop(k, None)
        u = M2 * np.sqrt(gam * design.gas.R * T2)
    return vtk_io.write_axisymmetric(path, x, g.radius(x), fields, n_r, n_theta, u, field2d, True, compress)


def export_moc(design, path, n_r=12, n_theta=48, compress=True, nx=200):
    g = design.geometry
    s = g.segment("nozzle")
    x = np.linspace(s.x0, s.x1, nx)
    Mm = moc_mach_on_grid(design, x, n_r)
    # fill the few points outside the lattice (throat region) with source/quasi-1D values
    q1 = np.interp(x, design.sol_operating.x, design.sol_operating.M)
    M2 = np.where(np.isfinite(Mm), Mm, q1[:, None])
    gam = design.gas.gamma
    T2 = design.spec.design.T0 * isen.T_T0(M2, gam)
    p2 = design.p0_run_set * isen.p_p0(M2, gam)
    u = M2 * np.sqrt(gam * design.gas.R * T2)
    return vtk_io.write_axisymmetric(path, x, g.radius(x), {}, n_r, n_theta, u,
                                     {"mach": M2, "pressure_Pa": p2, "temperature_K": T2,
                                      "p_over_p0_supply": p2 / design.p0_run_set}, True, compress)


def export_characteristics(design, path, compress=True):
    """C+ and C- lines of the MOC lattice in the meridian plane (z = 0),
    clipped at the wall, with Mach number and family (+1 / -1)."""
    n = design.nozzle
    x0 = design.geometry.segment("nozzle").x0
    X, R, M = n.lattice_x, n.lattice_r, n.lattice_M
    ok = np.isfinite(X)
    lines, mach, fam = [], [], []
    # rows: C+ lines (fixed id), columns: C- lines (fixed axis point)
    cands = [(X[i, ok[i]], R[i, ok[i]], M[i, ok[i]], 1.0) for i in range(X.shape[0])]
    cands += [(X[ok[:, j], j], R[ok[:, j], j], M[ok[:, j], j], -1.0) for j in range(X.shape[1])]
    for xx, rr, mm, sign in cands:
        k = rr <= np.interp(xx, n.x_wall, n.r_wall) * 1.0005
        if k.sum() >= 2:
            lines.append(np.column_stack([xx[k] + x0, rr[k], np.zeros(k.sum())]))
            mach.append(mm[k])
            fam.append(np.full(k.sum(), sign))
    vtk_io.write_vtp_lines(path, lines, {"mach": mach, "family": fam}, True, compress)
    return len(lines)


def export_transient(design, tr, folder, prefix="transient", n_r=4, n_theta=24, compress=True):
    g = design.geometry
    R0 = g.radius(tr.x_cells)
    entries = []
    for k, (t, s) in enumerate(zip(tr.snap_t, tr.snap)):
        name = f"{prefix}_{k:04d}.vts"
        R = np.sqrt(s["A"] / np.pi) if "A" in s else R0  # moving walls (variable throat)
        vtk_io.write_axisymmetric(folder / name, tr.x_cells, R,
                                  {"mach": s["M"], "pressure_Pa": s["p"], "temperature_K": s["T"],
                                   "density_kg_m3": s["rho"], "total_pressure_Pa": s["p0"]},
                                  n_r, n_theta, s["u"], None, True, compress, field_data={"TimeValue": t})
        entries.append((t, name))
    vtk_io.write_pvd(folder / f"{prefix}.pvd", entries)
    return [e[1] for e in entries]
