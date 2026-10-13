"""Figures. Every figure is a matplotlib Figure saved as SVG (text kept as
text) and reused for the PDF report.

Style rules: categorical colours in a fixed order, a single-hue sequential
ramp for magnitudes, one y-scale per axes (different quantities go into
separate panels), thin lines, recessive grid, a legend whenever a panel has
two or more series.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Reference palette (light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e2e1dc"
SURFACE = "#fcfcfb"
BAND = "#f1f0ec"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
SEQ_LINES = LinearSegmentedColormap.from_list("seq_blue_lines", SEQ[4:])  # visible on the light surface

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def _segments(ax, geom, labels=True, y=1.0):
    """Light alternating bands marking the tunnel components."""
    short = {"settling chamber": "settling", "contraction": "contraction", "nozzle": "nozzle",
             "test section": "test sect.", "second-throat convergent": "conv.", "second throat": "2nd throat",
             "subsonic diffuser": "diffuser"}
    for k, s in enumerate(geom.segments):
        if k % 2 == 1:
            ax.axvspan(s.x0, s.x1, color=BAND, zorder=0, lw=0)
        if labels:
            ax.text(0.5 * (s.x0 + s.x1), y, short.get(s.name, s.name), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=MUTED, rotation=0, clip_on=False)


def _segment_rows(ax, geom):
    """Horizontal boundaries of the components on an axis whose y is x [m];
    labels at the centre of each band, omitted for very short components."""
    L = geom.length
    for s in geom.segments:
        ax.axhline(s.x1, color=GRID, lw=0.8, zorder=0)
        if s.length > 0.06 * L:
            ax.text(1.0, 0.5 * (s.x0 + s.x1), " " + s.name, transform=ax.get_yaxis_transform(), fontsize=7,
                    color=MUTED, va="center", clip_on=False)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def fig_geometry(design):
    g = design.geometry
    fig, ax = plt.subplots(figsize=(10, 3.4))
    _segments(ax, g)
    ax.fill_between(g.x, -g.r, g.r, color=SEQ[1], lw=0, zorder=1)
    ax.plot(g.x, g.r, color=INK, lw=1.4, zorder=2)
    ax.plot(g.x, -g.r, color=INK, lw=1.4, zorder=2)
    ax.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (6, 3, 1, 3)), zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("r [m]")
    ax.set_title("Tunnel meridian profile (axisymmetric, dash-dot = axis)", pad=16)
    fig.tight_layout()
    return fig


def fig_geometry_3d(design):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    g = design.geometry
    sol = design.sol_operating
    xs = np.linspace(g.x[0], g.x[-1], 220)
    th = np.linspace(0.0, 1.5 * np.pi, 60)  # cut-away of 90 deg to show the inside
    X, TH = np.meshgrid(xs, th)
    Rr = g.radius(X)
    Y, Z = Rr * np.cos(TH), Rr * np.sin(TH)
    Mx = np.interp(xs, sol.x, sol.M)
    norm = plt.Normalize(0.0, max(Mx.max(), 1.0))
    colors = SEQ_CMAP(norm(np.broadcast_to(Mx, X.shape)))
    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=2, cstride=2, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((g.length, 2 * g.r_settling, 2 * g.r_settling))
    ax.set_xlabel("x [m]")
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=22, azim=-62)
    ax.set_title("Revolved geometry coloured by Mach number (running condition, 90 deg cut-away)")
    sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=norm)
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label="Mach")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Steady axial distributions
# --------------------------------------------------------------------------
def _axial(ax, design, key, label, scale=1.0):
    sols = [("starting (critical shock position)", design.sol_start, C[1]),
            ("running (operating set point)", design.sol_operating, C[0])]
    for name, sol, col in sols:
        y = getattr(sol, key) if key != "p_over_p0" else sol.p_over_p0
        ax.plot(sol.x, np.asarray(y) * scale, color=col, label=name)
    ax.set_ylabel(label)


def fig_axial_mach(design):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _segments(ax, design.geometry)
    _axial(ax, design, "M", "Mach number")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("x [m]")
    ax.set_title("Mach number along the axis", pad=16)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_axial_distributions(design):
    fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    spec = [("M", "Mach [-]", 1.0), ("p_over_p0", "p / p0 supply [-]", 1.0), ("p", "p [kPa]", 1e-3),
            ("T", "T [K]", 1.0), ("V", "V [m/s]", 1.0)]
    for k, (ax, (key, lab, sc)) in enumerate(zip(axs, spec)):
        _segments(ax, design.geometry, labels=(k == 0))
        _axial(ax, design, key, lab, sc)
    axs[0].legend(loc="upper left")
    axs[0].set_title("Quasi-1D axial distributions: starting vs running", pad=16)
    # total pressure on the pressure-ratio panel (same unit)
    for name, sol, col in [("p0 local / p0 supply, running", design.sol_operating, C[2])]:
        axs[1].plot(sol.x, sol.p0 / sol.p0_supply, color=col, ls="--", label=name)
    axs[1].legend(loc="lower left")
    axs[-1].set_xlabel("x [m]")
    fig.tight_layout()
    return fig


def fig_shock_hysteresis(design, sweep):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    g = design.geometry
    _segment_rows(ax, g)
    for key, col, lab, mk in [("up", C[1], "p0 increasing (start-up)", "o"),
                              ("down", C[0], "p0 decreasing (shut-down)", "s")]:
        pr = np.array([r[0] for r in sweep[key]])
        xs = np.array([np.nan if r[2] is None else r[2] for r in sweep[key]])
        ax.plot(pr, xs, color=col, marker=mk, ms=4, label=lab)
    ax.axvline(design.PR_start, color=C[1], lw=1.0, ls="--")
    ax.axvline(design.PR_run, color=C[0], lw=1.0, ls="--")
    ax.text(design.PR_start, 1.01, f"PR_start {design.PR_start:.3f}", transform=ax.get_xaxis_transform(),
            fontsize=7, color=INK2, ha="left")
    ax.text(design.PR_run, 1.06, f"PR_run {design.PR_run:.3f}", transform=ax.get_xaxis_transform(), fontsize=7,
            color=INK2, ha="right")
    ax.set_xlabel("p0 / p_ambient [-]")
    ax.set_ylabel("normal-shock position x [m]")
    ax.set_title("Steady shock position: start/unstart hysteresis", pad=18)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_shock_family(design, sweep):
    fig, ax = plt.subplots(figsize=(10, 4.0))
    _segments(ax, design.geometry)
    rows = [r for r in sweep["up"] if r[3] is not None and r[3].ok]
    rows = rows[:: max(1, len(rows) // 7)]
    prs = np.array([r[0] for r in rows])
    norm = plt.Normalize(prs.min(), prs.max())
    for pr, regime, xs, sol in rows:
        ax.plot(sol.x, sol.M, color=SEQ_LINES(norm(pr)), lw=1.2)
    sm = plt.cm.ScalarMappable(cmap=SEQ_LINES, norm=norm)
    fig.colorbar(sm, ax=ax, pad=0.01, label="p0 / p_ambient")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Mach [-]")
    ax.set_title("Start-up sequence: Mach distributions for increasing supply pressure", pad=16)
    fig.tight_layout()
    return fig


def fig_loss_budget(design):
    rows = design.loss_table
    names = [r["component"] for r in rows]
    vals = np.array([r["loss [% of regulator p0]"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=C[0], height=0.6, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.2f} %", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlabel("total-pressure loss [% of regulator outlet p0]")
    ax.set_title(f"Loss budget at the running set point (p0/p_a = {design.p0_run_set / design.spec.design.p_ambient:.3f})")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, vals.max() * 1.18)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# MOC
# --------------------------------------------------------------------------
def fig_moc(design):
    n = design.nozzle
    fig, ax = plt.subplots(figsize=(10, 4.2))
    X, R, M = n.lattice_x, n.lattice_r, n.lattice_M
    ok = np.isfinite(X)
    xw = np.interp(X[ok], n.x_wall, n.r_wall)
    inside = R[ok] <= xw * 1.0005
    xp, rp, mp = X[ok][inside], R[ok][inside], M[ok][inside]
    # source-flow kernel between the throat and the first lattice characteristic
    from .moc import source_flow_mach
    ax_ = n.axis
    first = np.isfinite(X[:, 0])
    cx, cr = X[first, 0], R[first, 0]
    order = np.argsort(cr)
    cx, cr = cx[order], cr[order]
    kx, kr, km = [], [], []
    for xx in np.linspace(n.x_wall[0], cx.max(), 40):
        rw = np.interp(xx, n.x_wall, n.r_wall)
        for rr in np.linspace(0.0, rw, 12):
            if rr <= cr.max() and xx > np.interp(rr, cr, cx):
                continue
            kx.append(xx)
            kr.append(rr)
            km.append(max(1.0, source_flow_mach(xx + n.x_shift, rr, ax_.R_star)
                          if np.hypot(xx + n.x_shift, rr) > ax_.R_star else 1.0))
    xp, rp, mp = np.concatenate([xp, kx]), np.concatenate([rp, kr]), np.concatenate([mp, km])
    levels = np.linspace(mp.min(), mp.max() * (1 + 1e-6), 15)
    tc = ax.tricontourf(xp, rp, mp, levels=levels, cmap=SEQ_CMAP)
    fig.colorbar(tc, ax=ax, pad=0.01, label="Mach")
    for i in range(0, X.shape[0], max(1, X.shape[0] // 40)):  # C+ lines (rows)
        row = ok[i]
        xx, rr = X[i][row], R[i][row]
        k = rr <= np.interp(xx, n.x_wall, n.r_wall) * 1.0005
        ax.plot(xx[k], rr[k], color=INK, lw=0.25, alpha=0.5)
    for j in range(0, X.shape[1], max(1, X.shape[1] // 40)):  # C- lines (columns)
        col = ok[:, j]
        xx, rr = X[:, j][col], R[:, j][col]
        k = rr <= np.interp(xx, n.x_wall, n.r_wall) * 1.0005
        ax.plot(xx[k], rr[k], color=INK, lw=0.25, alpha=0.5)
    ax.plot(n.x_wall, n.r_wall, color=INK, lw=1.6, label="wall contour")
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xlim(n.x_wall[0], n.x_wall[-1] * 1.02)
    ax.set_ylim(0, n.r_wall.max() * 1.08)
    ax.set_xlabel("x from throat [m]")
    ax.set_ylabel("r [m]")
    ax.set_title(f"Axisymmetric MOC nozzle, M = {n.M_exit:g}, cone {math.degrees(n.omega):.2f} deg, "
                 f"max wall angle {math.degrees(n.max_wall_angle):.2f} deg (characteristics thinned)")
    fig.tight_layout()
    return fig


def fig_nozzle_wall(design):
    n = design.nozzle
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axs[0].plot(n.x_wall, np.degrees(n.theta_wall), color=C[0])
    axs[0].set_ylabel("wall angle [deg]")
    axs[0].set_title("Nozzle wall: flow angle and Mach number")
    axs[1].plot(n.x_wall, n.M_wall, color=C[0], label="wall")
    xa = np.linspace(n.x_shift + 1e-9, n.axis.x_E * 1.1, 300)
    axs[1].plot(xa - n.x_shift, n.axis.mach(xa), color=C[1], label="centreline (prescribed)")
    axs[1].set_ylabel("Mach [-]")
    axs[1].set_xlabel("x from throat [m]")
    axs[1].legend(loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Blowdown (quasi-steady)
# --------------------------------------------------------------------------
def _started_band(ax, b):
    if np.isfinite(b.t_start):
        t1 = b.t_unstart if np.isfinite(b.t_unstart) else b.t[-1]
        ax.axvspan(b.t_start, t1, color=BAND, zorder=0, lw=0)


def fig_blowdown(design, b):
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    ax = axs[0]
    _started_band(ax, b)
    ax.plot(b.t, b.p_tank / 1e5, color=C[0], label="tank pressure")
    ax.plot(b.t, b.p0 / 1e5, color=C[1], label="supply p0 (regulated)")
    ax.axhline(design.PR_run * design.spec.design.p_ambient / 1e5, color=MUTED, lw=0.8, ls="--")
    ax.text(b.t[-1], design.PR_run * design.spec.design.p_ambient / 1e5, " PR_run p_a", fontsize=7, color=MUTED,
            va="bottom", ha="right")
    ax.set_ylabel("pressure [bar]")
    ax.set_title(f"Blowdown run (shaded = started, run time {b.run_time:.2f} s; stop: {b.stop_reason})")
    ax.legend(loc="upper right")
    ax = axs[1]
    _started_band(ax, b)
    ax.plot(b.t, b.T0, color=C[0], label="T0 = tank temperature")
    ax.plot(b.t, b.T_test, color=C[1], label="test-section static T")
    ax.set_ylabel("temperature [K]")
    ax.legend(loc="upper right")
    ax = axs[2]
    _started_band(ax, b)
    ax.plot(b.t, b.mdot, color=C[0])
    ax.set_ylabel("mass flow [kg/s]")
    ax = axs[3]
    _started_band(ax, b)
    ax.plot(b.t, b.Re_unit / 1e6, color=C[0])
    ax.set_ylabel("Re per metre [1e6/m]")
    ax.set_xlabel("time [s]")
    fig.tight_layout()
    return fig


def fig_blowdown_shock(design, b):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    _segment_rows(ax, design.geometry)
    ax.plot(b.shock_t, b.shock_x, color=C[0], marker="o", ms=3)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("shock position x [m]")
    ax.set_title("Quasi-steady normal-shock position during the run")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Transient (Euler)
# --------------------------------------------------------------------------
def fig_transient_xt(design, tr):
    t, x, M = tr.mach_xt()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    pc = ax.pcolormesh(x, t * 1e3, M, cmap=SEQ_CMAP, shading="auto", vmin=0.0, vmax=max(1.0, np.nanmax(M)))
    fig.colorbar(pc, ax=ax, pad=0.01, label="Mach")
    ok = np.isfinite(tr.x_shock)
    ax.plot(tr.x_shock[ok], tr.t[ok] * 1e3, ".", color=C[1], ms=1.5, label="detected normal shock")
    for s in design.geometry.segments:
        ax.axvline(s.x1, color=SURFACE, lw=0.6, alpha=0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("time [ms]")
    ax.set_title("Unsteady quasi-1D Euler: x-t Mach diagram (start, run, unstart)")
    ax.legend(loc="upper right", markerscale=6)
    fig.tight_layout()
    return fig


def fig_transient_history(design, tr, pr_run=None, pr_run_label="steady PR_run"):
    p_a = design.spec.design.p_ambient
    pr_run = design.PR_run if pr_run is None else pr_run
    vg = tr.throat_area_ratio is not None and np.any(np.isfinite(tr.throat_area_ratio))
    fig, axs = plt.subplots(4 if vg else 3, 1, figsize=(9, 9.5 if vg else 7.5), sharex=True)
    ax = axs[0]
    ax.plot(tr.t * 1e3, tr.p0_inlet / p_a, color=C[0], label="supply p0 / p_a")
    ax.axhline(design.PR_start, color=C[1], lw=1.0, ls="--", label=f"steady PR_start = {design.PR_start:.3f}")
    ax.axhline(pr_run, color=C[2], lw=1.0, ls="--", label=f"{pr_run_label} = {pr_run:.3f}")
    ax.set_ylabel("p0 / p_a [-]")
    ax.legend(loc="upper right")
    ax.set_title("Start-up transient: supply schedule, shock position, started flag"
                 + (", second-throat area" if vg else ""))
    ax = axs[1]
    for s in design.geometry.segments:
        ax.axhline(s.x1, color=GRID, lw=0.8, zorder=0)
    ax.plot(tr.t * 1e3, tr.x_shock, ".", color=C[0], ms=2)
    ax.set_ylabel("shock x [m]")
    ax = axs[2]
    ax.plot(tr.t * 1e3, tr.M_test, color=C[0], label="test-section centre Mach")
    ax.fill_between(tr.t * 1e3, 0, tr.started.astype(float) * design.spec.design.mach, color=BAND, step="post",
                    label="started")
    ax.set_ylabel("Mach [-]")
    ax.legend(loc="upper right")
    if vg:
        ax = axs[3]
        ax.plot(tr.t * 1e3, tr.throat_area_ratio, color=C[0])
        ax.set_ylabel("A_d2 / A_t [-]")
    axs[-1].set_xlabel("time [ms]")
    fig.tight_layout()
    return fig


def fig_transient_profiles(design, tr, n=8):
    fig, ax = plt.subplots(figsize=(10, 4))
    _segments(ax, design.geometry)
    idx = np.unique(np.linspace(0, len(tr.snap) - 1, n).astype(int))
    tt = np.array(tr.snap_t)
    norm = plt.Normalize(tt[idx].min(), tt[idx].max())
    for k in idx:
        ax.plot(tr.x_cells, tr.snap[k]["M"], color=SEQ_LINES(norm(tt[k])), lw=1.2)
    sm = plt.cm.ScalarMappable(cmap=SEQ_LINES, norm=norm)
    fig.colorbar(sm, ax=ax, pad=0.01, label="time [s]")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Mach [-]")
    ax.set_title("Euler snapshots of the Mach distribution", pad=16)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Optimisation and Mach sweep
# --------------------------------------------------------------------------
def fig_optimisation(design, opt, design_opt=None):
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    ax = axs[0]
    pr = np.array([h["PR_run"] if h["ok"] else np.nan for h in opt.history])
    it = np.arange(1, pr.size + 1)
    ax.plot(it, pr, "o", color=C[0], ms=3, label="evaluation")
    best = np.fmin.accumulate(np.where(np.isfinite(pr), pr, np.inf))
    ax.plot(it, np.where(np.isfinite(best), best, np.nan), color=C[1], label="best so far")
    ax.axhline(opt.baseline["PR_run"], color=MUTED, ls="--", lw=0.8)
    ax.text(it[-1], opt.baseline["PR_run"], " baseline", fontsize=7, color=MUTED, va="bottom", ha="right")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("PR_run [-]")
    ax.set_title(f"Diffuser optimisation: PR_run {opt.baseline['PR_run']:.4f} -> {opt.optimum['PR_run']:.4f}")
    ax.legend(loc="upper right")
    ax = axs[1]
    g = design.geometry
    ax.plot(g.x, g.r, color=C[0], label="baseline")
    ax.plot(g.x, -g.r, color=C[0])
    if design_opt is not None:
        go = design_opt.geometry
        ax.plot(go.x, go.r, color=C[1], label="optimum")
        ax.plot(go.x, -go.r, color=C[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("r [m]")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_mach_sweep(rows, T0_design=None):
    rows = [r for r in rows if r.get("ok")]
    M = np.array([r["mach"] for r in rows])
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    ax = axs[0, 0]
    ax.plot(M, [r["PR_start"] for r in rows], color=C[1], marker="o", ms=3, label="PR_start")
    ax.plot(M, [r["PR_run"] for r in rows], color=C[0], marker="s", ms=3, label="PR_run")
    ax.set_ylabel("p0 / p_a [-]")
    ax.set_title("Required pressure ratios")
    ax.legend(loc="upper left")
    ax = axs[0, 1]
    ax.plot(M, [r["run_time_s"] for r in rows], color=C[0], marker="o", ms=3)
    ax.set_ylabel("run time [s]")
    ax.set_title("Run time (same tank)")
    ax = axs[1, 0]
    ax.plot(M, [r["mdot_run_kg_s"] for r in rows], color=C[0], marker="o", ms=3)
    ax.set_ylabel("mass flow at running p0 [kg/s]")
    ax.set_xlabel("test-section Mach [-]")
    ax.set_title("Mass flow")
    ax = axs[1, 1]
    ax.plot(M, [r["T0_min_K"] for r in rows], color=C[0], marker="o", ms=3, label="minimum T0 without saturation")
    if T0_design is not None:
        ax.axhline(T0_design, color=C[1], lw=1.2, ls="--", label=f"design T0 = {T0_design:.1f} K")
    ax.set_ylabel("stagnation temperature [K]")
    ax.set_xlabel("test-section Mach [-]")
    ax.set_title("Liquefaction check")
    ax.legend(loc="lower right")
    fig.suptitle("Mach-number trade study", fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Second throat
# --------------------------------------------------------------------------
REGIME_FILL = {"R0": "#e4e3de", "R1": "#efeee9", "R2": "#dbe8f7", "R3": SURFACE}
REGIME_LABEL = {"R0": "R0 tunnel subsonic", "R1": "R1 shock locked upstream", "R2": "R2 variable walls only",
                "R3": "R3 fixed walls start"}


def _regime_bands(ax, st, x_lo, x_hi, labels=True):
    A_t = st.A_t
    edges = [x_lo, st.A_sub / A_t, st.A_run / A_t, st.A_start / A_t, x_hi]
    for k, code in enumerate(["R0", "R1", "R2", "R3"]):
        a, b = max(edges[k], x_lo), min(edges[k + 1], x_hi)
        if b <= a:
            continue
        ax.axvspan(a, b, color=REGIME_FILL[code], lw=0, zorder=0)
        if labels:
            ax.text(0.5 * (a + b), 1.01, REGIME_LABEL[code], transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7, color=INK2)


def fig_second_throat_map(design, st):
    rows = st.sensitivity
    A = np.array([r["A_d2/A_t [-]"] for r in rows])
    x_lo, x_hi = A.min(), A.max()
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 9.5), sharex=True)
    for k, ax in enumerate(axs):
        _regime_bands(ax, st, x_lo, x_hi, labels=(k == 0))
        for val, name in [(1.0, "A_t"), (st.A_design / st.A_t, "design")]:
            ax.axvline(val, color=MUTED, lw=0.8, ls="--")
            if k == 2:
                ax.text(val, 0.02, f" {name}", transform=ax.get_xaxis_transform(), fontsize=7, color=INK2,
                        rotation=90, va="bottom")
        if st.variable and st.variable.get("applicable"):
            v = st.variable["A_closed/A_t [-]"]
            ax.axvline(v, color=C[2], lw=1.0, ls="--")
            if k == 2:
                ax.text(v, 0.02, " closed (variable)", transform=ax.get_xaxis_transform(), fontsize=7,
                        color=INK2, rotation=90, va="bottom")
    ax = axs[0]
    ax.plot(A, [r["PR_start [-]"] for r in rows], color=C[1], marker="o", ms=3, label="PR_start (R2: open first)")
    ax.plot(A, [r["PR_run [-]"] for r in rows], color=C[0], marker="s", ms=3, label="PR_run")
    ax.set_ylabel("p0 / p_a [-]")
    ax.set_title("Second-throat area: operating regimes and pressure ratios", pad=16)
    ax.legend(loc="upper left")
    ax = axs[1]
    ax.plot(A, [r["run time [s]"] for r in rows], color=C[0], marker="o", ms=3)
    ax.set_ylabel("run time [s]")
    ax = axs[2]
    ax.plot(A, [r["throat Mach running [-]"] for r in rows], color=C[0], marker="o", ms=3,
            label="Mach at second throat, running (= running shock Mach)")
    ax.plot(A, [r["locked shock Mach [-]"] for r in rows], color=C[1], marker="s", ms=3,
            label="Mach of the shock locked by a choked second throat (R1)")
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_ylabel("Mach [-]")
    ax.set_xlabel("second-throat area A_d2 / A_t [-]")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_variable_throat_geometry(design, st):
    g = design.geometry
    gc = st.variable["closed-throat geometry"]
    x0 = g.segment("test section").x0
    x1 = g.segment("subsonic diffuser").x1
    sel = (g.x >= x0) & (g.x <= x1)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _segments(ax, g)
    for geom, col, lab in [(g, C[0], f"open (start), A_d2/A_t = {st.variable['A_open/A_t [-]']:.3f}"),
                           (gc, C[1], f"closed (run), A_d2/A_t = {st.variable['A_closed/A_t [-]']:.3f}")]:
        ax.plot(geom.x[sel], geom.r[sel], color=col, label=lab)
        ax.plot(geom.x[sel], -geom.r[sel], color=col)
    ax.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (6, 3, 1, 3)))
    ax.set_xlim(x0, x1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("r [m] (radial scale enlarged)")
    ax.set_title("Variable second throat: walls open for the start, closed for the run", pad=16)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_variable_throat_axial(design, st):
    sol_f = design.sol_operating
    sol_c = st.variable["solution closed"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for k, ax in enumerate(axs):
        _segments(ax, design.geometry, labels=(k == 0))
    axs[0].plot(sol_f.x, sol_f.M, color=C[0], label="fixed throat, running set point")
    axs[0].plot(sol_c.x, sol_c.M, color=C[1], label="closed variable throat, minimum running ratio")
    axs[0].axhline(1.0, color=MUTED, lw=0.8, ls="--")
    axs[0].set_ylabel("Mach [-]")
    axs[0].legend(loc="upper left")
    axs[0].set_title("Running condition: fixed vs closed variable second throat", pad=16)
    axs[1].plot(sol_f.x, sol_f.p0 / sol_f.p0_supply, color=C[0], label="fixed")
    axs[1].plot(sol_c.x, sol_c.p0 / sol_c.p0_supply, color=C[1], label="closed")
    axs[1].set_ylabel("p0 local / p0 supply [-]")
    axs[1].set_xlabel("x [m]")
    axs[1].legend(loc="lower left")
    fig.tight_layout()
    return fig

