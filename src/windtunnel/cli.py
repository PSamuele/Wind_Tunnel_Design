"""Command-line entry point.

    python run.py [inputs.py] [--results-root results] [--quick]
        [--no-transient] [--no-optimisation] [--no-sweep] [--no-second-throat]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import __version__
from . import paraview as pv
from . import plots
from .blowdown import simulate_blowdown
from .config import load_spec, parameter_table
from .design import design_tunnel
from .euler1d import startup_transient, variable_throat_transient
from .optimize import mach_sweep, optimise, optimised_spec
from .outputs import RunFolder, TOPICS, clean_name, fmt_value, split_label, summary_text
from .report import Section, write_html, write_pdf
from .second_throat import REGIMES, study


def apply_quick(spec):
    """Reduced resolution for smoke tests (results are coarser)."""
    spec.nozzle.n_axis = 60
    spec.transient.n_cells = 120
    spec.transient.n_snapshots = 12
    spec.transient.valve_time = 0.02
    spec.transient.hold_time = 0.03
    spec.transient.ramp_down_time = 0.02
    spec.second_throat.close_time = 0.02
    spec.second_throat.n_points = 8
    spec.sweep.n = 4
    spec.output.vtk_n_theta = 16
    spec.output.vtk_n_r = 4
    spec.output.vtk_transient_n_theta = 8
    spec.output.vtk_transient_n_r = 3
    return spec


def _clean(d: dict):
    """Drop non-tabular entries (geometries, solutions) from a result dict."""
    return {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool, np.floating, np.integer))}


def group_rows(group: dict):
    rows = []
    for key, v in group.items():
        label, unit = split_label(key)
        rows.append({"quantity": label, "value": fmt_value(v), "unit": unit})
    return rows


def run(spec, results_root="results", transient=True, optimisation=True, sweep=True, second_throat=True,
        verbose=True, input_file=None, name=None):
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f} s] {msg}", flush=True)

    rf = RunFolder(spec, results_root, input_file, name=name)
    out = spec.output
    figs = {}  # svg path -> Figure (re-used for the PDF)

    def save(topic, fig, name, description):
        p = rf.file(topic, name, description)
        fig.savefig(p, format="svg")
        figs[str(p)] = fig
        return p

    log(f"results folder: {rf.root}")

    # ---------------------------------------------------------------- design
    design = design_tunnel(spec)
    n = design.nozzle
    log(f"steady design: PR_start = {design.PR_start:.4f}, PR_run = {design.PR_run:.4f}")
    design.geometry.to_csv(rf.file("geometry", "geometry_profile.csv", "wall radius r(x) of the whole tunnel (CAD)"))
    F = {}
    F["geometry"] = save("geometry", plots.fig_geometry(design), "geometry.svg", "meridian profile")
    F["geometry3d"] = save("geometry", plots.fig_geometry_3d(design), "geometry_3d.svg",
                           "revolved geometry coloured by Mach")

    rf.write_csv("steady", "axial_running.csv", design.sol_operating.as_columns(),
                 "axial distributions at the running set point")
    rf.write_csv("steady", "axial_starting.csv", design.sol_start.as_columns(),
                 "axial distributions at the starting condition (critical shock position)")
    rf.write_csv("steady", "axial_running_minimum.csv", design.sol_run_min.as_columns(),
                 "axial distributions at the minimum running ratio")
    rf.write_csv("steady", "losses_running.csv", design.loss_table, "total-pressure loss budget, running set point")
    F["mach"] = save("steady", plots.fig_axial_mach(design), "axial_mach.svg", "Mach along the axis")
    F["axial"] = save("steady", plots.fig_axial_distributions(design), "axial_distributions.svg",
                      "Mach, p/p0, p, T, V along the axis")
    F["loss"] = save("steady", plots.fig_loss_budget(design), "loss_budget.svg", "loss budget")

    # ---------------------------------------------------------------- shocks
    sweep_bp = design.flow.back_pressure_sweep(design.PR_start, design.PR_run, n=16)
    hyst = []
    for branch, lst in sweep_bp.items():
        for pr, regime, xs, sol in lst:
            hyst.append({"branch": "p0 increasing" if branch == "up" else "p0 decreasing", "p0/p_a [-]": pr,
                         "regime": regime, "shock position [m]": np.nan if xs is None else xs,
                         "shock Mach [-]": sol.shocks[0][1] if (sol is not None and sol.shocks) else np.nan})
    rf.write_csv("shocks", "shock_hysteresis.csv", hyst, "steady shock position vs p0/p_a on both branches")
    ms = design.model_shock
    model_rows = [{"model": f"{spec.model.shape} {spec.model.half_angle_deg:g} deg", "attached": ms.attached,
                   "shock angle [deg]": math.degrees(ms.beta) if ms.attached else np.nan,
                   "drag coefficient [-]": design.drag.cd, "drag coefficient source": design.drag.source}]
    rf.write_csv("shocks", "model_shock.csv", model_rows, "shock on the model and drag coefficient")
    F["hyst"] = save("shocks", plots.fig_shock_hysteresis(design, sweep_bp), "shock_hysteresis.svg",
                     "hysteresis map")
    F["family"] = save("shocks", plots.fig_shock_family(design, sweep_bp), "startup_mach_family.svg",
                       "Mach distributions while p0 rises")
    log("shock analysis done")

    # ---------------------------------------------------------------- nozzle
    rf.write_csv("nozzle", "moc_wall.csv", {"x from throat [m]": n.x_wall, "r [m]": n.r_wall,
                                            "wall angle [deg]": np.degrees(n.theta_wall), "wall Mach [-]": n.M_wall},
                 "nozzle wall contour")
    ok = np.isfinite(n.lattice_x)
    rf.write_csv("nozzle", "moc_lattice.csv", {"x from throat [m]": n.lattice_x[ok], "r [m]": n.lattice_r[ok],
                                               "Mach [-]": n.lattice_M[ok],
                                               "flow angle [deg]": np.degrees(n.lattice_theta[ok])},
                 "characteristic net points")
    F["moc"] = save("nozzle", plots.fig_moc(design), "moc_nozzle.svg", "Mach field and characteristics")
    F["wall"] = save("nozzle", plots.fig_nozzle_wall(design), "moc_wall.svg", "wall angle and Mach")

    # ---------------------------------------------------------------- second throat
    st = None
    if second_throat and spec.second_throat.enabled:
        log("second-throat studies ...")
        st = study(design)
        rf.write_csv("second_throat", "critical_areas.csv", st.critical_table(),
                     "A_sub, A_run, A_start and the design area, with their meaning")
        rf.write_csv("second_throat", "sensitivity.csv", st.sensitivity,
                     "regime, pressure ratios, run time vs second-throat area")
        rf.write_csv("second_throat", "ramp_convergent.csv", st.ramp,
                     "isentropic vs oblique-shock convergent, fixed and closed throat")
        F["st_map"] = save("second_throat", plots.fig_second_throat_map(design, st), "second_throat_map.svg",
                           "regimes, pressure ratios, run time, Mach vs throat area")
        if st.variable and st.variable.get("applicable"):
            rf.write_csv("second_throat", "variable_geometry.csv", [_clean(st.variable)],
                         "variable second throat: open/closed areas, ratios, run time")
            rf.write_csv("second_throat", "axial_running_closed_throat.csv",
                         st.variable["solution closed"].as_columns(), "axial distributions, closed variable throat")
            st.variable["closed-throat geometry"].to_csv(rf.file("second_throat", "geometry_closed_throat.csv",
                                                                 "wall profile with the throat closed"))
            F["vg_geom"] = save("second_throat", plots.fig_variable_throat_geometry(design, st),
                                "variable_throat_geometry.svg", "open and closed walls")
            F["vg_axial"] = save("second_throat", plots.fig_variable_throat_axial(design, st),
                                 "variable_throat_axial.svg", "Mach and p0 with fixed and closed throat")
        log(f"second throat: design regime {st.design_regime}, A_run/A_t = {st.A_run / st.A_t:.4f}, "
            f"A_start/A_t = {st.A_start / st.A_t:.4f}")

    # ---------------------------------------------------------------- blowdown
    bd = simulate_blowdown(design)
    rf.write_csv("blowdown", "blowdown.csv", bd.as_columns(), "time history of the run")
    rf.write_csv("blowdown", "blowdown_shock.csv", {"time [s]": bd.shock_t, "shock position [m]": bd.shock_x,
                                                    "regime": bd.shock_regime}, "quasi-steady shock position")
    if np.nanmax(bd.liquefaction_ratio) >= 1.0:
        k = int(np.nanargmax(bd.liquefaction_ratio >= 1.0))
        design.warnings.append(f"liquefaction risk during the run from t = {bd.t[k]:.2f} s "
                               f"(tank cooling, T0 = {bd.T0[k]:.1f} K)")
    F["bd"] = save("blowdown", plots.fig_blowdown(design, bd), "blowdown.svg", "pressures, temperatures, flow, Re")
    if bd.shock_t.size:
        F["bdshock"] = save("blowdown", plots.fig_blowdown_shock(design, bd), "blowdown_shock.svg",
                            "shock position during the run")
    log(f"blowdown: run time {bd.run_time:.2f} s ({bd.stop_reason})")

    # ---------------------------------------------------------------- Euler
    tr = tr_vg = None
    if transient and spec.transient.enabled:
        log(f"Euler transient, fixed throat ({spec.transient.n_cells} cells) ...")
        tr = startup_transient(design)
        _transient_csv(rf, tr, "transient_fixed_throat.csv", "fixed throat: history (p0, shock, Mach, flow)")
        F["xt"] = save("euler", plots.fig_transient_xt(design, tr), "fixed_throat_xt.svg", "x-t Mach diagram")
        F["hist"] = save("euler", plots.fig_transient_history(design, tr), "fixed_throat_history.svg",
                         "schedule, shock position, started flag")
        F["prof"] = save("euler", plots.fig_transient_profiles(design, tr), "fixed_throat_profiles.svg",
                         "Mach snapshots")
        if st is not None and st.variable and st.variable.get("applicable"):
            log("Euler transient, variable throat ...")
            v = st.variable
            tr_vg = variable_throat_transient(design, v["closed-throat geometry"], v["PR_run closed [-]"],
                                              spec.second_throat.close_time)
            _transient_csv(rf, tr_vg, "transient_variable_throat.csv",
                           "variable throat: history (p0, throat area, shock, Mach, flow)")
            F["xt_vg"] = save("euler", plots.fig_transient_xt(design, tr_vg), "variable_throat_xt.svg",
                              "x-t Mach diagram, variable throat")
            F["hist_vg"] = save("euler", plots.fig_transient_history(design, tr_vg, v["PR_run closed [-]"],
                                                                     "steady PR_run, closed throat"),
                                "variable_throat_history.svg", "schedule, throat area, shock position")
        log("Euler transients done")

    # ---------------------------------------------------------------- optimisation
    opt = design_opt = None
    if optimisation and spec.optimisation.enabled:
        log("optimisation ...")
        opt = optimise(design)
        rf.write_csv("optimisation", "variables.csv", opt.table(), "baseline and optimum variables")
        rf.write_csv("optimisation", "history.csv", [
            {**{name: float(v) for name, v in zip(opt.names, h["x"])}, "PR_run [-]": h["PR_run"],
             "PR_start [-]": h["PR_start"], "length [m]": h["length"],
             "start pressure margin [-]": h["start_pressure_margin"], "feasible": bool(h["ok"])}
            for h in opt.history], "every evaluation")
        design_opt = design_tunnel(optimised_spec(design, opt), nozzle=design.nozzle)
        design_opt.geometry.to_csv(rf.file("optimisation", "geometry_optimised.csv", "optimised wall profile"))
        F["opt"] = save("optimisation", plots.fig_optimisation(design, opt, design_opt), "optimisation.svg",
                        "history and geometry")
        log(f"optimisation: PR_run {opt.baseline['PR_run']:.4f} -> {opt.optimum['PR_run']:.4f}")

    # ---------------------------------------------------------------- Mach sweep
    sw_rows = None
    if sweep and spec.sweep.enabled:
        log("Mach sweep ...")
        sw_rows = mach_sweep(spec)
        rf.write_csv("sweep", "mach_sweep.csv", sw_rows, "key figures vs Mach number")
        F["sweep"] = save("sweep", plots.fig_mach_sweep(sw_rows, spec.design.T0), "mach_sweep.svg",
                          "pressure ratios, run time, mass flow, liquefaction")
        log("Mach sweep done")

    # ---------------------------------------------------------------- ParaView
    pv.export_steady(design, rf.file("paraview", "steady/steady_running.vts", "running condition, whole tunnel"),
                     design.sol_operating, out.vtk_n_r, out.vtk_n_theta, out.vtk_compress)
    pv.export_steady(design, rf.file("paraview", "steady/steady_starting.vts", "starting condition"),
                     design.sol_start, out.vtk_n_r, out.vtk_n_theta, out.vtk_compress, use_moc=False)
    if st is not None and st.variable and st.variable.get("applicable"):
        pv.export_steady(design, rf.file("paraview", "steady/steady_running_closed_throat.vts",
                                         "running condition, closed variable throat"),
                         st.variable["solution closed"], out.vtk_n_r, out.vtk_n_theta, out.vtk_compress,
                         geom=st.variable["closed-throat geometry"])
    pv.export_moc(design, rf.file("paraview", "nozzle/moc_nozzle.vts", "nozzle MOC field"),
                  max(out.vtk_n_r, 6), out.vtk_n_theta, out.vtk_compress)
    pv.export_characteristics(design, rf.file("paraview", "nozzle/moc_characteristics.vtp",
                                              "characteristic lines (meridian plane)"), out.vtk_compress)
    for res, sub in [(tr, "transient_fixed_throat"), (tr_vg, "transient_variable_throat")]:
        if res is None:
            continue
        folder = rf.dir("paraview") / sub
        folder.mkdir(parents=True, exist_ok=True)
        pv.export_transient(design, res, folder, "frame", out.vtk_transient_n_r, out.vtk_transient_n_theta,
                            out.vtk_compress)
        rf.file("paraview", f"{sub}/frame.pvd", "time series: open in ParaView and press Play")
    log("ParaView files written")

    # ---------------------------------------------------------------- summary + report
    groups = design.summary()
    warnings = groups.pop("Warnings")
    if st is not None:
        groups["Second throat (studies)"] = {
            "A_sub / A_t (below: tunnel subsonic) [-]": st.A_sub / st.A_t,
            "A_run / A_t (running limit) [-]": st.A_run / st.A_t,
            "A_start / A_t (fixed-wall starting limit) [-]": st.A_start / st.A_t,
            "Design A_d2 / A_t [-]": st.A_design / st.A_t,
            "Design regime": f"{st.design_regime}: {REGIMES[st.design_regime]}",
            "Equal areas A_d2 = A_t": f"{st.equal_area_regime}: {REGIMES[st.equal_area_regime]}",
        }
        if st.variable and st.variable.get("applicable"):
            v = st.variable
            groups["Variable second throat"] = {
                "Open area A_d2/A_t (start) [-]": v["A_open/A_t [-]"],
                "Closed area A_d2/A_t (run) [-]": v["A_closed/A_t [-]"],
                "PR_run fixed [-]": v["PR_run fixed [-]"],
                "PR_run closed [-]": v["PR_run closed [-]"],
                "Mass flow running, closed [kg/s]": v["mass flow running, closed [kg/s]"],
                "Run time fixed [s]": v["run time fixed [s]"],
                "Run time variable [s]": v["run time variable [s]"],
                "Run time gain [%]": v["run time gain [%]"],
            }
    groups["Blowdown run"] = {
        "Run time [s]": bd.run_time, "Stop reason": bd.stop_reason, "Air used [kg]": float(bd.mass_used[-1]),
        "Final tank pressure [bar]": float(bd.p_tank[-1] / 1e5), "Final T0 [K]": float(bd.T0[-1]),
        "Max liquefaction saturation ratio [-]": float(np.nanmax(bd.liquefaction_ratio)),
    }
    for res, name in [(tr, "Euler transient, fixed throat"), (tr_vg, "Euler transient, variable throat")]:
        if res is None:
            continue
        e = res.events
        g = {"p0/p_a at start [-]": e.get("PR_at_start", np.nan),
             "Time of start [ms]": 1e3 * e.get("t_start", np.nan),
             "p0/p_a at unstart [-]": e.get("PR_at_unstart", np.nan),
             "Time of unstart [ms]": 1e3 * e.get("t_unstart", np.nan), "Time steps": res.n_steps}
        if "A_d2/A_t_at_unstart" in e:
            g["Throat area at unstart A_d2/A_t [-]"] = e["A_d2/A_t_at_unstart"]
        groups[name] = g
    if opt is not None:
        groups["Optimisation"] = {"PR_run baseline [-]": opt.baseline["PR_run"],
                                  "PR_run optimum [-]": opt.optimum["PR_run"],
                                  "Length at optimum [m]": opt.optimum["length"],
                                  **{f"{r['variable']}": r["optimum"] for r in opt.table()}}
    if st is not None:
        groups["Second-throat recommendations"] = list(st.solutions)
    groups["Warnings"] = warnings

    title = (f"Supersonic blowdown wind tunnel - M {spec.design.mach:g}, "
             f"D {spec.design.test_section_diameter * 1000:g} mm")
    text = summary_text(title, groups)
    rf.write_text("summary.txt", f"Simulation: {rf.name}\nRun id: {rf.id}\n\n" + text, "key results, readable")
    rf.write_json("summary.json", groups, "key results, machine-readable (same content)")

    sections = _sections(design, bd, tr, tr_vg, opt, sw_rows, st, F, sweep_bp, groups)
    meta = [f"Simulation: {rf.name}", f"Run id: {rf.id}", f"Input file: {input_file or '-'}", f"windtunnel {__version__}"]
    kpis = _kpis(design, bd, st)
    write_html(rf.file("report", "report.html", "full report with figures"), title, meta, kpis, warnings, sections)
    if out.pdf:
        write_pdf(rf.file("report", "report.pdf", "full report, PDF"), title, meta, kpis, warnings, sections, figs)
    for f in figs.values():
        plt.close(f)
    rf.write_index(title)
    rf.write_manifest(__version__, {"elapsed_s": round(time.time() - t0, 1)})
    log("report written")
    return rf, design, groups


def _transient_csv(rf, tr, name, desc):
    cols = {"time [s]": tr.t, "supply p0 [Pa]": tr.p0_inlet, "shock position [m]": tr.x_shock,
            "test-section Mach [-]": tr.M_test, "started [1/0]": tr.started.astype(int),
            "mass flow at throat [kg/s]": tr.mdot_throat, "mass flow at exit [kg/s]": tr.mdot_exit}
    if tr.throat_area_ratio is not None and np.any(np.isfinite(tr.throat_area_ratio)):
        cols["second-throat area A_d2/A_t [-]"] = tr.throat_area_ratio
    rf.write_csv("euler", name, cols, desc)


def _kpis(d, bd, st):
    k = [
        ("test-section Mach", f"{d.spec.design.mach:g}"),
        ("throat diameter", f"{2000 * d.geometry.r_throat:.2f} mm"),
        ("second-throat diameter", f"{2000 * d.geometry.r_second_throat:.2f} mm"),
        ("total length", f"{d.geometry.length:.3f} m"),
        ("starting p0/p_a", f"{d.PR_start:.3f}"),
        ("minimum running p0/p_a", f"{d.PR_run:.3f}"),
        ("mass flow (running)", f"{d.mdot_per_p0 * d.p0_run_set:.3f} kg/s"),
        ("run time", f"{bd.run_time:.2f} s"),
    ]
    if st is not None and st.variable and st.variable.get("applicable"):
        k.append(("run time, variable throat", f"{st.variable['run time variable [s]']:.2f} s"))
        k.append(("PR_run, variable throat", f"{st.variable['PR_run closed [-]']:.3f}"))
    return k


def _sections(d, bd, tr, tr_vg, opt, sw_rows, st, F, sweep_bp, groups):
    s = d.spec
    folder = {k: v[0] for k, v in TOPICS.items()}
    secs = []
    secs.append(Section("Key results", [
        "Pressure ratios are p0/p_ambient; p0 is the nozzle supply total pressure (downstream of honeycomb and "
        "screens). The same groups are in summary.txt.",
    ], [], [(name, group_rows(g)) for name, g in groups.items() if isinstance(g, dict)]))
    secs.append(Section(f"Geometry ({folder['geometry']})", [], [
        (F["geometry"], "Meridian profile; the tunnel is a body of revolution about the dash-dot axis."),
        (F["geometry3d"], "Revolved geometry coloured by the quasi-1D Mach number at the running set point.")]))
    secs.append(Section(f"Steady flow ({folder['steady']})", [
        "Starting condition: the normal shock must sweep from the throat to the end of the test section; the "
        "starting ratio is the highest p0/p_a over these positions. Running condition: supersonic flow through "
        "the second throat, normal shock in the diffuser where the exit static pressure equals p_ambient.",
        "Losses: honeycomb and screens (K q), Fanno friction in constant-area ducts, model drag as a mixed-out "
        "momentum loss (equivalent f L/D = Cd A_m/A_ts), normal shock, subsonic diffuser (Idelchik-type "
        "correlation), exhaust dynamic pressure.",
    ], [(F["mach"], "Mach number along the axis at start and in running conditions."),
        (F["axial"], "Mach, pressure ratios, static pressure, temperature and velocity."),
        (F["loss"], "Total-pressure losses per component at the running set point.")],
        [("Loss budget (running set point)", _fmt_rows(d.loss_table))]))
    hyst_rows = []
    for branch, lst in sweep_bp.items():
        for pr, regime, xs, sol in lst:
            hyst_rows.append({"branch": "p0 up" if branch == "up" else "p0 down", "p0/p_a": fmt_value(pr),
                              "regime": regime, "shock x [m]": fmt_value(np.nan if xs is None else xs)})
    secs.append(Section(f"Shock analysis ({folder['shocks']})", [
        f"Starting ratio {d.PR_start:.4f}, minimum running ratio {d.PR_run:.4f}: between the two the state depends "
        "on the history (hysteresis).",
    ], [(F["hyst"], "Steady shock position versus supply pressure on the start-up and shut-down branches."),
        (F["family"], "Mach distributions along the start-up branch.")], [("Hysteresis map", hyst_rows)]))
    n = d.nozzle
    secs.append(Section(f"Nozzle, method of characteristics ({folder['nozzle']})", [
        f"Source-flow kernel and prescribed centreline Mach distribution. Cone half-angle {np.degrees(n.omega):.2f} "
        f"deg, maximum wall angle {np.degrees(n.max_wall_angle):.2f} deg, length {n.length:.4f} m, exit area "
        f"error {100 * n.area_ratio_error:+.4f} % versus the isentropic A/A*.",
    ], [(F["moc"], "Mach field and characteristic net (thinned) inside the nozzle."),
        (F["wall"], "Wall angle and Mach number; prescribed centreline distribution.")]))
    if st is not None:
        tables = [("Critical areas", _fmt_rows(st.critical_table())),
                  ("Sensitivity to the second-throat area", _fmt_rows(st.sensitivity)),
                  ("Isentropic vs oblique-shock (ramp) convergent", _fmt_rows(st.ramp))]
        if st.variable and st.variable.get("applicable"):
            tables.insert(1, ("Variable second throat", _fmt_rows([_clean(st.variable)])))
        figs = [(F["st_map"], "Regimes R0-R3 versus second-throat area: pressure ratios, run time, Mach.")]
        if "vg_geom" in F:
            figs += [(F["vg_geom"], "Variable second throat: open walls for the start, closed walls for the run."),
                     (F["vg_axial"], "Running condition with the fixed and the closed second throat.")]
        paragraphs = list(st.explanation)
        paragraphs.append("Regimes: " + "; ".join(f"{k} = {v}" for k, v in REGIMES.items()) + ".")
        paragraphs += [f"Recommendation: {x}" for x in st.solutions]
        paragraphs.append("The oblique-shock pair is a planar estimate of the compression at the convergent "
                          "entrance; the quasi-1D Euler solver does not see oblique shocks.")
        secs.append(Section(f"Second throat ({folder['second_throat']})", paragraphs, figs, tables))
    if tr is not None:
        ev_rows = [{"event": k, "value": fmt_value(v)} for k, v in tr.events.items() if not k.startswith("schedule")]
        figs = [(F["xt"], "Fixed throat: x-t diagram of the Mach number with the detected normal shock."),
                (F["hist"], "Fixed throat: supply schedule, shock position and started flag."),
                (F["prof"], "Fixed throat: Mach distribution snapshots.")]
        tables = [("Fixed throat events", ev_rows)]
        if tr_vg is not None:
            figs += [(F["xt_vg"], "Variable throat: x-t diagram of the Mach number."),
                     (F["hist_vg"], "Variable throat: supply schedule, shock position, started flag, throat area.")]
            tables.append(("Variable throat events", [{"event": k, "value": fmt_value(v)}
                                                      for k, v in tr_vg.events.items()
                                                      if not k.startswith("schedule")]))
        secs.append(Section(f"Transient: unsteady quasi-1D Euler ({folder['euler']})", [
            "Finite volumes (HLLC flux, MUSCL with minmod limiter, SSP-RK3). Fixed throat: p0 ramps to the "
            "starting set point, holds, drops to the running set point (still started: hysteresis), then below "
            "the running ratio (unstart). Variable throat: after the start the throat closes while p0 drops to "
            "the closed-throat running set point; the moving walls add the work term -p dA/dt.",
        ], figs, tables))
    bd_figs = [(F["bd"], "Tank and supply pressure, temperatures, mass flow and unit Reynolds number.")]
    if "bdshock" in F:
        bd_figs.append((F["bdshock"], "Quasi-steady shock position during the run."))
    secs.append(Section(f"Transient: blowdown run ({folder['blowdown']})", [
        f"Tank {s.tank.volume:g} m^3 at {s.tank.pressure / 1e5:g} bar, polytropic exponent {s.tank.polytropic_n:g}, "
        f"regulator margin {100 * s.tank.regulator_margin:g} %, schedule '{s.run.p0_schedule}'.",
    ], bd_figs))
    if opt is not None:
        secs.append(Section(f"Optimisation ({folder['optimisation']})", [
            "Minimise the running pressure ratio over second-throat margin and length, subsonic diffuser half-angle "
            "and area ratio; constraints: total length (if set) and tank pressure sufficient to start.",
        ], [(F["opt"], "Optimisation history and baseline vs optimum geometry.")],
            [("Variables", _fmt_rows(opt.table()))]))
    if sw_rows is not None:
        secs.append(Section(f"Mach trade study ({folder['sweep']})", [
            "Same test-section diameter, tank, model, losses and margins; Mach number varied.",
        ], [(F["sweep"], "Pressure ratios, run time, mass flow and liquefaction limit versus Mach.")],
            [("Sweep", _fmt_rows(sw_rows))]))
    secs.append(Section("ParaView files (paraview)", [
        "steady/: revolved grids of the whole tunnel (running, starting, closed variable throat); nozzle/: MOC "
        "field and characteristic lines; transient_fixed_throat/ and transient_variable_throat/: Euler time "
        "series, open frame.pvd and press Play (the walls move in the variable-throat series).",
    ]))
    secs.append(Section("Inputs", ["All parameters of this run (inputs_used.py in the results folder)."], [],
                        [("Parameters", _fmt_rows(parameter_table(s)))]))
    secs.append(Section("Assumptions and limitations", [
        "Calorically perfect air. Quasi-1D flow: no boundary-layer displacement correction of the nozzle contour, "
        "no shock/boundary-layer interaction, no shock train. Friction only in constant-area ducts in the steady "
        "model. Model drag as a mixed-out momentum loss; automatic Cd from shock theory plus the base-pressure "
        "rule Cp_base = -1/M^2, skin friction neglected. Oblique shocks in the convergent: planar shock-pair "
        "estimate. Liquefaction by Clausius-Clapeyron without supersaturation (conservative). The diffuser loss "
        "correlation is incompressible; check it against data for a final design.",
    ]))
    return secs


def _fmt_rows(rows):
    return [{k: fmt_value(v) for k, v in r.items()} for r in rows]


def ask_name(results_root, input_fn=input) -> str:
    """Ask the user for the simulation name, which becomes the results folder name.

    Spaces become underscores; only letters, digits and underscores are accepted.
    If a folder with that name already exists, ask before replacing it.
    """
    while True:
        raw = input_fn("What would you like to name this simulation? ")
        try:
            name = clean_name(raw)
        except ValueError as e:
            print(f"  Invalid name: {e}. Please try again.")
            continue
        if name != raw.strip():
            print(f"  Saved as: {name}")
        if (Path(results_root) / name).exists():
            answer = input_fn(f"  A simulation called '{name}' already exists. Overwrite it? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                continue
        return name


def main(argv=None, default_config="inputs.py"):
    ap = argparse.ArgumentParser(prog="windtunnel", description="Supersonic blowdown wind tunnel sizing tool")
    ap.add_argument("config", nargs="?", default=default_config,
                    help=f"input file, Python (SPEC = TunnelSpec(...)) or TOML (default: {default_config})")
    ap.add_argument("--results-root", default="results", help="root folder of the outputs (default: results)")
    ap.add_argument("--name", help="simulation name, used as the results folder name (letters, numbers and _; "
                                   "spaces become _). Skips the question at start; an existing folder with "
                                   "this name is replaced")
    ap.add_argument("--quick", action="store_true", help="reduced resolution for a fast check")
    ap.add_argument("--no-transient", action="store_true", help="skip the Euler transients")
    ap.add_argument("--no-optimisation", action="store_true", help="skip the optimisation")
    ap.add_argument("--no-sweep", action="store_true", help="skip the Mach sweep")
    ap.add_argument("--no-second-throat", action="store_true", help="skip the second-throat studies")
    a = ap.parse_args(argv)
    spec = load_spec(a.config)
    if a.name is not None:
        try:
            name = clean_name(a.name)
        except ValueError as e:
            ap.error(f"--name: {e}")
    else:
        try:
            name = ask_name(a.results_root)
        except (EOFError, KeyboardInterrupt):
            print("\nNo simulation name given: run cancelled. Use --name NAME to run without the question.")
            return 1
    if a.quick:
        apply_quick(spec)
    rf, design, groups = run(spec, a.results_root, not a.no_transient, not a.no_optimisation, not a.no_sweep,
                             not a.no_second_throat, input_file=str(a.config), name=name)
    print()
    print((rf.root / "summary.txt").read_text())
    print(f"All results: {rf.root}   (index: README.txt, report: report/report.html)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
