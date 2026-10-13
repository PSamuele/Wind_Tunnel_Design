"""Sizing pipeline: from a TunnelSpec to a complete TunnelDesign."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import isentropic as isen
from . import losses as ls
from . import shocks as sh
from .config import TunnelSpec
from .gas import Gas
from .geometry import TunnelGeometry, build_geometry
from .moc import MOCNozzle, design_nozzle
from .steady import AxialSolution, QuasiOneD, TunnelFlow


@dataclass
class TunnelDesign:
    spec: TunnelSpec
    gas: Gas
    nozzle: MOCNozzle
    geometry: TunnelGeometry
    flow: TunnelFlow
    # sizing
    A_throat: float
    A_test: float
    A_second_throat_min: float
    A_second_throat: float
    M_second_throat: float  # Mach at the second-throat entrance, running condition
    mdot_per_p0: float  # kg/s per Pa of supply pressure (at design T0)
    # operating ratios
    PR_start: float
    PR_run: float
    x_shock_run: float
    sol_start: AxialSolution
    sol_run_min: AxialSolution
    sol_operating: AxialSolution
    p0_start_set: float
    p0_run_set: float
    x_shock_operating: float | None
    # checks
    test_static: dict
    rhombus_length: float
    max_model_area: float
    drag: "ModelDrag"
    liquefaction: dict
    T0_min_no_liquefaction: float
    loss_table: list
    warnings: list = field(default_factory=list)

    @property
    def model_shock(self):
        return self.drag.shock

    def summary(self) -> dict:
        """Key results grouped by topic; keys carry the units."""
        s = self.spec
        g = self.geometry
        p_a = s.design.p_ambient
        ms = self.model_shock
        out = {
            "Geometry": {
                "Throat diameter [mm]": 2000 * g.r_throat,
                "Settling-chamber diameter [mm]": 2000 * g.r_settling,
                "Test-section diameter [mm]": 2000 * g.r_test,
                "Second-throat diameter [mm]": 2000 * g.r_second_throat,
                "Diffuser exit diameter [mm]": 2000 * g.r_exit,
                "Contraction ratio A_sc/A_t [-]": (g.r_settling / g.r_throat) ** 2,
                "Area ratio A_ts/A_t [-]": self.A_test / self.A_throat,
                "Nozzle length [m]": self.nozzle.length,
                "Nozzle cone half-angle [deg]": math.degrees(self.nozzle.omega),
                "Nozzle max wall angle [deg]": math.degrees(self.nozzle.max_wall_angle),
                "Nozzle exit area error [%]": 100 * self.nozzle.area_ratio_error,
                "Total length [m]": g.length,
            },
            "Test section (running set point)": {
                "Mach [-]": s.design.mach,
                "Static pressure [kPa]": self.test_static["p"] / 1e3,
                "Static temperature [K]": self.test_static["T"],
                "Velocity [m/s]": self.test_static["V"],
                "Dynamic pressure [kPa]": self.test_static["q"] / 1e3,
                "Unit Reynolds number [1e6/m]": self.test_static["Re_unit"] / 1e6,
                "Test rhombus length [m]": self.rhombus_length,
            },
            "Operating pressures": {
                "Starting ratio PR_start = p0/p_a [-]": self.PR_start,
                "Minimum running ratio PR_run = p0/p_a [-]": self.PR_run,
                "Starting set point p0 [bar]": self.p0_start_set / 1e5,
                "Running set point p0 [bar]": self.p0_run_set / 1e5,
                "Mass flow at start [kg/s]": self.mdot_per_p0 * self.p0_start_set,
                "Mass flow running [kg/s]": self.mdot_per_p0 * self.p0_run_set,
                "Ambient pressure [kPa]": p_a / 1e3,
            },
            "Second throat (fixed geometry)": {
                "Starting limit A_d2,start/A_t [-]": self.A_second_throat_min / self.A_throat,
                "Design A_d2/A_t [-]": self.A_second_throat / self.A_throat,
                "Mach at second-throat entrance, running [-]": self.M_second_throat,
                "Convergent model": s.diffuser.convergent_model,
            },
            "Model": {
                "Frontal area [cm^2]": s.model.frontal_area * 1e4,
                "Blockage A_m/A_ts [%]": 100 * s.model.frontal_area / self.A_test,
                "Kantrowitz limit for the frontal area [cm^2]": self.max_model_area * 1e4,
                "Drag coefficient, supersonic [-]": self.drag.cd,
                "Drag coefficient, subsonic [-]": self.drag.cd_subsonic,
                "Drag coefficient source": self.drag.source,
                "Equivalent friction length Cd A_m/A_ts [-]": self.drag.fLD,
                "Shock": (f"attached, angle {math.degrees(ms.beta):.2f} deg" if ms is not None and ms.attached
                          else "detached"),
            },
            "Liquefaction": {
                "Saturation ratio N2 [-]": self.liquefaction["N2"],
                "Saturation ratio O2 [-]": self.liquefaction["O2"],
                "Minimum T0 without saturation [K]": self.T0_min_no_liquefaction,
            },
            "Warnings": list(self.warnings),
        }
        return out


def _test_section_state(spec: TunnelSpec, gas: Gas, p0):
    M, T0 = spec.design.mach, spec.design.T0
    g = gas.gamma
    p, T, rho, V = isen.static_state(M, p0, T0, g, gas.R)
    return {"p": float(p), "T": float(T), "rho": float(rho), "V": float(V), "q": float(0.5 * rho * V * V),
            "Re_unit": float(gas.unit_reynolds(p, T, M))}


def _build(spec: TunnelSpec, nozzle: MOCNozzle, r_t, r_ts, r_d2) -> TunnelGeometry:
    g = spec.gas.gamma
    r_sc = r_t * math.sqrt(float(isen.area_ratio(spec.settling.mach, g)))
    return build_geometry(
        r_sc, r_t, r_ts, r_d2, nozzle,
        settling_length=spec.settling.length_factor * 2 * r_sc,
        contraction_length=spec.contraction.length_factor * 2 * r_sc,
        test_length=spec.test_section.length_factor * 2 * r_ts,
        convergent_half_angle=math.radians(spec.diffuser.convergent_half_angle_deg),
        second_throat_length=spec.diffuser.second_throat_length_factor * 2 * r_d2,
        diffuser_half_angle=math.radians(spec.diffuser.subsonic_half_angle_deg),
        diffuser_area_ratio=spec.diffuser.area_ratio,
    )


@dataclass
class ModelDrag:
    cd: float  # supersonic, on frontal area
    cd_subsonic: float
    source: str
    fLD: float  # equivalent Fanno length C_D A_m / A_ts
    fLD_subsonic: float
    shock: object  # ConeShock / ObliqueShock or None


def model_drag(spec: TunnelSpec, g=1.4) -> ModelDrag:
    """Drag coefficient of the model on its frontal area.

    Automatic value (model.cd = 0): forebody pressure drag from the
    attached conical (Taylor-Maccoll) or wedge shock, plus base drag with the
    rough rule Cp_base = -1/M^2 (about 70 % of the vacuum limit -2/(g M^2));
    skin friction is neglected. With a detached shock the forebody pressure
    is taken as the pitot pressure (upper bound).
    """
    m = spec.model
    M = spec.design.mach
    A_ts = math.pi * (0.5 * spec.design.test_section_diameter) ** 2
    ang = math.radians(m.half_angle_deg)
    shock = sh.cone_shock(M, ang, g) if m.shape == "cone" else sh.oblique_shock(M, ang, True, g)
    if m.frontal_area <= 0.0:
        return ModelDrag(0.0, 0.0, "no model", 0.0, 0.0, shock)
    q_over_p = 0.5 * g * M * M
    if m.cd > 0.0:
        cd, source = m.cd, "input"
    else:
        if shock.attached:
            p_surf = shock.pc_p1 if m.shape == "cone" else shock.p2_p1
            fore = (p_surf - 1.0) / q_over_p
            source = f"automatic: {m.shape} shock theory + base pressure -1/M^2"
        else:
            p_pitot = sh.normal_shock_p0_ratio(M, g) / isen.p_p0(M, g)
            fore = (float(p_pitot) - 1.0) / q_over_p
            source = "automatic: detached shock, pitot pressure + base pressure -1/M^2"
        cd = fore + 1.0 / (M * M)
    cd_sub = cd if m.cd_subsonic < 0.0 else m.cd_subsonic
    return ModelDrag(cd, cd_sub, source, cd * m.frontal_area / A_ts, cd_sub * m.frontal_area / A_ts, shock)


def _solver(spec, geom, gas, friction=True, drag: ModelDrag | None = None, convergent_model=None):
    """Quasi-1D solver for a geometry. Convergent and diffuser angles and
    the diffuser area ratio are read from the geometry, so that a modified
    second throat (variable geometry) keeps its real wall angles."""
    ts = geom.segment("test section")
    x_model = ts.x0 + spec.model.position * ts.length
    fld = drag.fLD if drag is not None else 0.0
    fld_sub = drag.fLD_subsonic if drag is not None else 0.0
    cv = geom.segment("second-throat convergent")
    df = geom.segment("subsonic diffuser")
    conv_angle = math.atan2(geom.r_test - geom.r_second_throat, max(cv.length, 1e-12))
    diff_angle = math.atan2(geom.r_exit - geom.r_second_throat, max(df.length, 1e-12))
    return QuasiOneD(geom, gas, spec.settling.K_honeycomb + spec.settling.K_screens,
                     max(diff_angle, 1e-6), (geom.r_exit / geom.r_second_throat) ** 2,
                     spec.diffuser.K_override, spec.test_section.wall_roughness, friction,
                     model_station=x_model, model_fLD=fld, model_fLD_subsonic=fld_sub,
                     convergent_model=convergent_model or spec.diffuser.convergent_model,
                     convergent_half_angle=max(conv_angle, 0.0))


def start_critical_area(flow: TunnelFlow, p0_guess, n=9):
    """Smallest second-throat area that passes the flow behind a normal
    shock standing anywhere between the nozzle exit and the end of the test
    section (model drag and friction included). With friction the critical
    position can be the test-section end."""
    g = flow.s.geom
    x_ne = g.segment("nozzle").x1
    x_te = g.segment("test section").x1
    return max(flow.critical_area_after_shock(xs, p0_guess) for xs in np.linspace(x_ne + 1e-9, x_te - 1e-9, n))


def design_tunnel(spec: TunnelSpec, nozzle: MOCNozzle | None = None, fast: bool = False) -> TunnelDesign:
    """Run the full steady sizing.

    nozzle: reuse a nozzle already designed for the same Mach and throat
    (used by the optimiser). fast: skip the cone/wedge model analysis.
    """
    spec.validate()
    gas = Gas(spec.gas.gamma, spec.gas.R)
    g = gas.gamma
    M = spec.design.mach
    T0 = spec.design.T0
    p_a = spec.design.p_ambient
    warnings = []

    r_ts = 0.5 * spec.design.test_section_diameter
    A_ts = math.pi * r_ts ** 2
    AR = float(isen.area_ratio(M, g))
    A_t = A_ts / AR
    r_t = math.sqrt(A_t / math.pi)

    if nozzle is None:
        n = spec.nozzle
        omega = math.radians(n.inflection_angle_deg) if n.inflection_angle_deg > 0 else None
        nozzle = design_nozzle(M, r_t, omega=omega, rc_ratio=n.throat_curvature_ratio, n_axis=n.n_axis,
                               nu_fraction=n.nu_fraction, length_factor=n.length_factor, g=g)
    warnings += [f"nozzle: {w}" for w in nozzle.warnings]

    drag = model_drag(spec, g)
    # First pass with a provisional second throat to find A_d2,min
    geom0 = _build(spec, nozzle, r_t, r_ts, 0.9 * r_ts)
    flow0 = TunnelFlow(_solver(spec, geom0, gas, drag=drag), T0, p_a)
    p0_guess = 3.0 * p_a / sh.normal_shock_p0_ratio(M, g)
    A_d2_min = start_critical_area(flow0, p0_guess)
    A_d2 = A_d2_min * (1.0 + spec.diffuser.second_throat_margin)
    if A_d2 > A_ts:
        warnings.append("second throat larger than the test section: no second throat possible, "
                        "the convergent becomes a divergent")
        A_d2 = A_ts * 0.999
    r_d2 = math.sqrt(A_d2 / math.pi)

    geom = _build(spec, nozzle, r_t, r_ts, r_d2)
    solver = _solver(spec, geom, gas, drag=drag)
    flow = TunnelFlow(solver, T0, p_a)

    PR_start, x_start, sol_start = flow.starting()
    PR_run, x_run, sol_run = flow.running()
    if not np.isfinite(PR_start):
        raise RuntimeError("no starting solution found (check second throat and friction)")
    if not np.isfinite(PR_run):
        raise RuntimeError("no running solution found")
    mach_at = float(isen.mach_from_area_ratio(A_d2 / A_t, True, g))
    # Mach at second throat in running conditions (with friction losses)
    i_d2 = np.searchsorted(sol_run.x, geom.segment("second throat").x0)
    if i_d2 < sol_run.M.size:
        mach_at = float(sol_run.M[i_d2])

    p0_start_set = PR_start * p_a * (1.0 + spec.run.start_margin)
    if spec.run.p0_schedule == "reduce_after_start":
        p0_run_set = PR_run * p_a * (1.0 + spec.run.run_margin)
    else:
        p0_run_set = p0_start_set
    regime, x_op, sol_op = flow.operating_point(p0_run_set, started=True)
    if sol_op is None:
        warnings.append("operating point at the running set point not resolved")
        sol_op, x_op = sol_run, x_run

    mdot_per_p0 = float(isen.choked_mass_flow(1.0, T0, A_t, g, gas.R))
    test = _test_section_state(spec, gas, p0_run_set)
    mu = math.asin(1.0 / M)
    rhombus = 2.0 * r_ts / math.tan(mu)

    # Kantrowitz: the flow behind a normal shock at M must pass the free area
    M2 = float(sh.normal_shock_M2(M, g))
    A_free_min = A_ts / float(isen.area_ratio(M2, g))
    max_model = A_ts - A_free_min
    if spec.model.frontal_area > max_model:
        warnings.append(f"model blockage {spec.model.frontal_area:.4g} m^2 exceeds the Kantrowitz limit "
                        f"{max_model:.4g} m^2: the tunnel will not start")
    if not drag.shock.attached:
        warnings.append(f"detached shock on the model ({spec.model.shape}, {spec.model.half_angle_deg} deg)")

    liq = ls.liquefaction_margin(test["p"], test["T"])
    liq_start = ls.liquefaction_margin(_test_section_state(spec, gas, p0_start_set)["p"], test["T"])
    liq_all = {k: max(liq[k], liq_start[k]) for k in liq}
    T0_min = ls.minimum_T0_no_liquefaction(M, p0_start_set, g)
    if max(liq_all.values()) >= 1.0:
        warnings.append(f"liquefaction risk in the test section (saturation ratio "
                        f"{max(liq_all.values()):.2f}); minimum T0 without saturation = {T0_min:.1f} K")

    if p0_start_set * (1.0 + spec.tank.regulator_margin) > spec.tank.pressure:
        warnings.append("tank pressure too low to start the tunnel with the requested margins")
    if nozzle.area_ratio_error and abs(nozzle.area_ratio_error) > 5e-3:
        warnings.append("nozzle exit area deviates more than 0.5 % from the isentropic value")

    loss_table = _loss_table(sol_op)
    return TunnelDesign(spec, gas, nozzle, geom, flow, A_t, A_ts, A_d2_min, A_d2, mach_at, mdot_per_p0,
                        PR_start, PR_run, x_run, sol_start, sol_run, sol_op, p0_start_set, p0_run_set, x_op,
                        test, rhombus, max_model, drag, liq_all, T0_min, loss_table, warnings)


def _loss_table(sol: AxialSolution):
    """Total-pressure losses per component, in the order met by the flow.
    Percentages refer to the regulator outlet total pressure (upstream of
    honeycomb and screens); their sum is the total loss to the atmosphere."""
    rows = []
    p_ref = sol.p0_upstream

    def row(name, p_in, p_out):
        rows.append({"component": name, "loss [kPa]": (p_in - p_out) / 1e3,
                     "loss [% of regulator p0]": 100 * (p_in - p_out) / p_ref,
                     "p0 after [kPa]": p_out / 1e3, "p0 after / regulator p0 [-]": p_out / p_ref})

    for name, p_in, p_out in sol.events:
        if p_in <= 0:
            continue
        if abs(p_in - p_out) < 1e-9 * p_ref and "shock" not in name:
            continue
        row(name, p_in, p_out)
    # Exhaust: the exit dynamic pressure is lost to the atmosphere
    row("exhaust (exit dynamic pressure)", sol.exit_p0, sol.exit_p)
    return rows
