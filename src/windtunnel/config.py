"""Input specification: parameters with units and descriptions, defaults,
validation, Python/TOML loading and canonical form.

Inputs are normally declared in a Python file (see inputs.py at the
repository root) that builds a TunnelSpec. A TOML file with the same
sections and keys is also accepted. Every parameter has a default, and
unknown keys are rejected so that typos never pass silently.

Each field carries metadata (unit, description) used to write the resolved
inputs back as Python, and for the input table of the report.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


def P(default, unit="-", doc=""):
    """Parameter with unit and description."""
    return field(default=default, metadata={"unit": unit, "doc": doc})


@dataclass
class GasSpec:
    """Working gas (calorically perfect)."""
    gamma: float = P(1.4, "-", "ratio of specific heats")
    R: float = P(287.05, "J/(kg K)", "gas constant")


@dataclass
class DesignSpec:
    """Design point."""
    mach: float = P(2.0, "-", "test-section Mach number")
    test_section_diameter: float = P(0.15, "m", "test-section diameter")
    T0: float = P(288.15, "K", "stagnation temperature (initial tank temperature)")
    p_ambient: float = P(101325.0, "Pa", "exhaust back pressure")
    T_ambient: float = P(288.15, "K", "ambient temperature (initial state of the tunnel)")


@dataclass
class SettlingSpec:
    """Settling chamber with honeycomb and screens."""
    mach: float = P(0.05, "-", "settling-chamber Mach number (sets the contraction ratio)")
    length_factor: float = P(1.0, "-", "length / settling-chamber diameter")
    K_honeycomb: float = P(0.5, "-", "honeycomb loss coefficient, dp0 = K q (typical value, use hardware data)")
    K_screens: float = P(1.5, "-", "sum of the screen loss coefficients (typical value, use hardware data)")


@dataclass
class ContractionSpec:
    """Subsonic contraction."""
    length_factor: float = P(1.0, "-", "length / settling-chamber diameter")


@dataclass
class NozzleSpec:
    """Supersonic nozzle (method of characteristics)."""
    inflection_angle_deg: float = P(0.0, "deg", "cone half-angle of the source-flow kernel, 0 = automatic")
    throat_curvature_ratio: float = P(2.0, "-", "throat wall radius of curvature / throat radius")
    n_axis: int = P(120, "-", "MOC resolution: centreline points")
    length_factor: float = P(1.0, "-", ">= 1 stretches the centreline Mach distribution")
    nu_fraction: float = P(0.75, "-", "nu on the axis at the end of the source region / nu(exit Mach)")


@dataclass
class TestSectionSpec:
    """Test section."""
    length_factor: float = P(1.5, "-", "length / test-section diameter")
    wall_roughness: float = P(0.0, "m", "equivalent sand roughness of the walls")


@dataclass
class ModelSpec:
    """Test model (blockage and drag)."""
    frontal_area: float = P(0.0, "m^2", "model frontal area, 0 = empty test section")
    shape: str = P("cone", "-", "'cone' or 'wedge' (model shock and automatic drag coefficient)")
    half_angle_deg: float = P(10.0, "deg", "cone or wedge half-angle")
    cd: float = P(0.0, "-", "drag coefficient on frontal area in supersonic flow, 0 = automatic "
                            "(shock theory + base pressure)")
    cd_subsonic: float = P(-1.0, "-", "drag coefficient behind the starting shock, < 0 = same as cd")
    position: float = P(0.4, "-", "model station as a fraction of the test-section length")


@dataclass
class DiffuserSpec:
    """Second throat and subsonic diffuser (fixed geometry of the design)."""
    second_throat_margin: float = P(0.10, "-", "fixed second throat: A_d2 = A_d2,start (1 + margin)")
    convergent_half_angle_deg: float = P(4.0, "deg", "half-angle of the second-throat convergent")
    convergent_model: str = P("isentropic", "-", "'isentropic' or 'oblique_shocks' (shock pair at the "
                                                 "convergent entrance, ramp diffuser)")
    second_throat_length_factor: float = P(2.0, "-", "second-throat length / its diameter")
    subsonic_half_angle_deg: float = P(3.0, "deg", "subsonic diffuser half-angle")
    area_ratio: float = P(3.0, "-", "subsonic diffuser exit area / second-throat area")
    K_override: float = P(-1.0, "-", ">= 0 replaces the subsonic diffuser loss correlation")


@dataclass
class SecondThroatSpec:
    """Second-throat studies: sonic compatibility, variable geometry, sensitivity."""
    enabled: bool = P(True, "-", "run the second-throat studies")
    variable_geometry: bool = P(True, "-", "study a variable second throat (open to start, closed to run)")
    run_margin: float = P(0.05, "-", "closed throat: A_d2 = A*_run (1 + run_margin)")
    close_time: float = P(0.05, "s", "Euler transient: closing time of the variable throat")
    n_points: int = P(14, "-", "sensitivity study: number of second-throat areas")
    area_min_factor: float = P(0.9, "-", "sensitivity study: smallest area / A*_run")
    area_max_factor: float = P(1.3, "-", "sensitivity study: largest area / A_d2,start")


@dataclass
class TankSpec:
    """Pressure tank and regulator."""
    volume: float = P(2.0, "m^3", "tank volume")
    pressure: float = P(20.0e5, "Pa", "initial tank pressure")
    polytropic_n: float = P(1.4, "-", "tank expansion exponent: 1.4 adiabatic, 1.0 isothermal")
    regulator_margin: float = P(0.10, "-", "minimum p_tank / p0 - 1 for regulation")


@dataclass
class RunSpec:
    """Operating schedule of a blowdown run."""
    start_margin: float = P(0.10, "-", "starting set point: p0 = PR_start p_a (1 + start_margin)")
    run_margin: float = P(0.05, "-", "running set point: p0 = PR_run p_a (1 + run_margin)")
    p0_schedule: str = P("reduce_after_start", "-", "'reduce_after_start' or 'constant'")
    dt: float = P(0.01, "s", "time step of the blowdown integration")
    t_max: float = P(600.0, "s", "maximum simulated time")


@dataclass
class TransientSpec:
    """Unsteady quasi-1D Euler start-up transient."""
    enabled: bool = P(True, "-", "run the Euler transient")
    n_cells: int = P(400, "-", "number of cells")
    cfl: float = P(0.7, "-", "CFL number")
    valve_time: float = P(0.04, "s", "regulator opening time (linear p0 ramp)")
    hold_time: float = P(0.08, "s", "hold time at each set point")
    unstart_demo: bool = P(True, "-", "final ramp below the running ratio (unstart)")
    ramp_down_time: float = P(0.05, "s", "duration of each p0 reduction")
    n_snapshots: int = P(60, "-", "number of saved snapshots")
    friction: bool = P(True, "-", "wall friction in constant-area ducts")
    limiter: str = P("minmod", "-", "'minmod' or 'van_leer'")


@dataclass
class OptimisationSpec:
    """Optimisation of the second throat and of the subsonic diffuser."""
    enabled: bool = P(True, "-", "run the optimisation")
    max_total_length: float = P(0.0, "m", "maximum tunnel length, 0 = no limit")
    max_subsonic_half_angle_deg: float = P(8.0, "deg", "upper bound of the diffuser half-angle")
    min_second_throat_margin: float = P(0.05, "-", "lower bound of the second-throat margin")
    max_second_throat_margin: float = P(0.40, "-", "upper bound of the second-throat margin")


@dataclass
class SweepSpec:
    """Mach-number trade study."""
    enabled: bool = P(True, "-", "run the Mach trade study")
    mach_min: float = P(1.5, "-", "smallest Mach number")
    mach_max: float = P(4.0, "-", "largest Mach number")
    n: int = P(11, "-", "number of Mach numbers")


@dataclass
class OutputSpec:
    """Output options."""
    vtk_n_theta: int = P(48, "-", "ParaView revolved grids: points around the axis")
    vtk_n_r: int = P(8, "-", "ParaView revolved grids: points from axis to wall")
    vtk_transient_n_theta: int = P(24, "-", "transient series: points around the axis")
    vtk_transient_n_r: int = P(4, "-", "transient series: points from axis to wall")
    vtk_compress: bool = P(True, "-", "zlib compression of the VTK files")
    pdf: bool = P(True, "-", "write the PDF report")


@dataclass
class TunnelSpec:
    gas: GasSpec = field(default_factory=GasSpec)
    design: DesignSpec = field(default_factory=DesignSpec)
    settling: SettlingSpec = field(default_factory=SettlingSpec)
    contraction: ContractionSpec = field(default_factory=ContractionSpec)
    nozzle: NozzleSpec = field(default_factory=NozzleSpec)
    test_section: TestSectionSpec = field(default_factory=TestSectionSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    diffuser: DiffuserSpec = field(default_factory=DiffuserSpec)
    second_throat: SecondThroatSpec = field(default_factory=SecondThroatSpec)
    tank: TankSpec = field(default_factory=TankSpec)
    run: RunSpec = field(default_factory=RunSpec)
    transient: TransientSpec = field(default_factory=TransientSpec)
    optimisation: OptimisationSpec = field(default_factory=OptimisationSpec)
    sweep: SweepSpec = field(default_factory=SweepSpec)
    output: OutputSpec = field(default_factory=OutputSpec)

    def validate(self):
        d = self.design
        checks = [
            (1.2 <= d.mach <= 5.0, "design.mach must be in [1.2, 5]"),
            (d.test_section_diameter > 0, "design.test_section_diameter must be > 0"),
            (0.01 <= self.settling.mach <= 0.3, "settling.mach must be in [0.01, 0.3]"),
            (0.3 <= self.nozzle.nu_fraction <= 0.95, "nozzle.nu_fraction must be in [0.3, 0.95]"),
            (self.diffuser.second_throat_margin >= 0, "diffuser.second_throat_margin must be >= 0"),
            (self.diffuser.area_ratio >= 1, "diffuser.area_ratio must be >= 1"),
            (self.diffuser.convergent_model in ("isentropic", "oblique_shocks"),
             "diffuser.convergent_model must be 'isentropic' or 'oblique_shocks'"),
            (self.model.shape in ("cone", "wedge"), "model.shape must be 'cone' or 'wedge'"),
            (self.model.frontal_area >= 0, "model.frontal_area must be >= 0"),
            (0.0 < self.model.position < 1.0, "model.position must be in (0, 1)"),
            (self.model.cd >= 0, "model.cd must be >= 0 (0 = automatic)"),
            (self.second_throat.run_margin >= 0, "second_throat.run_margin must be >= 0"),
            (self.transient.limiter in ("minmod", "van_leer"), "transient.limiter must be 'minmod' or 'van_leer'"),
            (self.run.p0_schedule in ("constant", "reduce_after_start"),
             "run.p0_schedule must be 'constant' or 'reduce_after_start'"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)
        return self

    def to_dict(self):
        return asdict(self)

    def copy(self) -> "TunnelSpec":
        return copy.deepcopy(self)


SECTIONS = [f.name for f in fields(TunnelSpec)]


def parameter_table(spec: TunnelSpec):
    """Rows (section, name, value, unit, description) for reports."""
    rows = []
    for sec in fields(spec):
        obj = getattr(spec, sec.name)
        for f in fields(obj):
            rows.append({"parameter": f"{sec.name}.{f.name}", "value": getattr(obj, f.name),
                         "unit": f.metadata.get("unit", ""), "description": f.metadata.get("doc", "")})
    return rows


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _apply(obj, data: dict, path: str):
    names = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"unknown parameter '{path}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"'{path}{key}' must be a table")
            _apply(current, value, f"{path}{key}.")
        else:
            if isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, int) and not isinstance(current, bool):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(obj, key, value)


def spec_from_dict(data: dict) -> TunnelSpec:
    spec = TunnelSpec()
    _apply(spec, data, "")
    return spec.validate()


def _normalise(spec: TunnelSpec) -> TunnelSpec:
    """Coerce numeric types (an int written where a float is expected)."""
    return spec_from_dict(spec.to_dict())


def load_spec(path: str | Path) -> TunnelSpec:
    """Load inputs from a Python file (variable SPEC) or a TOML file."""
    path = Path(path)
    if path.suffix == ".py":
        mod_spec = importlib.util.spec_from_file_location(f"_windtunnel_inputs_{abs(hash(path))}", path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
        spec = getattr(module, "SPEC", None)
        if not isinstance(spec, TunnelSpec):
            raise TypeError(f"{path} must define SPEC = TunnelSpec(...)")
        return _normalise(spec)
    with open(path, "rb") as fh:
        return spec_from_dict(tomllib.load(fh))


# --------------------------------------------------------------------------
# Canonical form, hashing, Python and TOML output
# --------------------------------------------------------------------------
def _canon(v):
    if isinstance(v, dict):
        return {k: _canon(v[k]) for k in sorted(v)}
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        if math.isfinite(v):
            return float(f"{v:.12g}")
        return str(v)
    return v


def canonical_json(spec: TunnelSpec) -> str:
    return json.dumps(_canon(spec.to_dict()), sort_keys=True, separators=(",", ":"))


def _py_value(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, int):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    raise TypeError(type(v))


def spec_to_python(spec: TunnelSpec, header: str = "") -> str:
    """Write a TunnelSpec as an editable Python input file (all parameters,
    with units and descriptions)."""
    classes = ["TunnelSpec"] + [type(getattr(spec, f.name)).__name__ for f in fields(spec)]
    imports = ["from windtunnel.config import ("]
    line = "   "
    for c in classes:
        if len(line) + len(c) + 2 > 100:
            imports.append(line.rstrip())
            line = "   "
        line += f" {c},"
    imports += [line, ")"]
    lines = ['"""' + (header or "Wind tunnel inputs (all parameters).") + '"""', ""] + imports + ["",
                                                                                                "SPEC = TunnelSpec("]
    for sec in fields(spec):
        obj = getattr(spec, sec.name)
        cls = type(obj)
        doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        lines.append(f"    # {doc}")
        lines.append(f"    {sec.name}={cls.__name__}(")
        rows = []
        for f in fields(obj):
            val = f"{f.name}={_py_value(getattr(obj, f.name))},"
            rows.append((val, f.metadata.get("unit", ""), f.metadata.get("doc", "")))
        w = max(len(r[0]) for r in rows)
        for val, unit, d in rows:
            lines.append(f"        {val:<{w}}  # [{unit}] {d}")
        lines.append("    ),")
    lines.append(")")
    return "\n".join(lines) + "\n"


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    raise TypeError(type(v))


def spec_to_toml(spec: TunnelSpec) -> str:
    lines = ["# Resolved input (defaults applied)"]
    for section, values in spec.to_dict().items():
        lines.append("")
        lines.append(f"[{section}]")
        for k, v in values.items():
            lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines) + "\n"
