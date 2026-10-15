"""Run folders and readable result files.

Every run writes into results/<simulation name>/. The name is chosen by the
user (letters, digits and underscores; see clean_name). Without a name, the
folder falls back to the run id: a readable key with the main design inputs
plus the first 8 hex digits of the SHA-256 of the canonical, defaults-applied
input. The run id is also stored in summary.txt and manifest.json, so every
run can be traced back to its exact input.

Inside, one numbered folder per topic holds its CSV tables and SVG figures;
README.txt indexes every file, summary.txt lists the key results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import TunnelSpec, canonical_json, spec_to_python

TOPICS = {
    "geometry": ("01_geometry", "tunnel geometry: meridian profile, revolved view, profile for CAD"),
    "steady": ("02_steady_flow", "steady quasi-1D flow at start and in running conditions, loss budget"),
    "shocks": ("03_shock_analysis", "start/unstart hysteresis, shock position vs supply pressure"),
    "nozzle": ("04_nozzle_moc", "supersonic nozzle designed with the method of characteristics"),
    "second_throat": ("05_second_throat", "second throat: two-throat sonic compatibility, regimes, variable "
                                          "geometry, sensitivity, ramp convergent"),
    "euler": ("06_transient_euler", "unsteady quasi-1D Euler: start-up and unstart, fixed and variable throat"),
    "blowdown": ("07_transient_blowdown", "blowdown run: tank, regulator, run time"),
    "optimisation": ("08_optimisation", "optimisation of the second throat and of the subsonic diffuser"),
    "sweep": ("09_mach_sweep", "trade study versus test-section Mach number"),
    "paraview": ("paraview", "ParaView files (.vts grids, .vtp lines, .pvd time series)"),
    "report": ("report", "report.html (figures inline) and report.pdf"),
}


def input_hash(spec: TunnelSpec) -> str:
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def run_id(spec: TunnelSpec) -> str:
    d, t = spec.design, spec.tank
    key = (f"M{d.mach:.2f}_D{d.test_section_diameter * 1000:.0f}mm_T0-{d.T0:.0f}K_pa{d.p_ambient / 1000:.0f}kPa"
           f"_Vt{t.volume:.2f}m3_pt{t.pressure / 1e5:.1f}bar")
    return f"{key}_{input_hash(spec)[:8]}"


NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
NAME_MAX_LENGTH = 100
# Device names that Windows does not accept as folder names.
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                     *(f"LPT{i}" for i in range(1, 10))}


def clean_name(raw: str) -> str:
    """Simulation name -> folder name: spaces become underscores.

    Only letters, digits and underscores are accepted; anything else raises
    ValueError with a message meant for the user.
    """
    name = "_".join(raw.split())
    if not name:
        raise ValueError("the name cannot be empty")
    if len(name) > NAME_MAX_LENGTH:
        raise ValueError(f"the name is too long (at most {NAME_MAX_LENGTH} characters)")
    if not NAME_PATTERN.match(name):
        bad = sorted({c for c in name if not (c.isascii() and (c.isalnum() or c == "_"))})
        raise ValueError(f"only letters, numbers and _ are allowed (not allowed: {' '.join(bad)})")
    if name.upper() in _WINDOWS_RESERVED:
        raise ValueError(f"'{name}' is a reserved name on Windows")
    return name


def split_label(key: str):
    """'Throat diameter [mm]' -> ('Throat diameter', 'mm')."""
    m = re.match(r"^(.*?)\s*\[([^\]]*)\]\s*$", key)
    return (m.group(1), m.group(2)) if m else (key, "")


def fmt_value(v, digits=5):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v != v:
            return "-"
        if v == 0.0:
            return "0"
        a = abs(v)
        if a >= 1e6 or a < 1e-3:
            return f"{v:.{digits - 1}e}"
        return f"{v:.{digits}g}"
    return str(v)


class RunFolder:
    def __init__(self, spec: TunnelSpec, results_root: str | Path = "results", input_file: str | None = None,
                 clean=True, name: str | None = None):
        self.spec = spec
        self.id = run_id(spec)
        self.name = clean_name(name) if name is not None else self.id
        self.root = Path(results_root) / self.name
        if clean and self.root.exists():
            shutil.rmtree(self.root)  # an existing folder with this name is replaced by the new run
        self.root.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, list] = {k: [] for k in TOPICS}
        self.index["."] = []
        self.input_file = input_file
        self.file(".", "inputs_used.py", "all inputs of this run (defaults applied); re-run with "
                                         "python run.py <this file>").write_text(
            spec_to_python(spec, f"Inputs used for simulation {self.name} (run id {self.id})."))

    def dir(self, topic: str) -> Path:
        d = self.root if topic == "." else self.root / TOPICS[topic][0]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def file(self, topic: str, name: str, description: str = "") -> Path:
        p = self.dir(topic) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        rel = str(p.relative_to(self.root))
        lst = self.index[topic]
        if not any(r == rel for r, _ in lst):
            lst.append((rel, description))
        return p

    def write_csv(self, topic: str, name: str, columns: dict | list, description: str = ""):
        """columns: dict of equal-length arrays, or list of row dicts."""
        p = self.file(topic, name, description)
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            if isinstance(columns, dict):
                keys = list(columns)
                w.writerow(keys)
                arrs = [np.asarray(columns[k]) for k in keys]
                for row in zip(*arrs):
                    w.writerow([_csv(v) for v in row])
            else:
                keys = []
                for r in columns:
                    for k in r:
                        if k not in keys:
                            keys.append(k)
                w.writerow(keys)
                for r in columns:
                    w.writerow([_csv(r.get(k, "")) for k in keys])
        return p

    def write_json(self, name: str, data, description: str = ""):
        p = self.file(".", name, description)
        p.write_text(json.dumps(data, indent=2, default=_json_default))
        return p

    def write_text(self, name: str, text: str, description: str = ""):
        p = self.file(".", name, description)
        p.write_text(text)
        return p

    def write_index(self, title: str):
        """README.txt: what is in each folder and file."""
        lines = [title, "=" * len(title), "", f"Simulation: {self.name}", f"Run id: {self.id}"]
        if self.input_file:
            lines.append(f"Input file: {self.input_file}")
        lines += ["", "Start with summary.txt (key results) or report/report.html (everything, with figures).",
                  "CSV columns carry their units in brackets; figures are SVG.", ""]
        for rel, desc in self.index["."] + [("README.txt", "this index")]:
            lines.append(f"  {rel:<28} {desc}")
        for topic, (folder, desc) in TOPICS.items():
            files = self.index[topic]
            if not files:
                continue
            lines += ["", f"{folder}/  -  {desc}"]
            w = max(len(Path(r).name) for r, _ in files)
            for rel, d in files:
                name = str(Path(rel).relative_to(folder))
                lines.append(f"  {name:<{max(w, 24)}}  {d}")
        (self.root / "README.txt").write_text("\n".join(lines) + "\n")

    def write_manifest(self, version: str, extra: dict | None = None):
        files = sorted([r for lst in self.index.values() for r, _ in lst] + ["README.txt", "manifest.json"])
        data = {
            "simulation_name": self.name,
            "run_id": self.id,
            "input_sha256": input_hash(self.spec),
            "input_file": self.input_file,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "package_version": version,
            "files": files,
        }
        if extra:
            data.update(extra)
        (self.root / "manifest.json").write_text(json.dumps(data, indent=2, default=_json_default))


def summary_text(title: str, groups: dict, width: int = 78) -> str:
    """Grouped key results as aligned text: label ....... value unit."""
    lines = [title, "=" * len(title)]
    for group, items in groups.items():
        lines += ["", group.upper(), "-" * len(group)]
        if isinstance(items, list):
            lines += [f"  - {w}" for w in items] if items else ["  (none)"]
            continue
        for key, v in items.items():
            label, unit = split_label(key)
            val = fmt_value(v)
            dots = "." * max(2, width - 6 - len(label) - len(val) - len(unit))
            lines.append(f"  {label} {dots} {val} {unit}".rstrip())
    return "\n".join(lines) + "\n"


def _csv(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.8g}"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return int(bool(v))
    return v


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
