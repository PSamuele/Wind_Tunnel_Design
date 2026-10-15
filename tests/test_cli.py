import json
from pathlib import Path

import pytest

from windtunnel.cli import apply_quick, ask_name, main, run
from windtunnel.config import TunnelSpec

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_quick(tmp_path):
    spec = apply_quick(TunnelSpec())
    spec.model.frontal_area = 0.0015
    spec.transient.n_cells = 80
    spec.second_throat.n_points = 4
    spec.output.pdf = False
    rf, design, groups = run(spec, tmp_path / "results", transient=True, optimisation=False, sweep=False,
                             verbose=False, name="end_to_end")
    root = rf.root
    assert root == tmp_path / "results" / "end_to_end"
    for f in ("inputs_used.py", "README.txt", "summary.txt", "summary.json", "manifest.json",
              "01_geometry/geometry_profile.csv", "02_steady_flow/axial_running.csv",
              "02_steady_flow/losses_running.csv", "04_nozzle_moc/moc_nozzle.svg",
              "05_second_throat/critical_areas.csv", "05_second_throat/second_throat_map.svg",
              "05_second_throat/variable_geometry.csv", "06_transient_euler/transient_fixed_throat.csv",
              "06_transient_euler/transient_variable_throat.csv", "07_transient_blowdown/blowdown.csv",
              "paraview/steady/steady_running.vts", "paraview/nozzle/moc_characteristics.vtp",
              "paraview/transient_fixed_throat/frame.pvd", "paraview/transient_variable_throat/frame.pvd",
              "report/report.html"):
        assert (root / f).is_file(), f
    manifest = json.loads((root / "manifest.json").read_text())
    assert "report/report.html" in manifest["files"]
    html = (root / "report" / "report.html").read_text()
    assert html.count("<svg") >= 12
    op = groups["Operating pressures"]
    assert op["Starting ratio PR_start = p0/p_a [-]"] > op["Minimum running ratio PR_run = p0/p_a [-]"] > 1.0
    assert "Second throat (studies)" in groups and "Variable second throat" in groups
    assert "SECOND THROAT (STUDIES)" in (root / "summary.txt").read_text()
    assert "Simulation: end_to_end" in (root / "summary.txt").read_text()


def _answers(*values):
    it = iter(values)
    return lambda prompt="": next(it)


def test_ask_name_retries_until_valid(tmp_path):
    assert ask_name(tmp_path, _answers("", "bad/name", "Mach 2 run")) == "Mach_2_run"


def test_ask_name_existing_folder(tmp_path):
    (tmp_path / "old_run").mkdir()
    # refuse to overwrite, then pick another name
    assert ask_name(tmp_path, _answers("old_run", "n", "new_run")) == "new_run"
    # empty answer means no
    assert ask_name(tmp_path, _answers("old_run", "", "other")) == "other"
    # accept to overwrite
    assert ask_name(tmp_path, _answers("old_run", "y")) == "old_run"


def test_main_rejects_bad_name_option(tmp_path):
    with pytest.raises(SystemExit):
        main([str(ROOT / "inputs.py"), "--name", "bad/name", "--results-root", str(tmp_path)])

