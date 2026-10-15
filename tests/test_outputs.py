import json
from pathlib import Path

import pytest

from windtunnel.config import TunnelSpec, load_spec, spec_from_dict, spec_to_python, spec_to_toml
from windtunnel.outputs import RunFolder, clean_name, run_id, split_label, summary_text

ROOT = Path(__file__).resolve().parents[1]


def test_same_input_same_folder():
    assert run_id(TunnelSpec()) == run_id(TunnelSpec())


@pytest.mark.parametrize("section, key, value", [
    ("design", "mach", 2.5), ("tank", "volume", 3.0), ("settling", "K_screens", 1.0),
    ("diffuser", "second_throat_margin", 0.2), ("transient", "n_cells", 300), ("model", "cd", 0.5),
])
def test_any_change_changes_folder(section, key, value):
    a = TunnelSpec()
    b = TunnelSpec()
    setattr(getattr(b, section), key, value)
    assert run_id(a) != run_id(b)


def test_folder_name_contains_problem_data():
    rid = run_id(TunnelSpec())
    assert rid.startswith("M2.00_D150mm_T0-288K_pa101kPa_Vt2.00m3_pt20.0bar_")
    assert len(rid.rsplit("_", 1)[1]) == 8


def test_run_folder_layout(tmp_path):
    rf = RunFolder(TunnelSpec(), tmp_path / "results")
    assert (rf.root / "inputs_used.py").is_file()
    p = rf.write_csv("steady", "a.csv", {"x [m]": [0.0, 1.0]}, "test table")
    assert p.parent.name == "02_steady_flow"
    rf.write_index("Title")
    rf.write_manifest("test")
    text = (rf.root / "README.txt").read_text()
    assert "02_steady_flow/" in text and "a.csv" in text and "test table" in text


def test_python_and_toml_roundtrip(tmp_path):
    s = TunnelSpec()
    s.design.mach = 3.0
    s.model.frontal_area = 0.001
    py = tmp_path / "in.py"
    py.write_text(spec_to_python(s))
    tm = tmp_path / "in.toml"
    tm.write_text(spec_to_toml(s))
    assert run_id(load_spec(py)) == run_id(s) == run_id(load_spec(tm))
    with pytest.raises(KeyError):
        spec_from_dict({"design": {"mach_number": 2.0}})


def test_inputs_files_load():
    for f in ("inputs.py", "examples/mach3.py", "examples/mach2.toml"):
        load_spec(ROOT / f)


def test_summary_text():
    assert split_label("Throat diameter [mm]") == ("Throat diameter", "mm")
    txt = summary_text("T", {"Group": {"Throat diameter [mm]": 115.47}, "Warnings": []})
    assert "Throat diameter" in txt and "115.47 mm" in txt and "(none)" in txt


@pytest.mark.parametrize("raw, name", [
    ("mach2_baseline", "mach2_baseline"), ("Mach 2 test", "Mach_2_test"), ("  run   3  ", "run_3"),
    ("case_01", "case_01"),
])
def test_clean_name(raw, name):
    assert clean_name(raw) == name


@pytest.mark.parametrize("raw", ["", "   ", "mach-2", "a/b", "run.1", "test?", "caffè", "CON", "nul", "x" * 101])
def test_clean_name_rejects(raw):
    with pytest.raises(ValueError):
        clean_name(raw)


def test_named_run_folder(tmp_path):
    rf = RunFolder(TunnelSpec(), tmp_path / "results", name="my run")
    assert rf.root == tmp_path / "results" / "my_run"
    assert rf.id == run_id(TunnelSpec())
    rf.write_manifest("test")
    manifest = json.loads((rf.root / "manifest.json").read_text())
    assert manifest["simulation_name"] == "my_run" and manifest["run_id"] == rf.id

