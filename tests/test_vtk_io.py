import base64
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest

from windtunnel import vtk_io


def _decode(text, compressed):
    if not compressed:
        raw = base64.b64decode(text)
        n = struct.unpack("<I", raw[:4])[0]
        return np.frombuffer(raw[4:4 + n], dtype=np.float32)
    nb = struct.unpack("<I", base64.b64decode(text[:8])[:4])[0]
    hlen = 4 * (3 + nb)
    hlen_b64 = 4 * ((hlen + 2) // 3)
    header = struct.unpack(f"<{3 + nb}I", base64.b64decode(text[:hlen_b64])[:hlen])
    data = base64.b64decode(text[hlen_b64:])
    out, pos = [], 0
    for c in header[3:]:
        out.append(zlib.decompress(data[pos:pos + c]))
        pos += c
    return np.frombuffer(b"".join(out), dtype=np.float32)


@pytest.mark.parametrize("compress", [False, True])
def test_revolved_grid_roundtrip(tmp_path, compress):
    x = np.linspace(0.0, 1.0, 30)
    R = 0.1 + 0.05 * x
    M = 1.0 + x
    p = tmp_path / "t.vts"
    dims = vtk_io.write_axisymmetric(p, x, R, {"mach": M}, n_r=4, n_theta=8, u=300.0 * M, compress=compress)
    assert dims == (30, 4, 9)
    root = ET.parse(p).getroot()
    assert root.get("type") == "StructuredGrid"
    arrays = {a.get("Name"): a for a in root.iter("DataArray")}
    mach = _decode(arrays["mach"].text, compress)
    pts = _decode(arrays["Points"].text, compress).reshape(-1, 3)
    vel = _decode(arrays["velocity"].text, compress).reshape(-1, 3)
    n = 30 * 4 * 9
    assert mach.size == n and pts.shape == (n, 3) and vel.shape == (n, 3)
    assert np.allclose(mach[:30], M, rtol=1e-6)
    assert np.all(np.isfinite(pts))
    # wall points lie on r = R(x)
    j, k = 3, 2
    idx = np.arange(30) + 30 * (j + 4 * k)
    assert np.allclose(np.hypot(pts[idx, 1], pts[idx, 2]), R, rtol=1e-6)


def test_vtk_library_reads_files(tmp_path):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    x = np.linspace(0.0, 1.0, 20)
    p = tmp_path / "t.vts"
    vtk_io.write_axisymmetric(p, x, 0.1 + 0 * x, {"mach": 1 + x}, n_r=3, n_theta=6, compress=True)
    r = vtk.vtkXMLStructuredGridReader()
    r.SetFileName(str(p))
    r.Update()
    out = r.GetOutput()
    assert out.GetNumberOfPoints() == 20 * 3 * 7
    assert np.allclose(vtk_to_numpy(out.GetPointData().GetArray("mach"))[:20], 1 + x, rtol=1e-6)


def test_pvd(tmp_path):
    vtk_io.write_pvd(tmp_path / "s.pvd", [(0.0, "a.vts"), (0.5, "b.vts")])
    ds = list(ET.parse(tmp_path / "s.pvd").getroot().iter("DataSet"))
    assert [d.get("timestep") for d in ds] == ["0", "0.5"]
