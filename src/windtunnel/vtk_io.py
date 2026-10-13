"""ParaView output: VTK XML StructuredGrid (.vts), PolyData (.vtp) and
time-series collections (.pvd), written without the VTK library.

The axisymmetric solution is revolved around the x axis: grid index i runs
along x, j from the axis to the wall (r = eta R(x)), k around the axis.
Quasi-1D quantities are uniform on each cross-section; the velocity vector
gets a radial component u eta dR/dx so that the wall is a stream surface.
"""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

_BLOCK = 1 << 15


def _encode(arr: np.ndarray, binary: bool, compress: bool) -> str:
    arr = np.ascontiguousarray(arr)
    if not binary:
        return " ".join(f"{v:.7g}" for v in arr.ravel())
    raw = arr.tobytes()
    if not compress:
        header = struct.pack("<I", len(raw))
        return base64.b64encode(header + raw).decode("ascii")
    blocks = [raw[i:i + _BLOCK] for i in range(0, len(raw), _BLOCK)] or [b""]
    comp = [zlib.compress(b, 6) for b in blocks]
    last = len(blocks[-1]) if raw else 0
    header = struct.pack(f"<{3 + len(comp)}I", len(blocks), _BLOCK, last, *[len(c) for c in comp])
    return base64.b64encode(header).decode("ascii") + base64.b64encode(b"".join(comp)).decode("ascii")


def _data_array(name, arr, binary, compress, ncomp=1):
    fmt = "binary" if binary else "ascii"
    arr = np.asarray(arr, dtype=np.float32)
    return (f'<DataArray type="Float32" Name="{escape(name)}" NumberOfComponents="{ncomp}" format="{fmt}">'
            f"{_encode(arr, binary, compress)}</DataArray>")


def _header(kind, binary, compress):
    comp = ' compressor="vtkZLibDataCompressor"' if (binary and compress) else ""
    return (f'<?xml version="1.0"?>\n<VTKFile type="{kind}" version="1.0" byte_order="LittleEndian" '
            f'header_type="UInt32"{comp}>\n')


def write_vts(path, points: np.ndarray, dims, point_data: dict, binary=True, compress=True, field_data=None):
    """points: (nx*ny*nz, 3) ordered with i fastest. point_data values are
    (n,) scalars or (n, 3) vectors."""
    nx, ny, nz = dims
    ext = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    out = [_header("StructuredGrid", binary, compress)]
    out.append(f'<StructuredGrid WholeExtent="{ext}">\n')
    if field_data:
        out.append("<FieldData>\n")
        for k, v in field_data.items():
            out.append(f'<DataArray type="Float64" Name="{escape(k)}" NumberOfTuples="1" format="ascii">'
                       f"{float(v):.12g}</DataArray>\n")
        out.append("</FieldData>\n")
    out.append(f'<Piece Extent="{ext}">\n<PointData>\n')
    for name, val in point_data.items():
        val = np.asarray(val)
        ncomp = 1 if val.ndim == 1 else val.shape[1]
        out.append(_data_array(name, val, binary, compress, ncomp) + "\n")
    out.append("</PointData>\n<Points>\n")
    out.append(_data_array("Points", points, binary, compress, 3) + "\n")
    out.append("</Points>\n</Piece>\n</StructuredGrid>\n</VTKFile>\n")
    Path(path).write_text("".join(out))


def write_vtp_lines(path, lines: list, point_data: dict, binary=True, compress=True):
    """Polylines: lines is a list of (n_i, 3) arrays; point_data maps names to
    lists of (n_i,) arrays aligned with lines."""
    pts = np.concatenate(lines, axis=0) if lines else np.zeros((0, 3))
    offsets = np.cumsum([len(l) for l in lines]).astype(np.int32)
    conn = np.arange(pts.shape[0], dtype=np.int32)
    out = [_header("PolyData", binary, compress), "<PolyData>\n"]
    out.append(f'<Piece NumberOfPoints="{pts.shape[0]}" NumberOfVerts="0" NumberOfLines="{len(lines)}" '
               f'NumberOfStrips="0" NumberOfPolys="0">\n<PointData>\n')
    for name, vals in point_data.items():
        out.append(_data_array(name, np.concatenate(vals) if vals else np.zeros(0), binary, compress) + "\n")
    out.append("</PointData>\n<Points>\n")
    out.append(_data_array("Points", pts, binary, compress, 3) + "\n")
    out.append("</Points>\n<Lines>\n")
    fmt = "binary" if binary else "ascii"
    out.append(f'<DataArray type="Int32" Name="connectivity" format="{fmt}">{_encode(conn, binary, compress)}'
               f"</DataArray>\n")
    out.append(f'<DataArray type="Int32" Name="offsets" format="{fmt}">{_encode(offsets, binary, compress)}'
               f"</DataArray>\n")
    out.append("</Lines>\n</Piece>\n</PolyData>\n</VTKFile>\n")
    Path(path).write_text("".join(out))


def write_pvd(path, entries):
    """entries: list of (time, relative file name)."""
    out = ['<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n',
           "<Collection>\n"]
    for t, f in entries:
        out.append(f'<DataSet timestep="{t:.9g}" group="" part="0" file="{escape(str(f))}"/>\n')
    out.append("</Collection>\n</VTKFile>\n")
    Path(path).write_text("".join(out))


# --------------------------------------------------------------------------
# Axisymmetric revolution
# --------------------------------------------------------------------------
def revolve(x, R, fields: dict, n_r=8, n_theta=48, u=None, field2d=None):
    """Build a revolved structured grid.

    x, R: axial stations and wall radius. fields: name -> (nx,) arrays,
    uniform on each section. field2d: optional name -> (nx, n_r) arrays
    overriding the uniform value (e.g. MOC data). u: axial velocity (nx,)
    or (nx, n_r) for the velocity vector.
    Returns points, dims, point_data.
    """
    x = np.asarray(x, dtype=float)
    R = np.asarray(R, dtype=float)
    nx = x.size
    eta = np.linspace(0.0, 1.0, n_r)
    th = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    # index order: i fastest, then j, then k -> arrays shaped (k, j, i)
    TH, ETA, X = np.meshgrid(th, eta, x, indexing="ij")
    RR = ETA * R[None, None, :]
    pts = np.stack([X, RR * np.cos(TH), RR * np.sin(TH)], axis=-1).reshape(-1, 3)
    data = {}
    for name, val in fields.items():
        v = np.broadcast_to(np.asarray(val, dtype=float)[None, None, :], X.shape)
        data[name] = v.reshape(-1)
    for name, val in (field2d or {}).items():
        v = np.asarray(val, dtype=float).T  # (n_r, nx)
        data[name] = np.broadcast_to(v[None, :, :], X.shape).reshape(-1)
    if u is not None:
        u = np.asarray(u, dtype=float)
        U = np.broadcast_to(u.T[None, :, :] if u.ndim == 2 else u[None, None, :], X.shape)
        dRdx = np.gradient(R, x)
        vr = U * ETA * dRdx[None, None, :]
        vel = np.stack([U, vr * np.cos(TH), vr * np.sin(TH)], axis=-1).reshape(-1, 3)
        data["velocity"] = vel
    return pts, (nx, n_r, n_theta + 1), data


def write_axisymmetric(path, x, R, fields, n_r=8, n_theta=48, u=None, field2d=None, binary=True, compress=True,
                       field_data=None):
    pts, dims, data = revolve(x, R, fields, n_r, n_theta, u, field2d)
    write_vts(path, pts, dims, data, binary, compress, field_data)
    return dims
