"""Self-contained HTML report (inline SVG figures) and PDF report."""

from __future__ import annotations

import html
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


@dataclass
class Section:
    title: str
    paragraphs: list = field(default_factory=list)
    figures: list = field(default_factory=list)  # (svg path, caption)
    tables: list = field(default_factory=list)  # (caption, list of row dicts)


CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984; --line:#e2e1dc; --band:#f1f0ec;
        --accent:#2a78d6; --warn-bg:#fff4dc; --warn-ink:#6b4a00; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#9a9990; --line:#3a3a37; --band:#242423;
  --accent:#3987e5; --warn-bg:#3a2f12; --warn-ink:#f3d08a; } }
:root[data-theme="dark"] { --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#9a9990; --line:#3a3a37;
  --band:#242423; --accent:#3987e5; --warn-bg:#3a2f12; --warn-ink:#f3d08a; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.7rem; margin: 0 0 4px; }
h2 { font-size: 1.25rem; margin: 40px 0 8px; padding-top: 8px; border-top: 1px solid var(--line); }
.meta { color: var(--ink2); font-size: .85rem; word-break: break-all; }
nav { margin: 16px 0; display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: .9rem; }
nav a { color: var(--accent); text-decoration: none; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 16px 0; }
.kpi { background: var(--band); border-radius: 8px; padding: 10px 12px; }
.kpi .v { font-size: 1.3rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.kpi .l { color: var(--ink2); font-size: .8rem; }
.warn { background: var(--warn-bg); color: var(--warn-ink); border-radius: 8px; padding: 10px 14px; margin: 12px 0; }
figure { margin: 18px 0; }
figure .svgwrap { background: #fcfcfb; border-radius: 8px; padding: 6px; overflow-x: auto; }
figure svg { max-width: 100%; height: auto; display: block; }
figcaption { color: var(--ink2); font-size: .85rem; margin-top: 4px; }
.tablewrap { overflow-x: auto; margin: 10px 0 18px; }
table { border-collapse: collapse; font-size: .85rem; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 4px 10px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink2); font-weight: 600; }
caption { text-align: left; color: var(--ink2); font-size: .85rem; padding-bottom: 4px; }
code { font-size: .85em; }
"""


def _fmt(v):
    if isinstance(v, float):
        if v != v:
            return "–"
        if v == 0.0:
            return "0"
        a = abs(v)
        if a >= 1e5 or a < 1e-3:
            return f"{v:.4e}"
        return f"{v:.5g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "; ".join(_fmt(x) for x in v)
    return str(v)


def _table_html(caption, rows):
    if not rows:
        return ""
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out = [f'<div class="tablewrap"><table><caption>{html.escape(caption)}</caption><thead><tr>']
    out += [f"<th>{html.escape(str(k))}</th>" for k in keys]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(_fmt(r.get(k, '')))}</td>" for k in keys) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _inline_svg(path):
    s = Path(path).read_text()
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>", "", s, flags=re.S)
    return s.strip()


def _slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def write_html(path, title, meta_lines, kpis, warnings, sections: list[Section]):
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width, initial-scale=1'>",
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>",
           f"<h1>{html.escape(title)}</h1>"]
    for m in meta_lines:
        out.append(f"<div class='meta'>{html.escape(m)}</div>")
    out.append("<nav>" + "".join(f"<a href='#{_slug(s.title)}'>{html.escape(s.title)}</a>" for s in sections)
               + "</nav>")
    if kpis:
        out.append("<div class='kpis'>")
        for label, value in kpis:
            out.append(f"<div class='kpi'><div class='v'>{html.escape(value)}</div>"
                       f"<div class='l'>{html.escape(label)}</div></div>")
        out.append("</div>")
    if warnings:
        out.append("<div class='warn'><strong>Warnings</strong><ul>")
        out += [f"<li>{html.escape(w)}</li>" for w in warnings]
        out.append("</ul></div>")
    for s in sections:
        out.append(f"<h2 id='{_slug(s.title)}'>{html.escape(s.title)}</h2>")
        for p in s.paragraphs:
            out.append(f"<p>{html.escape(p)}</p>")
        for cap, rows in s.tables:
            out.append(_table_html(cap, rows))
        for svg, cap in s.figures:
            out.append(f"<figure><div class='svgwrap'>{_inline_svg(svg)}</div>"
                       f"<figcaption>{html.escape(cap)}</figcaption></figure>")
    out.append("</main></body></html>")
    Path(path).write_text("\n".join(out))


def write_pdf(path, title, meta_lines, kpis, warnings, sections: list[Section], figures: dict):
    """figures: svg path -> matplotlib Figure (re-rendered as vector PDF)."""
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.95
        fig.text(0.07, y, title, fontsize=15, fontweight="bold")
        y -= 0.03
        for m in meta_lines:
            for line in textwrap.wrap(m, 100):
                fig.text(0.07, y, line, fontsize=7, color="#52514e")
                y -= 0.016
        y -= 0.01
        for label, value in kpis:
            fig.text(0.07, y, f"{label}:", fontsize=9)
            fig.text(0.62, y, value, fontsize=9, fontweight="bold")
            y -= 0.021
        if warnings:
            y -= 0.01
            fig.text(0.07, y, "Warnings", fontsize=10, fontweight="bold", color="#6b4a00")
            y -= 0.02
            for w in warnings:
                for line in textwrap.wrap(w, 105):
                    fig.text(0.09, y, line, fontsize=7.5, color="#6b4a00")
                    y -= 0.015
        pdf.savefig(fig)
        plt.close(fig)
        for s in sections:
            for cap, rows in s.tables:
                _pdf_table(pdf, f"{s.title}: {cap}", rows)
            for svg, cap in s.figures:
                f = figures.get(str(svg))
                if f is not None:
                    pdf.savefig(f)


def _pdf_table(pdf, caption, rows, per_page=40):
    if not rows:
        return
    if len(rows) == 1 and len(rows[0]) > 4:
        # a single wide record reads better as a two-column list
        rows = [{"quantity": k, "value": v} for k, v in rows[0].items()]
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    for start in range(0, len(rows), per_page):
        chunk = rows[start:start + per_page]
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.04, 0.95, caption, fontsize=11, fontweight="bold")
        ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
        ax.axis("off")
        cells = [[_fmt(r.get(k, "")) for k in keys] for r in chunk]
        t = ax.table(cellText=cells, colLabels=[str(k) for k in keys], loc="upper left", cellLoc="right")
        t.auto_set_font_size(False)
        t.set_fontsize(6.5 if len(keys) > 8 else 8)
        t.auto_set_column_width(list(range(len(keys))))
        pdf.savefig(fig)
        plt.close(fig)
