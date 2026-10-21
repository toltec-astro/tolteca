"""Render the normalization manuscript and its figure from saved comparisons.

Run from the Python workspace with its development interpreter::

    python tolteca/design/probes/render_drive_fit_normalization_report.py

The numerical comparisons are inputs; this program does not rerun either
workflow or modify their products. Generated files remain under ``run/``.
"""

from __future__ import annotations

# ruff: noqa: INP001, T201 -- Standalone manuscript renderer.
import hashlib
import json
import re
import shutil
from html import escape
from pathlib import Path

import matplotlib as mpl
import numpy as np
from markdown_it import MarkdownIt

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser, math_to_image

ROOT = Path(__file__).resolve().parents[3]
DESIGN = Path(__file__).resolve().parents[1]
OUT = ROOT / "run/drive-fit-normalization-20261021"
SOURCE = DESIGN / "drive-fit-normalization-20261021.md"


def residual_figure():
    """Plot the saved forward calculations without statistical error bars."""
    primary = json.loads((OUT / "comparison/summary.json").read_text())["networks"]
    historical = json.loads((OUT / "calibration/comparison/summary.json").read_text())[
        "networks"
    ]
    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.linewidth": 0.7,
            "svg.hashsalt": "drive-fit-normalization",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), layout="constrained")
        for ax in axes:
            ax.axhspan(0, 0.25, facecolor="#e9edf0", zorder=0)
            ax.axhline(0, color="0.35", linewidth=0.65)
            ax.axhline(0.25, color="0.55", linewidth=0.65, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(direction="in", width=0.7)
            ax.set_ylabel(r"Power residual $\Delta p$ (dB)")

        ax = axes[0]
        rows = primary + historical
        x = np.arange(len(rows))
        previous = [row["old_model_error_db"] for row in rows]
        corrected = [row["new_model_error_db"] for row in rows]
        for index, left, right in zip(x, previous, corrected, strict=True):
            ax.plot([index - 0.08, index + 0.08], [left, right], color="0.6", lw=0.8)
        ax.plot(
            x - 0.08, previous, "o", mfc="white", mec="0.25", ms=5, label="Previous"
        )
        ax.plot(x + 0.08, corrected, "o", color="0.1", ms=5, label="Corrected")
        ax.set_xticks(x, ["0\n2025", "7\n2025", "11\n2025", "0\n2022"])
        ax.set_xlim(-0.45, 3.45)
        ax.set_ylim(-0.3, 2.6)
        ax.set_xlabel("Network / observing year")
        ax.set_title("(a) Fixed total DAC power", loc="left", fontsize=12, pad=12)
        ax.legend(loc="upper right", frameon=False, fontsize=10)

        ax = axes[1]
        x = np.arange(len(primary))
        fixed = [row["new_model_error_db"] for row in primary]
        peak = [row["waveform_error_after_db"] for row in primary]
        for index, left, right in zip(x, fixed, peak, strict=True):
            ax.plot([index - 0.08, index + 0.08], [left, right], color="0.6", lw=0.8)
        ax.plot(x - 0.08, fixed, "o", color="0.1", ms=5, label="Fixed total power")
        ax.plot(
            x + 0.08, peak, "^", mfc="white", mec="0.25", ms=6, label="Waveform peak"
        )
        ax.set_xticks(x, [str(row["network"]) for row in primary])
        ax.set_xlim(-0.45, 2.45)
        ax.set_ylim(-0.04, 0.49)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.set_xlabel("Network (2025 sequence)")
        ax.set_title("(b) Corrected attenuation", loc="left", fontsize=12, pad=12)
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        for suffix in ("svg", "png", "pdf"):
            fig.savefig(
                OUT / f"comparison/normalization-residuals.{suffix}",
                dpi=200,
                metadata={"Creator": "TolTEC drive-fit normalization report"},
            )
        plt.close(fig)


STYLE = """
*{box-sizing:border-box}

body{margin:0;
background:#fff;
color:#171717;
font:18px/1.58 Georgia,'Times New Roman',serif}

main{width:min(100% - 48px,800px);
margin:56px auto 72px}

h1{font-size:31px;
line-height:1.25;
text-align:center;
font-weight:normal;
margin:0 25px 12px}

.dateline{text-align:center;
font-size:15px;
margin-bottom:38px;
color:#444}

h2{font-size:23px;
line-height:1.3;
font-weight:normal;
margin:36px 0 16px}

h3{font-size:19px;
line-height:1.4;
font-weight:normal;
font-style:italic;
margin:27px 0 12px}

p{margin:0 0 14px}
a{color:inherit;
text-decoration-color:#999;
text-underline-offset:3px}

.abstract{margin:0 32px 34px;
font-size:16.5px;
line-height:1.56}

.abstract h2{text-align:center;
font-size:19px;
margin:0 0 13px}

code{font:0.81em/1.45 ui-monospace,Menlo,monospace;
overflow-wrap:anywhere}

.equation{display:flex;
align-items:center;
justify-content:center;
position:relative;
gap:16px;
margin:26px 0;
padding:5px 40px;
min-height:45px}

.equation img{max-width:100%;
height:auto}

.equation-number{position:absolute;
right:0;
font-size:16px}

.math-inline{display:inline-block;
max-width:100%;
object-fit:contain}

.table-caption,figcaption{font-size:15.5px;
line-height:1.5}

.table-caption{margin:24px 0 9px}

table{border-collapse:collapse;
width:100%;
font-size:15px;
line-height:1.5;
margin:0 0 24px;
border-top:1.2px solid #222;
border-bottom:1.2px solid #222;
font-variant-numeric:tabular-nums}

thead{border-bottom:0.7px solid #777}

th{font-weight:normal}
td,th{padding:8px 6px;
text-align:right}

td:first-child,th:first-child{text-align:left}

figure{margin:32px 0}
figure>img{display:block;
width:100%;
height:auto;
margin:0 0 12px}

figcaption{margin:0 4px}
ol{padding-left:25px;
font-size:15.5px}
li{padding-left:5px;
margin-bottom:10px}

.supplement{border-top:1px solid #aaa;
margin-top:32px;
padding-top:16px;
font-size:14px;
line-height:1.8}

.supplement a{white-space:nowrap}
.supplement span{padding:0 7px;
color:#888}

@media(max-width:650px){body{font-size:17px}
main{width:calc(100% - 30px);
margin-top:30px}
h1{font-size:26px;
margin-left:0;
margin-right:0}
.abstract{margin-left:0;
margin-right:0}
.equation{padding-left:0;
padding-right:28px}
table{font-size:12px}
td,th{padding:7px 3px}
}

@media print{@page{size:A4;
margin:20mm}
body{font-size:11pt;
line-height:1.45}
main{width:auto;
margin:0}
h1{font-size:20pt}
h2{font-size:15pt;
break-after:avoid}
h3{font-size:12pt;
break-after:avoid}
.abstract{font-size:10pt}
.equation,figure,table{break-inside:avoid}
p{orphans:3;
widows:3}
.table-caption,figcaption{font-size:9pt}
table{font-size:9pt}
.supplement{display:none}
}

"""


def render():
    """Typeset the Markdown manuscript, with local vector math and figures."""
    residual_figure()
    math_dir = OUT / "manuscript-math"
    math_dir.mkdir(exist_ok=True)
    mpl.rcParams["mathtext.fontset"] = "stix"
    parser = MathTextParser("path")
    equation_count = 0

    def math_image(expression, *, inline):
        expression = " ".join(expression.strip().splitlines())
        key = hashlib.sha256(f"{inline}:{expression}".encode()).hexdigest()[:16]
        path = math_dir / f"{key}.svg"
        prop = FontProperties(family="STIXGeneral", size=13 if inline else 15)
        parsed = parser.parse(f"${expression}$", dpi=72, prop=prop)
        math_to_image(f"${expression}$", path, format="svg", prop=prop, color="#171717")
        width = parsed.width * 4 / 3
        style = f"width:{width:.2f}px;"
        if inline:
            style += (
                f"height:{parsed.height * 4 / 3:.2f}px;"
                f"vertical-align:-{parsed.depth * 4 / 3:.2f}px"
            )
        return (
            f'<img class="{"math-inline" if inline else "math-display"}" '
            f'src="manuscript-math/{path.name}" alt="{escape(expression, quote=True)}" '
            f'style="{style}">'
        )

    def equation(match):
        nonlocal equation_count
        equation_count += 1
        return (
            f'<div class="equation" id="eq-{equation_count}">'
            + math_image(match.group(1), inline=False)
            + f'<span class="equation-number">({equation_count})</span></div>'
        )

    source = SOURCE.read_text()
    text = re.sub(r"\$\$(.*?)\$\$", equation, source, flags=re.DOTALL)
    text = re.sub(
        r"(?<!\$)\$([^$\n]+)\$(?!\$)",
        lambda match: math_image(match.group(1), inline=True),
        text,
    )
    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(text)
    body = body.replace(
        "<h2>Abstract</h2>", '<section class="abstract"><h2>Abstract</h2>'
    )
    body = body.replace(
        "<h2>1. Introduction</h2>", "</section><h2>1. Introduction</h2>"
    )
    body = body.replace("../../run/drive-fit-normalization-20261021/", "")
    supplement = """
<footer class="supplement" aria-label="Supplementary material">
Supplementary material:
<a href="index.html#plots">channel plots</a><span>·</span>
<a href="index.html#configs">workflow configurations</a><span>·</span>
<a href="comparison/summary.json">numerical results</a><span>·</span>
<a href="comparison/normalization-residuals.pdf">Figure 1 (PDF)</a><span>·</span>
<a href="report.md">manuscript source</a>
</footer>"""
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Absolute power normalization in TolTEC drive-fit commit</title>"
        f"<style>{STYLE}</style></head><body><main><article>{body}</article>"
        f"{supplement}</main></body></html>"
    )
    (OUT / "report.html").write_text(html)
    (OUT / "report.md").write_text(source)
    for name in ("drive-fit-power-audit-20261020.md", "drive-fit-workflow-spec.md"):
        shutil.copyfile(DESIGN / name, OUT / name)
    (OUT / "probes").mkdir(exist_ok=True)
    for name in (
        "drive_fit_normalization.py",
        "drive_fit_power_audit.py",
        "render_drive_fit_normalization_report.py",
    ):
        shutil.copyfile(DESIGN / "probes" / name, OUT / "probes" / name)
    print(
        f"Rendered {equation_count} numbered equations and Figure 1: "
        f"{OUT / 'report.html'}"
    )


if __name__ == "__main__":
    render()
