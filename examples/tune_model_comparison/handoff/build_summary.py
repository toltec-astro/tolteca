#!/usr/bin/env python
"""Build the concise scientific handoff PDF from its bundled evidence.

Requires reportlab and Pillow. No raw data or science environment is needed.
"""

# ruff: noqa: E501, T201 -- typeset technical report text
import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
WIDTH = 492
BODY = ParagraphStyle(
    "body", fontName="Times-Roman", fontSize=11, leading=14.5, spaceAfter=8
)
TITLE = ParagraphStyle(
    "title", parent=BODY, fontName="Times-Bold", fontSize=22, leading=25, spaceAfter=13
)
H1 = ParagraphStyle(
    "chapter",
    parent=BODY,
    fontName="Times-Bold",
    fontSize=16,
    leading=19,
    spaceAfter=11,
)
H2 = ParagraphStyle(
    "section",
    parent=BODY,
    fontName="Times-Bold",
    fontSize=12,
    leading=15,
    spaceBefore=8,
    spaceAfter=6,
)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=9.3, leading=12, spaceAfter=8)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=10, leading=12, spaceAfter=0)


def paragraph(text, style=BODY):
    """Typeset one body paragraph or heading."""
    return Paragraph(text, style)


def figure(path, width=WIDTH):
    """Preserve the scientific figure's aspect ratio."""
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=h * width / w)


def table(rows, widths):
    """Typeset a numbered table with restrained horizontal rules."""
    result = Table(
        [[paragraph(str(v), CELL) for v in row] for row in rows],
        colWidths=widths,
        hAlign="LEFT",
    )
    result.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def footer(canvas, doc):
    """Identify the release and page without obscuring the body."""
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setStrokeColor(colors.HexColor("#999999"))
    canvas.line(60, 43, 552, 43)
    canvas.drawString(60, 30, "TolTEC Tune-model comparison | 23 November 2026")
    canvas.drawRightString(552, 30, str(doc.page))
    canvas.restoreState()


def build(output):
    """Keep primary results first and alternatives in explicitly separate appendices."""
    e = HERE / "evidence"
    data = json.loads((e / "summary.json").read_text())
    lock = json.loads((HERE / "revisions.json").read_text())
    s = [
        paragraph("TolTEC Tune-model comparison", TITLE),
        paragraph("Observation 152434 - comparison release for local evaluation", H2),
        paragraph(
            "<b>Summary.</b> The Python V3 model fits the measured sweeps substantially better than the legacy tune.txt model. With recorded joint-component identities included, the map reduction uses nearly the same number of contributors as the legacy reference: <b>4,057 versus 4,070</b>. The best validated V3 maps have lower background fluctuations in a1100 and a2000; a1400 remains slightly worse. This is a comparison release, with the remaining validation limits recorded in Appendix A."
        ),
        paragraph("1. Principal results", H1),
        paragraph(
            "The retained V3 path is Python nonlinear group fitting, ECSV-to-stub transport, native group solving and Citlali map making with corrected gap handling. It returns optical detuning, x<sub>opt</sub>, and uses physical sample acceptance. Joint components are admitted through saved finding/channel/APT correspondences; no new frequency-nearest match is introduced."
        ),
        paragraph(
            "Table 1. Actual map contributors and background fluctuations.", SMALL
        ),
    ]
    rows = [["Array", "Contributors<br/>legacy / V3", "RMS legacy", "RMS V3", "Change"]]
    for i, row in enumerate(data["arrays"]):
        old, new = [
            row["rms_mJy_beam"][k] for k in ("legacy-broader", "joint-physical")
        ]
        counts = [
            data["runs"][k]["arrays"][i]["contributors"]
            for k in ("legacy-broader", "joint-physical")
        ]
        rows.append(
            [
                row["array"],
                f"{counts[0]:,} / {counts[1]:,}",
                f"{old:.2f}",
                f"{new:.2f}",
                f"{100 * (new / old - 1):+.1f}%",
            ]
        )
    s += [
        table(rows, [66, 135, 100, 100, 91]),
        Spacer(1, 8),
        paragraph(
            "RMS is the standard deviation in mJy/beam on the same 952 covered pixels in a 20-40 arcsec annulus. It includes residual astronomical and correlated structure; it is not a calibrated detector sensitivity or mapping-speed measurement.",
            SMALL,
        ),
        paragraph("1.1 What is held fixed", H2),
        paragraph(
            "All three arrays use the same raw integration, original beammap-good gate, telescope/pointing inputs, beam and flux columns, filtering, PCA settings and weighting prescription. Full-integration per-row weights are identical between the broader legacy and both broader V3 runs. No map registration, interpolation between sky grids or signal rescaling is used."
        ),
        paragraph(
            "The reference uses legacy model coefficients in the <i>corrected comparison engine</i>. It is not an untouched production V2 binary. Applying the same gap/filter/PCA corrections to both paths isolates the model comparison from the previously identified gap contamination."
        ),
        paragraph("1.2 Calibration still inherited from V2", H2),
        paragraph(
            "<b>Yes: the APT remains V2-derived.</b> Beam positions, good/bad flags, flux calibration and sensitivity come from the original observation-matched APT. V3 adds model and component bindings; it has not independently recalibrated those optical properties. Producing a beammap APT through this same fitting and solving path is the next scientific validation step."
        ),
        PageBreak(),
        paragraph("2. Sweep-model quality", H1),
        paragraph(
            "The model jointly explains M acquired channels with N resonators and includes nonlinearity for every fit. Singlet/multiplet and nonlinearity bins describe coverage; they do not select different physical models. The comparison uses identical measured frequencies and valid samples in each response window."
        ),
        figure(e / "sweep-quality.png", 420),
        paragraph(
            "Figure 1. Median complex-IQ residual divided by the measured IQ excursion RMS about its mean. This dimensionless normalization allows different channels and group sizes to be compared without signal rescaling. It measures in-sample fit quality, not held-out predictive accuracy.",
            SMALL,
        ),
    ]
    ratios = [data["sweep"][str(i)] for i in range(3)]
    s += [
        paragraph(
            "V3 has the smaller residual for "
            + ", ".join(
                f"{r['v3_smaller_residual']:,}/{r['responses']:,} ({100 * r['v3_smaller_residual'] / r['responses']:.1f}%)"
                for r in ratios
            )
            + " selected responses in a1100, a1400 and a2000, respectively. These response-window denominators differ from acquired APT-row counts.",
            SMALL,
        ),
        figure(e / "joint-sweep.png", 365),
        paragraph(
            "Figure 2. A measured two-channel, two-resonator example (network 0, group 404). Black points are the sweep, grey is legacy tune.txt and purple is the full V3 group model. Both resonances are evaluated together at each acquired probe.",
            SMALL,
        ),
        PageBreak(),
        paragraph("3. Maps from the broader detector population", H1),
        figure(e / "maps.png", 480),
        paragraph(
            "Figure 3. Full-integration legacy and V3 physical-acceptance maps. Each array shares one intensity scale between its two panels; labels count rows with at least one accepted map sample. All filled/rejected samples remain excluded from map accumulation. Table 1 gives the common-annulus RMS.",
            SMALL,
        ),
        PageBreak(),
        paragraph("4. Coverage and reproducible delivery", H1),
        paragraph(
            "The acquired APT contains 5,518 rows: 4,646 originally good and 872 originally bad. The broader selection retains all 3,096 scalar-control rows and adds 1,045 jointly fitted components, reaching <b>4,141 eligible rows (89.1% of the originally good population)</b>. The original 872 bad rows remain excluded."
        ),
        figure(e / "population.png", 475),
        paragraph(
            "Figure 4. Disposition of the originally beammap-good acquired rows. Fit/measurement limits exclude 410 rows, while 95 lack a unique supported component binding. Missing identity evidence is distinct from a failed fit. Scalar-route controls include some multiplet components and should not be labelled singlets.",
            SMALL,
        ),
        paragraph(
            "Of the 1,045 newly admitted components, 1,007 contribute accepted samples. The remaining 38 are explicitly recorded as entirely rejected. Adding the supported joint components reduces V3 physical-acceptance RMS by 15.9%, 16.6% and 17.8% relative to the preceding scalar-route controls."
        ),
        paragraph("4.1 Delivered workflow (Table 2)", H2),
        table(
            [
                ["Stage", "Maintained implementation / recorded output"],
                [
                    "Fit recorded Tune",
                    "tolteca_kids ModelFit, Python backend; both blocks retained with common per-channel ECSV and full group metadata.",
                ],
                [
                    "Bind and materialize",
                    "Raw observation context selects the header-recorded sweep/block; stub artifacts carry the same model. Saved correspondences bind components to the existing APT.",
                ],
                [
                    "Solve and map",
                    "Coordinated kidscpp/Citlali consumes the selected stubs and mixed scalar/component APT. Exact native config, logs, input hashes and reduNN are retained.",
                ],
                [
                    "Inspect",
                    "APT eligibility, component disposition, native sample flags, maps and execution receipts remain available for audit.",
                ],
            ],
            [113, 379],
        ),
        Spacer(1, 10),
        paragraph("4.2 Next calibration product", H2),
        paragraph(
            "Run a beammap through the same Python group fitting, model stubs and component solving; derive and review component identities, good flags, beam positions, flux calibration and sensitivities. Then repeat the science comparison with that self-consistent APT. This release intentionally preserves the existing APT so model effects can be assessed first."
        ),
        PageBreak(),
        paragraph("Appendix A. Controls, alternatives and limits", H1),
        paragraph("A.1 Sample acceptance and recovered coordinates", H2),
        paragraph(
            "Physical and numerical acceptance use the same nonlinear fitted model and return x<sub>opt</sub>. Both reject hard numerical failures. Physical acceptance also rejects finite domain/stability warnings; numerical acceptance retains those finite results. Numerical acceptance does not extend the joint solver's search domain."
        ),
        paragraph(
            "Table 3. Alternative acceptance control, broader population (mJy/beam).",
            SMALL,
        ),
        table(
            [["Array", "Legacy", "V3 numerical", "V3 physical"]]
            + [
                [
                    r["array"],
                    *[
                        f"{r['rms_mJy_beam'][k]:.2f}"
                        for k in ("legacy-broader", "joint-numerical", "joint-physical")
                    ],
                ]
                for r in data["arrays"]
            ],
            [90, 134, 134, 134],
        ),
        Spacer(1, 10),
        paragraph(
            "The apparent detuning x' and optical detuning x<sub>opt</sub> controls remain in the extended diagnostic report. Loss-detuning cross-spectra and model assumptions must accompany any noise interpretation: lower sweep residual alone does not guarantee lower map noise. No instantaneous fitted-nonlinearity proxy is substituted for an independently measured optical response."
        ),
        paragraph("A.2 Gap handling", H2),
        paragraph(
            "Rejected samples are filled for filtering and covariance construction using the maintained gap/despiker-style treatment, while masks propagate through the filter and PCA. Filled samples do not enter maps. Earlier maps with contaminated PCA are diagnostic history, not the principal result here."
        ),
        paragraph("A.3 Scope of the validation", H2),
        paragraph(
            "The ECSV/stub transport preserves the fitted model. This does not imply that all Python and native numerical solutions agree: the full-record 145609 replay still found 14 affine and 16 affine-plus-delay groups disagreeing out of 473 routed groups. Those differences are retained as release limitations, not hidden by relaxed tolerances. The 152434 map result is demonstrated with the coordinated native consumer."
        ),
        paragraph(
            "Joint-state covariance is retained diagnostically, but map accumulation uses per-row weights. Independently calibrated joint-component beam/flux/sensitivity and covariance-aware map weighting are not established by this comparison. The remaining a1400 excess is also unresolved."
        ),
        paragraph(
            "The evidence covers 11 available networks (0-5, 7-9, 11-12), not missing networks 6 and 10. Comparison summaries span 12 saved map chunks and 3,628 saved timestream samples. Raw input and full model products remain unchanged."
        ),
        paragraph(
            "The portable preparation reproduces the completed 152434 APT columns, full joint-model payload and native science configuration. Native release Git trees match the tested source trees byte-for-byte. A fresh Python installation is checked separately from native compilation; a recipient Linux/Spack build is still to be exercised."
        ),
        PageBreak(),
        paragraph("Appendix B. Recipient handoff and provenance", H1),
        paragraph(
            "The GitHub branch <b>codex/grant-tune-comparison-20261120</b> in toltec-astro/tolteca contains the handoff under <b>examples/tune_model_comparison/handoff/</b>. Start with README.md and AGENT_HANDOFF.md. revisions.json pins the coordinated repositories; do not combine this interface with an unrelated Citlali checkout."
        ),
        paragraph("B.1 Run on another observation (Table 4)", H2),
        table(
            [
                ["Input", "Recipient supplies"],
                [
                    "Citlali configuration",
                    "A working one-acquisition config with explicit raw KIDs/telescope files, pointing/flux calibration and an observation-matched APT.",
                ],
                [
                    "Recorded calibration",
                    "Tune or target-sweep files named by acquisition headers, for exactly the selected networks. A Tune of another obsnum can be the recorded calibration.",
                ],
                [
                    "Execution",
                    "bootstrap.py installs the pinned Python sources and prepares native sources. run_observation.py fits, materializes stubs, binds the APT and invokes the coordinated Citlali binary.",
                ],
                [
                    "Controls",
                    "--prepare-only audits inputs without mapping; --model-index reuses an exact fit; --also-legacy runs original legacy coefficients with the same broader eligibility.",
                ],
            ],
            [110, 382],
        ),
        Spacer(1, 9),
        paragraph("B.2 Source lock (Table 5)", H2),
        table(
            [["Repository", "Pinned revision (prefix)"]]
            + [
                [
                    name,
                    row["revision"][:16]
                    if row["revision"] != "self"
                    else "Record the checked-out branch commit",
                ]
                for name, row in lock["repositories"].items()
            ],
            [115, 377],
        ),
        Spacer(1, 10),
        paragraph(
            "The handoff contains the complete source lock, Python dependency constraints, compact figures, aggregate evidence and report builders. Raw acquisitions, APT payloads and large map/timestream products are not embedded in the Git release; the recipient provides their own input files."
        ),
        paragraph(
            "The original matched APT SHA-256 is recorded in evidence/summary.json, together with hashes of the comparison summaries and six displayed FITS maps. Maintained source references: design/spec/kids-model-reports.md; docs/kids-tune-comparison.md; examples/tune_model_comparison/README.md. The longer historical comparison can be regenerated with handoff/build_pdf.py when the local evidence is available.",
            SMALL,
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=(612, 792),
        rightMargin=60,
        leftMargin=60,
        topMargin=48,
        bottomMargin=55,
        title="TolTEC Tune-model comparison - 152434",
        author="TolTEC comparison release",
        pageCompression=1,
    )
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
