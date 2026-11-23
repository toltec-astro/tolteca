#!/usr/bin/env python
"""Render the Tune comparison review from saved, hashed evidence.

Requires Pandoc and pdflatex. Does not rerun fitting or reduction.
"""

# ruff: noqa: E501, T201 -- report prose/tables and CLI artifact paths
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ARRAYS = ("a1100", "a1400", "a2000")
NAMES = {"legacy": "Legacy full", "common": "Legacy eligible", "python": "V3 models"}


def table(headers, rows):
    """Render a Pandoc table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines) + "\n\n"


def broader_comparison(result, figure):
    """Lead with complete integration maps including recorded joint components."""
    text = r"""
# 152434: broader maps with jointly fitted components

The same final Tune-block Python model and model stubs now feed joint solving
and map projection. A component follows its **saved ModelFit plan, KidsFind
finding, unique saved channel match and existing observation APT UID**.
No new frequency-nearest-neighbour match or beam position is invented.
The native solver evaluates the complete nonlinear group at its acquired probes.
Both V3 policies return optical detuning from that same nonlinear model.
Numerical and physical refer to sample acceptance: the latter additionally
rejects finite domain/stability warnings. Both reject hard solving failures.
Scalar controls denote the earlier independent-channel route, not only singlets.

The expanded list retains all 3,096 scalar-control rows and adds 1,045 joint
components, reaching **4,141 of 4,646 beammap-good rows (89.1%)**. Eligibility
is 2,480 / 889 / 772 across a1100 / a1400 / a2000. There are still 410 rows
excluded by fitting/measurement limits and 95 without a unique supported
component binding. All 872 originally beammap-bad rows remain excluded.
These are acquired APT row counts, not all fitted resonators. The component
ledger has a different denominator and preserves every unmatched finding.

The existing APT beam, flux and sensitivity columns are unchanged. Only newly
admitted component rows get their actual recorded probe as tone_freq. All
scalar-control probe frequencies and APT sensitivities are preserved. The same population-median outlier rule changes three scalar a1400 weights by about 2%; the audit records the exact change. The broader native APT
has explicit channel rows as well as component bindings; repeated placeholder
UIDs on excluded rows do not become detector identities. [R10]

"""
    text += (
        "**Table 0. Map fluctuations under matched sample policies (mJy/beam).**\n\n"
    )
    text += table(
        [
            "Array",
            "Added",
            "Scalar numerical",
            "Broader numerical",
            "Scalar physical",
            "Broader physical",
        ],
        [
            [
                r["array"],
                r["joint_added"],
                *[f"{r['rms_mJy_beam'][n]:.2f}" for n in result["order"]],
            ]
            for r in result["arrays"]
        ],
    )
    text += r"""
RMS is in mJy/beam. A broader detector population changes the estimated PCA
basis and coverage under the same weighting rule. Existing APT sensitivities stay fixed; the median outlier reset depends on the population.
Both policies reject failed solutions; numerical acceptance retains finite
domain/stability warnings while physical acceptance also rejects those samples.
The checks below retain the original diagnostic flags under either policy.
The numerical projection additionally labels unavailable joint outputs as
nonfinite; physical acceptance already excludes them by their domain flags.
The audit verifies that these samples enter neither map. Numerical acceptance
does not expand the joint solver's search domain.

\newpage

"""
    text += figure(
        "joint-map-summary",
        "Broader versus scalar-only maps at the same sample policy. Eligibility includes all supported additions; actual contributing counts are given on the map panels. RMS uses identical covered 20--40 arcsec pixels.",
    )
    text += "\n\\newpage\n\n## Broader maps and actual contributors\n\n"
    text += "The corrected legacy reference uses the same broader APT eligibility and calibration columns. Its full-integration per-row weights match both broader V3 runs exactly. Legacy channels retain their scalar proximity veto; V3 joint components use group-solving validity. Actual contributors therefore need not match. Legacy responses remain effective per-channel coordinates, not resolved component signals.\n\n"
    text += figure(
        "joint-legacy-maps",
        "Full-integration broader maps for all three arrays. The legacy reference and two V3 policies share an intensity scale within each array. Panel counts show actual contributors. No registration or signal rescaling is applied.",
    )
    text += "\n\\newpage\n\n"
    text += "**Table 0. Broader-population map fluctuations (mJy/beam).**\n\n"
    text += table(
        ["Array", "Legacy", "V3 numerical", "V3 physical"],
        [
            [
                r["array"],
                *[f"{r['rms_mJy_beam'][name]:.2f}" for name in result["broader_order"]],
            ]
            for r in result["arrays"]
        ],
    )
    text += "**Table 0. Actual map contributors in the broader runs.**\n\n"
    text += table(
        [
            "Array",
            "Numerical contributors",
            "Physical contributors",
            "Added numerical",
            "Added physical",
        ],
        [
            [
                r["array"],
                result["runs"]["joint-numerical"]["arrays"][i]["contributors"],
                result["runs"]["joint-physical"]["arrays"][i]["contributors"],
                result["runs"]["joint-numerical"]["arrays"][i][
                    "added_row_contributors"
                ],
                result["runs"]["joint-physical"]["arrays"][i]["added_row_contributors"],
            ]
            for i, r in enumerate(result["arrays"])
        ],
    )
    text += r"""
**Interpretation limits.** The inherited acquisition-to-APT correspondence is
not an independent component beam or flux calibration. Joint covariance is
retained in diagnostics, but the map accumulator uses per-row weights rather
than full cross-component covariance. Added components must not be counted as
independent sensitivity gains. Annulus RMS includes residual astronomical and
correlated signals; signal transfer and mapping speed remain unvalidated.
The measured Python/native parity discrepancy from 145609 remains a release
limitation. These are comparison maps, not newly promoted pointing calibration.

The following corrected scalar controls retain their original population and
remain useful for separating model and sample-policy effects. The older
pre-gap maps are only in the appendices.

\newpage

"""
    return text


def current_comparison(gap, summary, figure):
    """Present the corrected policies before the diagnostic history."""
    text = r"""
# 152434: current numerical and physical acceptance comparison

Both policies use the **same fitted nonlinear V3 model and optical-detuning
output**. They are sample-acceptance policies, not numerical versus physical
models. The complete pointing integration spans 3,628 saved samples in 12 chunks
on networks 0--5, 7--9, 11 and 12; networks 6 and 10 are unavailable.

**Table 7. Current controls, all using the corrected native engine.**

| Run | Model | Accepted samples |
| --- | --- | --- |
| Legacy reference | Legacy tune.txt coefficients | Numerical validity; legacy has no equivalent physical-domain gate. |
| V3 numerical | Python affine nonlinear fit, final-block stubs | Finite algebraic solutions; domain/stability warnings remain recorded. |
| V3 physical | Exactly the same V3 fit and stubs | Numerical validity plus domain/stability checks. |

The same common-eligibility APT and per-channel weights apply to all three.
Legacy and V3 numerical have identical sample masks, map weights and coverage.
Physical acceptance changes coverage. Invalid models, singular or nonfinite
responses, and failed joint recovery remain rejected under either policy.

**Population denominator.** The original matched APT contains 4,646 beammap-good
rows. The model/association gate admits 3,096: 1,935 / 663 / 498 by array.
The current runs include this entire eligible population, without choosing an
always-valid subset. They do not restore the missing joint-component APT
bindings or unusable models. Counts are channel/APT rows, not independent
optical detectors. Section 7 explains those upstream exclusions. [R5, R9]

"""
    text += figure(
        "gap-populations",
        "Current contributing channel/APT rows. All three runs start from the same eligible population; physical acceptance rejects additional samples and entirely removes two contributors each in a1100 and a1400.",
    )
    text += "\n\\newpage\n\n## Map fluctuations and the cost of physical acceptance\n\n"
    text += figure(
        "map-quality-summary",
        "Corrected maps: annulus RMS and additional samples rejected by physical acceptance. RMS uses identical covered 20--40 arcsec pixels. Sample percentages are relative to V3 numerical accepted channel-samples; per-channel weights are unchanged.",
    )
    text += table(
        [
            "Array",
            "Legacy RMS",
            "Numerical RMS",
            "Physical RMS",
            "RMS decrease",
            "Samples retained",
        ],
        [
            [
                r["array"],
                *[f"{r['rms_mJy_beam'][n]:.2f}" for n in summary["order"]],
                f"{100 * (1 - r['physical_to_numerical_rms']):.2f}%",
                f"{100 * r['physical_retained_fraction']:.3f}%",
            ]
            for r in summary["arrays"]
        ],
    )
    text += r"""
RMS is in mJy/beam. The last two columns compare physical with numerical
acceptance. Physical acceptance lowers the measured fluctuations most in a1100
and a2000 and only slightly in a1400. It changes the samples used to estimate
the cleaning basis as well as those entering the map. The sample cost is
35,137 / 26,600 / 7,321 additional rejected channel-samples, respectively. [R9]

Against the matched legacy reference, the earlier large a1100 excess is absent.
V3 numerical remains higher in a1400 and slightly higher in a2000. These are
the current results; the original maps in Appendix A use superseded processing.

**Sensitivity and mapping speed are not validated.** Annulus RMS includes
correlated residuals and possible source wings. The APT flux conversion is held
fixed; equal astronomical signal transfer and noise depth versus time/area have
not been demonstrated. Lower RMS alone is insufficient to select a policy or
claim improved optical performance. No new pointing calibration is adopted.

\newpage

## Current maps

"""
    text += figure(
        "gap-maps",
        "All three arrays with corrected gap handling. Columns: legacy numerical, V3 numerical, V3 physical. Each array shares one intensity scale. No registration or signal rescaling; physical acceptance has different coverage.",
    )
    text += r"""
\newpage

## How rejected samples pass through filtering and PCA

The correction makes filling independent of spike detection. The shared
despiker/RTC filler uses an endpoint baseline, observed network fluctuations and
reproducible residual noise. Filled values never become map measurements;
every FIR output whose support touches a gap remains rejected.

PCA covariance uses valid overlaps. For a centered detector vector $y$, mode
matrix $V$ and validity mask $M$, mode amplitudes are estimated using

\[
a=(V^T M V)^{-1}V^T M y,\qquad z=y-Va.
\]

Thus stored replacement values cannot enter valid-probe amplitudes. Insufficient
mode support rejects that network/time sample. Tests cover arbitrary replacement
values, endpoint gaps, entirely missing channels and the actual map accumulator.
The saved native projection is reproduced within $1.2\times10^{-9}$ mJy/beam.

```yaml
kids:
  solver:
    sample_acceptance: numerical  # physical remains the default
```

Both V3 policies preserve identical solver diagnostic bits. Keeping a finite
solved value under numerical acceptance is distinct from admitting a gap fill.
This setting does not change sweep fitting, nonlinear correction or branch
tracking. The earlier failure mechanism and the diagnostic strict subset are
retained in Appendix C; neither is the current processing path. [R8, R9]

\newpage

## Current network trajectories

"""
    text += figure(
        "gap-trajectories",
        "After corrected PCA, network means use identical APT weights and each policy's accepted samples. Filled values receive no weight. Curves are not noise-only estimates and policy differences include changed sample acceptance.",
    )
    text += table(
        ["Run", "a1100", "a1400", "a2000"],
        [
            [
                name,
                *[
                    f"{r['accepted_channel_samples']:,}"
                    for r in gap["runs"][name]["arrays"]
                ],
            ]
            for name in summary["order"]
        ],
    )
    text += "\nAccepted channel-samples by array; the complete denominators and native checks are in gap-control.json.\n\n\\newpage\n\n"
    return text


def assemble_document(content, current):
    """Keep historical controls out of the current scientific conclusions."""
    intro, history = content.split("# Example: 152434 and the APT control\n", 1)
    apt, controls = history.split("## Map RMS and sensitivity status\n", 1)
    _, controls = controls.split(
        "## Coordinate control: map the same V3 fits as x'\n", 1
    )
    coordinate, following = controls.split(
        "## Trace a1100 with matched detectors and weights\n", 1
    )
    pca, following = following.split(
        "# Sweep-model quality: legacy tune.txt versus the full V3 model\n", 1
    )
    models, following = following.split(
        "# Multiplicity, nonlinearity and quality of actual map contributors\n", 1
    )
    populations, following = following.split("# 152434: a1100 map comparison\n", 1)
    maps, handoff = following.split("# Handoff contents and remaining work\n", 1)
    maps = "## a1100: original map comparison\n" + maps
    for array in ARRAYS[1:]:
        maps = maps.replace(
            f"# 152434: {array} map comparison", f"## {array}: original map comparison"
        )
    maps = maps.replace(
        "## Representative retained joint sweep model",
        "### Representative retained joint sweep model",
    )
    coordinate = coordinate.replace("\n### ", "\n## ")
    pca = pca.replace("\n### ", "\n## ")
    appendix = (
        "\n\\newpage\n\\appendix\n\n# Original maps and population accounting\n\n"
        "**Historical captures, before the gap correction.** The following APT audit, contributor plots, maps and native pointing fits document the initial investigation. Their RMS values and final populations do not describe the current numerical/physical runs in Section 6. The original calibrations and fitted models remain traceable.\n\n"
        + apt
        + "\n## Original contributor composition\n"
        + populations
        + maps
        + "\n\\newpage\n\n# Historical coordinate experiment\n\n"
        "**This experiment retains the old sample handling.** It asks whether changing optical detuning to x' alone explains the original maps. It is not the current acceptance-policy comparison.\n\n"
        + coordinate
        + "\n\\newpage\n\n# Historical PCA diagnosis\n\n"
        "**The 1,855-channel subset was a diagnostic control.** The correction and full eligible-population runs are now complete in Section 6. This appendix preserves the evidence that motivated that correction.\n\n"
        + pca
    )
    appendix = appendix.replace("![", "![Historical evidence: ")
    assembled = (
        intro
        + current
        + "# Sweep-model quality: legacy tune.txt versus the full V3 model\n"
        + models
        + "# Handoff contents and remaining work\n"
        + handoff
        + appendix
    )
    # Table labels follow the reorganized document, rather than stale chapter order.
    number = 0

    def label(_match):
        nonlocal number
        number += 1
        return f"**Table {number}."

    return re.sub(r"\*\*Table \d+\.", label, assembled)


def build(root, output):  # noqa: C901, PLR0912, PLR0915 -- sequential document assembly
    """Freeze source evidence and build the review."""
    root, output = root.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    release = root / "run/grant-tune-release-20261122"
    result = root / "run/map-comparison-152434/full-array"
    sources = {
        "release-manifest.json": release / "release-manifest.json",
        "release-readme.md": release / "README.md",
        "full-replay.json": release / "evidence/full-replay.json",
        "populations.json": result / "report/populations.json",
        "population-ledger.ecsv": result / "report/population-ledger.ecsv",
        "metrics.json": result / "report/metrics.json",
        "native-coverage.json": result / "report/native-coverage.json",
        "native-build.json": result / "native-build.json",
        "workflow-guide.md": root / "docs/kids-tune-comparison.md",
        "model-contract.md": root / "design/spec/kids-model-reports.md",
    }
    for family in ("affine", "delay"):
        sources[f"full-native-parity-{family}.json"] = (
            release / f"evidence/full-native-parity-{family}.json"
        )
    for name in (
        "sweep-model-quality.json",
        "sweep-model-responses.ecsv",
        "sweep-model-channels.ecsv",
        "sweep-example-selection.json",
        "population-flow.pdf",
        "model-coverage.pdf",
        "model-exclusions.pdf",
        "sweep-model-quality.pdf",
        "map-quality-summary.pdf",
        "map-quality-summary.json",
        "xprime-control.json",
        "xprime-rms.pdf",
        "xprime-a1100.pdf",
        "xprime-a1400.pdf",
        "xprime-a2000.pdf",
        "pca-trace.json",
        "pca-flag-projection.json",
        "pca-channel-chunks.ecsv",
        "pca-spectra.npz",
        "pca-flag-trajectories.npz",
        "pca-matched-maps.pdf",
        "pca-flag-leakage.pdf",
        "pca-stage-spectra.pdf",
        "pca-stage-trajectories.pdf",
        "pca-transfer.pdf",
    ):
        sources[name] = result / "report" / name
    for name in (
        "pca_trace.py",
        "pca_trace_analysis.py",
        "pca_flag_projection.py",
        "pca_trace_report.py",
    ):
        sources[f"pca-control/{name}"] = root / "examples/tune_model_comparison" / name
    for name in ("selection.json", "channel-selection.ecsv", "matched-apt.ecsv"):
        sources[f"pca-control/{name}"] = result / "pca-control" / name
    for phase in ("capture", "matched"):
        for model in ("legacy", "v3"):
            for name in ("config.yaml", "execution.json"):
                sources[f"pca-control/{phase}/{model}/{name}"] = (
                    result / "pca-control" / phase / model / "redu00" / name
                )
    sources["pca-control/native-clean.h"] = (
        root
        / "run/sweep-anchored-solving/native/citlali/include/citlali/core/timestream/ptc/clean.h"
    )
    sources["pca-control/test_pca_trace.py"] = (
        root / "tests/tolteca_kids/test_pca_trace.py"
    )
    for name in (
        "gap-control.json",
        "gap-populations.pdf",
        "gap-maps.pdf",
        "gap-trajectories.pdf",
        "gap-trajectories.npz",
    ):
        sources[name] = result / "report" / name
    for name in ("build_gap_control.py", "gap_control.py", "gap_control_report.py"):
        sources[f"gap-control/{name}"] = root / "examples/tune_model_comparison" / name
    for name in ("gap-citlali.patch", "gap-kidscpp.patch"):
        sources[f"gap-control/{name}"] = (
            root / "examples/tune_model_comparison/native" / name
        )
    sources["gap-control/build.json"] = root / "run/gap-handling-native/build.json"
    for trial in ("legacy", "v3-physical", "v3-numerical"):
        for name in ("config.yaml", "execution.json"):
            sources[f"gap-control/{trial}/{name}"] = (
                result / "gap-control" / trial / "redu00" / name
            )
    joint_path = result / "report/joint-control.json"
    if joint_path.exists():
        for name in (
            "joint-control.json",
            "joint-map-summary.pdf",
            "joint-maps.pdf",
            *(f"joint-maps-{a}.pdf" for a in ARRAYS),
            "joint-legacy-maps.pdf",
            *(f"joint-legacy-{a}.pdf" for a in ARRAYS),
        ):
            sources[name] = result / "report" / name
        for name in (
            "selection.json",
            "channel-disposition.ecsv",
            "component-bindings.ecsv",
            "component-bindings.json",
            "joint-contributors.ecsv",
            "joint-apt.ecsv",
            "legacy-apt.ecsv",
        ):
            sources[f"joint-control/{name}"] = result / "joint-control" / name
        for name in ("prepare_joint_map_control.py", "joint_map_report.py"):
            sources[f"joint-control/{name}"] = (
                root / "examples/tune_model_comparison" / name
            )
        for name in ("joint-map-citlali.patch", "joint-map-kidscpp.patch"):
            sources[f"joint-control/{name}"] = (
                root / "examples/tune_model_comparison/native" / name
            )
        sources["joint-control/build.json"] = root / "run/joint-map-native/build.json"
        for policy in ("numerical", "physical", "legacy"):
            run = Path(
                json.loads(joint_path.read_text())["runs"][
                    "legacy-broader" if policy == "legacy" else f"joint-{policy}"
                ]["path"]
            )
            for name in ("config.yaml", "execution.json"):
                sources[f"joint-control/{policy}/{name}"] = run / name
    for name in (
        "build_xprime_control.py",
        "run_xprime_control.py",
        "array_coordinate_control.py",
    ):
        sources[f"xprime-control/{name}"] = (
            root / "examples/tune_model_comparison" / name
        )
    for name in ("build.json", "xprime-output.patch", "coordinate_fixture.cpp"):
        sources[f"xprime-control/native/{name}"] = (
            result / "xprime-control/native" / name
        )
    for trial in ("standard", "no-variance-cuts-opt", "no-variance-cuts-prime"):
        for name in ("config.yaml", "execution.json"):
            sources[f"xprime-control/{trial}/{name}"] = (
                result / "xprime-control" / trial / "redu00" / name
            )
    for path in (result / "report/timestream-examples").iterdir():
        if path.is_file():
            sources[f"timestream-examples/{path.name}"] = path
    for array in ARRAYS:
        for suffix in (
            "",
            "-populations",
            "-populations-print",
            "-sweep-example",
            "-sweep-quality",
        ):
            name = f"{array}{suffix}.pdf"
            sources[name] = result / "report" / name
    evidence = output.parent / (output.stem + "-evidence")
    evidence.mkdir(exist_ok=True)
    receipt = {"date": "2026-11-23", "status": "technical-review-draft", "sources": {}}
    for name, path in sources.items():
        (evidence / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, evidence / name)
        receipt["sources"][name] = {
            "original": str(path),
            "sha256": hashlib.sha256((evidence / name).read_bytes()).hexdigest(),
        }

    def read(name):
        return json.loads((evidence / name).read_text())

    _manifest, replay, audit, metrics = [
        read(n)
        for n in (
            "release-manifest.json",
            "full-replay.json",
            "populations.json",
            "metrics.json",
        )
    ]
    populations = {int(float(k)): v for k, v in audit["arrays"].items()}
    parity = {f: read(f"full-native-parity-{f}.json") for f in ("affine", "delay")}
    assert (
        audit["apt"]["sources"]["legacy"]["sha256"]
        == audit["apt"]["sources"]["python"]["sha256"]
    )
    assert audit["apt"]["common_changed_columns"] == ["flag"]
    assert all(
        p["apt_bad"] == 0
        for a in populations.values()
        for p in a["populations"].values()
    )

    def figure(name, caption, *, width="100%"):
        path = (evidence / f"{name}.pdf").as_posix()
        return f"![{caption}]({path}){{width={width}}}\n\n"

    quality = read("sweep-model-quality.json")
    coordinate = read("xprime-control.json")
    sections = [
        r"""---
title: "TolTEC Tune-model comparison"
subtitle: "V2 reference, V3 physical modeling, and local reduction evidence"
author: "Technical review for Zhiyuan Ma and Grant Wilson"
date: "23 November 2026"
documentclass: article
fontsize: 11pt
geometry: "a4paper,margin=0.75in"
colorlinks: true
linkcolor: black
urlcolor: blue
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\small TolTEC Tune-model comparison}
  - \fancyhead[R]{\small Technical review draft}
  - \fancyfoot[C]{\small 2026-11-23 \quad | \quad \thepage}
  - \setlength{\headheight}{14pt}
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\small}
---

# Status and scope

**This is a review draft, not a validated native comparison release.**
The maintained Python workflow fits nonlinear resonator groups, saves complete
models, prepares acquisition-specific stubs, solves measured I/Q, and persists
diagnostics. The full 145609 network replay is reproducible locally. Observation
152434 provides full-integration maps for all three arrays.

C++ ECSV/stub transport passes, but strict measured Python/native solving
parity remains failed. These results do not establish improved sensitivity.

**Updated a1100 finding:** the earlier 1,855-channel control identified
flagged-value leakage through PCA. Appendices B--C retain that investigation; Section 6 compares
the corrected scalar acceptance controls and the broader joint-component maps. [R7--R9]

**Table 1. Current handoff status.**

| Part | Status |
| --- | --- |
| Python workflow | Coordinated wheels, source archives and commit series exist. Full 145609 replay passed, retaining all fitted components. |
| Native transport | Exact ECSV/stub equality over measured data; controlled trajectories passed for both Tune blocks. |
| Native parity | Measured disagreement in 14 affine and 16 delay groups out of 473. |
| Full-array maps | 152434 completes on 11 available networks. APT and actual contributors are audited separately. |
| Release package | The 152434 changes and native patches postdate the 22 November wheel bundle; incorporation into a refreshed release remains. |
| External handoff | Local materials only. This report does not publish code or message Grant. |

Current map controls use one corrected V3 engine with legacy coefficients or
V3 models. The original legacy-engine trials are in Appendix A; that development
executable included a packet-counter correction. Grant's NGC4449 observations
and refactored Citlali target are not yet reproduced.

Section 6 is the current map comparison; Section 7 presents model evidence.
Appendices A--C preserve the earlier investigation.

\newpage

# What changes from the legacy reference

**Table 2. Scope of this comparison, rather than every historical deployment.**

| Concern | Legacy reference | V3 comparison |
| --- | --- | --- |
| Response | Per-channel GainLinTrend coefficients. | Nonlinear resonator groups with complex affine or delay gain; multiple resonators and probes. |
| Nonlinearity | Legacy asymmetry $A$ is not this physical $a_{\rm nl}$. | Continuous fitted $a_{\rm nl}$ for every component; bins do not switch algorithms. |
| Calibration | Recorded acquisition references and existing channel model/APT conventions. | Exact sweep block, model realization and complete group coefficients retained together. |
| Handoff | Legacy numeric parameters and tune/fit reports. | Per-block kids_model.ecsv, with full metadata encoded into target-acquisition stubs. |
| Solving | Legacy channel outputs remain the control. | Same fitted response at measured probes, with sweep anchoring, continuity and explicit failures. |
| Diagnostics | Existing map/quicklook summaries. | Saved trajectories, $x'$ versus $x_{\rm opt}$, cross-spectra, fit adequacy and independent response evidence. |
| Map gate | Beammap APT flag-zero rows plus later cuts. | Same beammap gate plus model usability and component-association requirements. |

**Backend roles.** Python is the reference fitting backend for this deliverable.
“Native” means the C++ Kidscpp consumer/solver used by Citlali; it is not a
separate scientific calibration. Kidscpp remains useful for rapid
acquisition-time fitting. This comparison uses Python-fitted models.

Tolteca prepares configurations, calibration context and data products.
Citlali is the map engine. A stub changes selected metadata presented to
a consumer; it does not rewrite raw data.

**Outside scope.** This review does not change Tlaloc acquisition operations,
propose new detector-power settings or promote a new beammap/pointing
calibration. Earlier detector-operation work is outside this release.

\newpage

# The fitted model and recovered coordinates

For probe $k$ and resonator $j$, the implemented response is

\[
z_k=H_k(f_{p,k})\left[1-\sum_j\frac{c_{kj}}{r_j+i x'_{kj}}\right].
\]

The sweep determines gain, coupling and reference resonator parameters.
The local fixed-drive solving model uses

\[
x_{{\rm opt},kj}=\frac{f_{p,k}}{f_{r,j}}-1,\qquad
x'_{kj}=x_{{\rm opt},kj}+\frac{b_j}{r_j^2+x_{kj}'^2},
\qquad b_j=2a_{{\rm ref},j}r_{{\rm ref},j}^{3}.
\]

During solving, probe frequency, gain, coupling and $b$ remain fixed; intrinsic
resonance frequency and loss vary. Sweep-anchored branch selection and
continuity operate on this same model. Unsupported or ambiguous output
retains explicit flags.

**Table 3. Coordinates and assumptions.**

| Quantity | Meaning |
| --- | --- |
| $r$ | Recovered dimensionless loss, calibrated by the sweep. |
| $x'$ | Effective detuning in the response; for blended probes, evaluated from the complete group model. |
| $x_{\rm opt}$ | Intrinsic detuning after the modeled readout-power shift; not a separation of optical loading from all intrinsic fluctuations. |
| $b$ | Fixed local drive coefficient derived from the reference fit, not measured independently at each sample. |
| $a_{\rm nl}$ bins | Descriptive, not optical acceptance thresholds or alternative solving modes. |

Recomputing $b/[2r(t)^3]$ adds no independent measurement of instantaneous
nonlinearity. Fixed $b$ across loading changes is a hypothesis to test,
not a consequence of Skydip measurements.

Diagnostics retain signed covariance. Locally,

\[
\delta x_{\rm opt}\simeq D\,\delta x'+\mu\,\delta r,\qquad
P_{\rm opt}\simeq D^2P_{x'}+\mu^2P_r+2D\mu\,{\rm Re}\,C_{x'r},
\]
\[
D=1+\frac{2bx'}{(r^2+x'^2)^2},\qquad
\mu=\frac{2br}{(r^2+x'^2)^2}.
\]

A larger $x_{\rm opt}$ PSD can include transformed loss noise and correlations.
It is not by itself a lower optical signal-to-noise ratio. Equal-signal
comparison requires independent response evidence.

\newpage

# From acquisition to reproducible products

\begin{center}
\begin{tabular}{c}
Raw Tune, both sweep blocks\\
$\downarrow$\\
Python ModelFit: full group model, quality, per-block ECSV\\
$\downarrow$\\
Recorded calibration binding $\longrightarrow$ target-acquisition stub\\
$\downarrow$\\
Measured I/Q + same selected fitted model\\
$\downarrow$\\
Solving $\longrightarrow$ diagnostics and map trials\\
$\downarrow$\\
DataProdOutput: scientific arrays, artifacts and portable index
\end{tabular}
\end{center}

**Figure 1.** One model has two consumer representations. Python reads the model
product; the native route reconstructs the ECSV schema from the selected stub.
Transport and numerical solving are separate checks.

\setcounter{figure}{1}

**Table 4. Maintained workflow responsibilities.**

| Operation | Saved evidence |
| --- | --- |
| ModelFit, both blocks | Zarr arrays, fit quality/group selection, one ECSV per fitted block and figures. |
| Target context | Raw observation, recorded sweep block, exact model reference and one target-associated stub per network. |
| TimestreamSolve | Recovered states, branch and numerical/sample flags; failed components remain in the ledger. |
| Diagnostics and Plot | Same-model $x'$, cross-spectra, I/Q residuals, fit/evidence categories and interactive figures. |
| DataProdOutput | array.zarr, declared artifacts, index.yaml and run records. |
| Citlali trial | Resolved config, reduNN, FITS, native PPT and per-chunk weight/flag statistics. |

Both Tune blocks are retained; only the final recorded calibration sweep
supplies the measured scan-2 model. Refinement history does not make earlier
sweeps interchangeable. I/Q normalization follows recorded accumulation lengths.

Common model reports retain variable-sized groups. The fit_unusable flag
vetoes use; fit_joint_required suppresses independent channel solving.
Warnings preserve results; zero flags do not establish optical suitability.

\newpage

# Example: 145609 full-network Python replay

Network 0: all 632 acquired probes, 7,802 measured samples at 122.0703125 Hz.
Both Tune blocks are fitted; the final block calibrates scan 2. Affine and delay
gain families use the same nonlinear configuration. This is a network test,
not a three-array map comparison.

**Table 5. Full-record checks. Categories are independent, not a selection funnel.**

"""
    ]
    rows = []
    for label, key in (
        ("Fitted components retained", "fitted_components"),
        ("Complete trajectories and spectra", "spectral_complete"),
        ("Declared artifacts", "artifacts"),
    ):
        rows.append([label] + [replay["families"][f][key] for f in ("affine", "delay")])
    rows += [
        ["Separate 5% sweep-residual diagnostic policy", 400, 401],
        ["Independent response-qualified comparisons", 113, 113],
        ["Routed groups", 473, 473],
        ["Measured native/Python disagreeing groups"]
        + [len(parity[f]["disagreeing_groups"]) for f in ("affine", "delay")],
    ]
    sections.append(table(["Check", "Affine", "Delay"], rows))
    sections.append(r"""
All four fitted block reports match the bundled ECSV examples byte-for-byte.
Every declared artifact was found on disk and raw-file hashes were unchanged.
A fresh Python 3.13 installation passed 54 focused numerical/workflow tests
with warnings treated as errors. [R1, R2]

**Table 6. Transport and solving have distinct validation results.**

| Check | Result | Interpretation |
| --- | --- | --- |
| Controlled samples, both blocks | Passed | 16 controlled samples per routed group in the recorded check. |
| Native ECSV versus stub | Passed | Exact equality over all 7,802 measured samples, both gain families. |
| Measured Python versus native | Failed | Flags, validity or states disagree in 14/16 groups; the strict gate was not relaxed. |

The 5% diagnostic threshold compares fit residual RMS with measured within-probe
sweep excursion. The paired sweep comparison in Section 7 uses the same excursion concept;
the older group-quality metric retained in the ledger uses total IQ instead.
Response qualification uses pinned Skydip/APT evidence. Missing evidence remains
distinct from fitting/readout failure; the 113-component cohort is not the
count of all useful detectors.

**Release implication.** Python is the detailed fitting and solving baseline.
The native route is reviewable, but measured-data parity remains unresolved.
The native maps below do not override that failure.

\newpage

# Example: 152434 and the APT control

The full pointing integration is reduced in 12 chunks on networks 0--5, 7--9,
11 and 12. Networks 6 and 10 are absent from the available acquisition.
Both Tune blocks are fitted on all 11 networks; final-block stubs feed the
affine nonlinear model trial.

**Table 7. Controls and what they actually hold fixed.**

| Trial | Model / engine | APT selection |
| --- | --- | --- |
| Legacy full | Legacy coefficients, local legacy development engine. | Original matched beammap APT. |
| Engine control | Legacy coefficients, pinned V3 engine. | Original APT; isolates the engine change. |
| Legacy eligible | Legacy coefficients, same V3 engine as the model trial. | Original APT restricted to V3 initial model eligibility. |
| V3 models | Python-fitted models via stubs, consumed by V3 Citlali. | Original APT plus native validity/routing restrictions. |

**Full legacy and V3 use identical APT bytes.** Of 5,518 rows, 4,646 have
flag == 0. The common-eligibility APT has 3,096 flag-zero rows; only its flag
column changes. Positions, flux factors, units and table metadata are unchanged.
No beammap-bad row contributes to the audited maps. [R3]

""")
    sections.append(
        figure(
            "population-flow",
            "Beammap selection, added scalar-map restrictions and actual map contributions. Counts are acquired channel/APT rows, not independent optical detector identities.",
        )
    )
    sections.append(r"""
“Used” means finite positive saved map weight and finite flagged_frac < 1
in at least one chunk. It does not mean every sample is valid. Section 7
separates sweep model quality from the restrictions that prevent map use.
""")
    sections.append("\n\\newpage\n\n## Map RMS and sensitivity status\n\n")
    sections.append(
        figure(
            "map-quality-summary",
            "Common-eligibility legacy versus V3: measured background RMS and formal median inverse-square-root map weight on identical covered 20--40 arcsec annulus pixels. The actual maps and signed differences follow in the per-array sections.",
        )
    )
    sections.append(r"""
The formal uncertainties are almost unchanged: approximately 6.69, 5.88 and
4.09 mJy/beam. The measured annulus RMS is substantially larger and worsens
in V3, especially in a1100 and a1400. The annulus includes correlated residuals
and possible source wings. Weight-derived errors assume the adopted noise
model; they do not capture all residual map structure. [R4, R6]

**Mapping speed has not been validated for these trials.** The saved FITS contain
signal, weight and coverage, but no noise realizations. We need matched
astronomical signal transfer and noise depth as a function of integration
time and area before reporting a mapping-speed comparison. Detector count,
raw-coordinate PSD or the reciprocal square of this annulus RMS would not
provide that validation. The NEFD and mapping-speed convention follows
[Perotto et al. (2020), Section 10.1](https://doi.org/10.1051/0004-6361/201936220).

No reduction is rerun or flux-scaled for these plots. These saved maps remain
comparison products; in particular the a1400 pointing fit reaches its width
bound and is not accepted as calibration.
""")
    sections.append(
        "\n\\newpage\n\n## Coordinate control: map the same V3 fits as x'\n\n"
    )
    sections.append(
        figure(
            "xprime-rms",
            "Coordinate control on shared covered 20--40 arcsec annulus pixels. Left: recorded map settings; right: both variance-dependent cuts disabled in both V3 runs. The legacy control is shown only at its recorded settings.",
        )
    )
    sections.append(r"""
**In this earlier experiment, the coordinate correction alone did not explain
the a1100 excess.** Its $x'$ map remained noisier than legacy; a1400 improved
substantially, while a2000 changes only slightly. These are background RMS
comparisons under the held APT calibration, not validated sensitivity gains.

The isolated native build returns $x'$ for valid scalar-model samples after
performing the existing $x_{\rm opt}$ validation. It keeps fitted $b$, loss
values and solver flags. No sweep is refitted, no nonlinear coefficient is
zeroed, and no blended-component map is introduced. The original native
source, library and executable remain unchanged. Controlled tests cover 18
samples across affine, delay and zero-$b$ cases; all seven valid outputs and
all rejected-sample flags match the intended projection. [R7]

The recorded settings change some downstream channel/chunk contributions.
The second pair sets only the lower and upper inverse-variance cuts to zero in
both raw and processed time chunks. Saved contributions, flagged fractions and
per-channel weights then agree exactly. PCA is still recomputed for each
coordinate. Equal flagged fractions alone do not prove bitwise identity of
every final sample mask.
""")
    sections.append("\n\\newpage\n\n### Contribution and calibration controls\n\n")
    sections.append(
        "**Table 8. Exact saved contributor audit. Counts are channel/APT rows.**\n\n"
    )
    sections.append(
        table(
            [
                "Array",
                "Recorded: x_opt / x'",
                "Changed chunk/channel entries",
                "Cuts off: shared rows / entries",
            ],
            [
                [
                    n["array"],
                    f"{n['contributors'][0]} / {n['contributors'][1]}",
                    n["changed_contribution_entries"],
                    f"{c['contributors'][0]} / {c['contributing_chunk_channels'][0]}",
                ]
                for n, c in zip(
                    coordinate["contributions"]["standard_opt_to_prime"],
                    coordinate["contributions"]["nocut_opt_to_prime"],
                    strict=True,
                )
            ],
        )
    )
    sections.append(r"""
The same raw acquisition, V3 fits, stubs and APT enter both coordinates. The
APT's flux conversion and sensitivity values remain fixed to their existing
calibration. This control therefore identifies coordinate-dependent map
structure under that calibration; it does not establish the correct optical
signal transfer or authorize changing the production science coordinate.

Each comparison uses identical finite covered sky pixels, with an intersection
of the maps' coverage cuts. Integer-aligned pixels are selected directly;
there is no image registration, interpolation or signal rescaling. The two
panels in the RMS figure use their own common pixel intersection.

The following maps use the recorded settings, with one signal scale per array
and a symmetric scale for the signed difference. Their differing variance-cut
decisions are part of that view. The second pair provides the contribution
control; its exact results, FITS hashes and configs are in xprime-control.json
and the frozen execution records. The workflow guide gives the build, three
reduction commands and report command.

The original a1100 residual structure persisted in both coordinates. Appendix C
traces it through matched validity and the old native PCA projection.
The fixed-$b$ assumption and coordinate-dependent response also remain part
of the scientific validation, particularly for a1400. [R7]
""")
    for row in coordinate["standard"]:
        array = row["array"]
        sections.append(
            f"\n\\newpage\n\n### {array}: legacy, x' and optical-coordinate maps\n\n"
        )
        sections.append(
            figure(
                f"xprime-{array}",
                f"{array}: recorded map settings, shared sky pixels and unchanged calibration. The difference is x' minus x_opt. Variance-dependent selection can differ.",
            )
        )
        legacy_rms, prime_rms, opt_rms = row["rms_mJy_per_beam"]
        sections.append(
            f"Annulus RMS: legacy **{legacy_rms:.2f}**, V3 $x'$ **{prime_rms:.2f}**, V3 $x_{{\\rm opt}}$ **{opt_rms:.2f}** mJy/beam.\n\n"
        )
    sections.append(r"""
\newpage

## Trace a1100 with matched detectors and weights

**The earlier V3 excess is absent on the strict common population.** The new
native runs give **55.13 mJy/beam for legacy and 40.76 for V3** on the same
covered 20--40 arcsec annulus. Map coverage and weights agree exactly. [R8]
""")
    sections.append(
        figure(
            "pca-matched-maps",
            "Native a1100 Jinc maps on 1,855 identical channel/APT rows, with equal sample masks and weights. One signal scale and a symmetric difference scale; no signal rescaling.",
        )
    )
    sections.append(r"""
Both model paths use the same pinned Citlali executable: legacy tune.txt
coefficients versus V3 $x_{\rm opt}$. Select only channels finite and unflagged
at every pre/post-PCA sample in both captures. No residual, nonlinearity or
noise-quality cut is added. Of 1,935 initially eligible a1100 rows, legacy
excludes 33 and V3 excludes 80, including those same 33. The resulting network
0--5 counts are 357, 276, 292, 329, 297 and 304.

There are 3,628 identical samples in 12 chunks. Every retained sample is
unflagged; timestamps, pointings and channel weights agree exactly. The
pre-PCA stage is after filtering, decimation by two and held APT flux
calibration. The post-PCA stage follows per-channel mean subtraction and
five-mode network cleaning. Variance cuts are disabled in both runs.

This is a validity control, not a proposed permanent detector exclusion or a
validated sensitivity gain. The existing APT flux factors remain fixed.

\newpage

### Flagged stored values contaminate valid channels

In the earlier implementation, covariance estimation masks flagged samples,
but the projection uses
the entire stored signal matrix when estimating mode amplitudes. Partially
flagged probes therefore contaminate the amplitudes subtracted from valid
probes. Legacy has no partially flagged probes in these captures; V3 does.
""")
    sections.append(
        figure(
            "pca-flag-leakage",
            "Same always-valid target channels and weights. The diagnostic valid-probe least-squares replay uses the original captured PCA basis. Native strict population additionally refits the basis using only the shared valid population.",
        )
    )
    sections.append(r"""
In network 1, the V3 weighted per-channel residual RMS falls from 1,651 to
672 mJy/beam; in network 4, from 2,538 to 642. These are timestream fluctuation
RMS values, not map RMS or noise-only sensitivity. The valid target channels'
pre-PCA samples are identical in the broader and strict captures. [R8]

\newpage

### Reproduce the native projection and isolate the leakage

Let $y$ be one centered detector vector, $V$ the five saved detector-space
modes, and $M$ a diagonal mask equal to one for valid samples. The old cleaning
used

\[
z=y-VV^T y.
\]

The diagnostic valid-probe fit estimates amplitudes from

\[
a=(V^T M V)^{-1}V^T M y,\qquad z=y-Va.
\]

The native output is reproduced from saved eigenvectors within 1.22e-9
mJy/beam maximum absolute error. Zeroing flagged contributions removes their
direct leakage; valid-probe least squares also accounts for missing basis
rows. The diagnostic rejects rank-deficient mode-amplitude systems explicitly.
Those algebraic diagnostic replays did not change native processing or create
science maps. The subsequent native correction is tested by the current runs.

On the strict common population, saved pre/post-PCA spectra follow below.
They use Hann Welch estimates with 256 samples and 128 overlap within each
native chunk. Source and correlated sky signal remain in these spectra;
they are not a noise-only sensitivity measurement.

The exported scan-start indices assume equal chunk lengths. The diagnostic
reader reconstructs append-order boundaries from the raw inner ranges and
decimation, then verifies the complete native PCA output. This fixes diagnostic
indexing without changing any acquisition sample or native map calculation.
""")
    sections.append(
        figure(
            "pca-stage-trajectories",
            "Array-weighted mean timestream on the same 1,855 channels. Pre-PCA chunk means are removed as in native processing. The time axis and channel weights are identical; the panels use different vertical ranges.",
        )
    )
    sections.append(r"""

\newpage

### Six-network pre/post-PCA spectra
""")
    sections.append(
        figure(
            "pca-stage-spectra",
            "The same channels, samples and weights in both model paths. Dashed: pre-PCA; solid: post-PCA. These are weighted per-channel fluctuation ASD values under the held APT flux calibration.",
        )
    )
    sections.append(r"""
PCA suppresses large common fluctuations in both paths. The strict-population
post-PCA differences vary by network; there is no uniform V3 noise penalty.
Per-channel/chunk records retain every regression gain, correlation, standard
deviation and map weight. The native modes are estimated without map weights,
as in Citlali; diagnostic averages use the exact saved map weights.

\newpage

### Signal transfer before PCA and remaining work
""")
    sections.append(
        figure(
            "pca-transfer",
            "Model-dependent response scale is already visible before PCA. Left: median and 10--90 percent intervals of channel/chunk regression gains. Right: weighted fluctuation RMS before and after each model's native PCA. No empirical gain is applied to any map.",
        )
    )
    sections.append(r"""
The input change and the change of cleaning basis are coupled. Exchanging the
saved bases gives larger residuals in both directions. The recorded algebraic
decomposition retains its signed cross term; its two positive powers are not
independent contributions that can be added as causal percentages.

The correction and full eligible-population rerun are complete in Section 6. The
1,855-channel control diagnosed the failure; it was not a justification for
permanently discarding the other channels. The fixed-$b$ model and held APT calibration remain separate
scientific-validation questions. [R8]

\newpage

# Sweep-model quality: legacy tune.txt versus the full V3 model

**The large map-eligibility gap is principally downstream of fitting.**
Of 4,646 beammap-good channels, 4,495 (96.7%) have a fitted response and
4,409 (94.9%) have at least one optimizer-converged response. The map trial
admits only 3,096 (66.6%) to its scalar-map eligibility set. A converged response
is not by itself proof of a scientifically adequate fit. [R5]
""")
    sections.append(
        figure(
            "model-coverage",
            "Sweep coverage and probe selection are distinct from scalar-map eligibility. All counts use the original beammap-good population.",
        )
    )
    sections.append(
        figure(
            "model-exclusions",
            "Mutually exclusive causes of the 1,550-row gap. The 410 unusable flags do not mean 410 failed sweep fits.",
        )
    )
    sections.append(r"""
The 1,140 joint-required rows retain fitted models but the comparison has not
provided accepted component-to-beam bindings for their joint outputs. Of the
410 remaining exclusions, 151 have no fitted response, 104 have a selected
optimizer failure, 28 have probes outside fitted windows, and 127 have a
converged fit but insufficient simultaneous measurements for independent
component solving. Missing fits and probe-window gaps do not establish wrong
APT associations. A sweep can constrain a model that simultaneous timestream
measurements cannot uniquely solve.

\newpage

## Compare predictions on identical measured sweep samples

We evaluate each original legacy tune.txt model and each saved full V3
$M$-channel/$N$-resonator prediction at the same valid measured frequencies
in the saved response window. The V3 prediction includes nonlinear components
and the complex affine baseline. Recorded IQ accumulation agrees, and the
stored fitting samples are checked against the saved SweepReduce data.
No refit, gain rescaling, frequency shift or residual acceptance cut is applied.

The paired error is

\[
E=\frac{\sqrt{\langle|z-z_{\rm model}|^2\rangle}}
{\sqrt{\langle|z-\langle z\rangle|^2\rangle}}.
\]

The denominator is the measured complex-IQ excursion in that same window,
not the much larger IQ baseline. Each selected response enters once, including
map-ineligible channels. All alternative group/channel responses are also
retained in the evidence, without choosing the best residual. Missing selected
models remain in the coverage counts. [R5]
""")
    pairs = [quality["arrays"][str(i)]["selected_responses"] for i in range(3)]
    sections.append(
        "For a1100, a1400 and a2000 respectively, V3 has a smaller residual for "
        + ", ".join(
            f"**{p['v3_smaller_residual'] / p['finite_pairs']:.1%}**" for p in pairs
        )
        + " of paired channels. Median within-pair V3/legacy residual ratios are "
        + ", ".join(f"**{p['median_paired_ratio']:.3f}**" for p in pairs)
        + ". These are not ratios of population medians.\n\n"
    )
    sections.append(r"""
This is an **in-sample description test** with a more flexible V3 model, not
held-out validation, an optical-sensitivity measurement or proof of unique
solving. The corrected maps no longer show the original large a1100 excess.
Fit adequacy, native solving, component/APT identity and signal transfer remain
separate checks; better in-sample fits do not establish a sensitivity gain.

""")
    for array in ARRAYS:
        sections.append("\n\\newpage\n\n")
        sections.append(
            figure(
                f"{array}-sweep-quality",
                f"{array}: paired residuals and cumulative distributions on identical measured samples. Points below the diagonal favor V3. Shape distinguishes one versus several resonators; color describes downstream map status.",
            )
        )
    examples = read("timestream-examples/summary.json")
    sections.append(r"""
\newpage

## Follow adequate singlet and joint fits into measured timestreams

These examples use **152434 network 0**, with the second Tune block calibrating
all 7,864 measured timestream samples (64.42 s at 122.0703125 Hz). Before looking
at timestream noise, require every member channel to be beammap-good, have a
converged fit, no group warning and an excursion-normalized residual below
0.05. Choose the group closest to the median maximum channel residual within
each $M=N=1$ or $M=N=2$ pool. There are 329 singlet and 25 joint candidates.
This selects adequate examples, without selecting for low noise. [R6]

The maintained Python group solver consumes the complete saved affine/nonlinear
fit and all simultaneous probes. Group 20 (channel 297, component 22) has two
unknown states; group 404 (channels 99 and 100, components 448 and 449) has four.
Both recover every sample without flagged failures. Numerical validity does
not independently prove physical uniqueness. These joint diagnostic signals
**are not included in the saved maps**: accepted component-to-beam bindings
have not been supplied to that map trial.

For each component, $x'$ and $x_{\rm opt}$ use the recorded probe nearest its
fitted resonance. This coordinate choice is not a new optical/APT identity.
The legacy $x,r$ curves are effective channel coordinates, not separate
legacy solutions for each blended component. The legacy tune.txt $f_{\rm in}$
is 6,457.47 Hz below the recorded probe. We preserve that as-used convention,
rather than silently shifting the legacy model. IQ accumulation is normalized
by the recorded ratio of four in both paths.

Every trace subtracts only its own median. No amplitude matching or smoothing
is applied. One-sided Welch spectra use the same continuous samples, 1,024-sample
Hann windows, 50% overlap and constant detrending. The 5--20 Hz band summarizes
measured fluctuations before decorrelation, telescope processing and flux
calibration; it is not a noise-only NEFD. The $x_{\rm opt}$ correction retains
the fitted fixed-$b$ assumption, rather than measuring instantaneous
nonlinearity independently.
""")
    for case in examples["cases"]:
        name, group = case["name"], case["group"]
        sections.append(
            f"\n\\newpage\n\n### {name.title()} group {group['group_id']}: measured sweep\n\n"
        )
        sections.append(
            figure(
                f"timestream-examples/{name}-sweep",
                "The measured sweep, original legacy model and full V3 prediction for the exact channels followed in the next figures. No refit or gain rescaling.",
            )
        )
        sections.append(
            f"Selected maximum channel excursion-normalized residual: **{case['selection']['selected_max_channel_residual']:.4f}**. Selection and full fitted coefficients are retained in the evidence.\n\n"
        )
        if case["status"] != "complete":
            sections.append(f"No spectral comparison: {case['status']}.\n\n")
            continue
        for j, component in enumerate(case["components"]):
            sections.append(
                f"\n\\newpage\n\n### {name.title()} component {component['resonator_id']}: recovered signal and fluctuations\n\n"
            )
            sections.append(
                figure(
                    f"timestream-examples/{name}-{j}-recovery",
                    "Same measured samples: detuning, loss, PSD and detuning--loss coherence. The legacy channel coordinate is not a separated joint-component reference. No optical sensitivity claim follows from these PSDs.",
                )
            )
            asd = component["band_asd"]
            sections.append(
                f"5--20 Hz ASD (dimensionless per square-root Hz): legacy x **{asd[0]:.3g}**, V3 x' **{asd[1]:.3g}**, V3 optical detuning **{asd[2]:.3g}**. The latter/x' ratio is **{asd[2] / asd[1]:.3f}**. Fitted nonlinearity is **{component['a_nl']:.3f}** and Qr **{component['Qr']:.0f}**. Compare the signal-transfer model before interpreting a PSD increase as worse optical sensitivity.\n\n"
            )
    sections.append(
        "\n\\newpage\n\n### Joint-component correlations and loss-to-detuning propagation\n\n"
    )
    sections.append(
        figure(
            "timestream-examples/joint-noise",
            "Component-frequency coherence, followed by the signed contributions to optical-detuning PSD for each component. The cross term can cancel or reinforce the diagonal terms.",
            width="85%",
        )
    )
    sections.append(r"""
The local propagation is
$P_{x_{\rm opt}}\simeq D^2P_{x'}+\mu^2P_r+2D\mu\operatorname{Re}C_{x'r}$,
evaluated at the median recovered state. The signed cross term must be retained;
loss fluctuations are not assumed to be independent noise. Correlation between
components can include common loading and readout effects as well as coupling
through solving. It does not by itself measure cross-talk or optical identity.
All samples, flags, signed spectra and source hashes are downloadable in R6.
""")
    sections.append(r"""
\newpage

# Multiplicity, nonlinearity and quality of actual map contributors

The plots below replace the population tables. Each count is a contributing
acquired channel/APT row. The same final-block Python fit supplies diagnostic
labels for all populations; it does not describe the legacy solving algorithm.

The 90 V3 contributors in multi-resonator groups each have one nonzero component
coupling in the exported model. These are scalar outputs, not full blended
component recovery. The 83 channels in multi-probe singlet groups are also not
83 independent optical detectors.

A blended probe has multiple component $a_{\rm nl}$ values and receives no
scalar bin. The intervals are descriptive; every V3 fit includes nonlinearity.
Legacy asymmetry $A$ is not physical $a_{\rm nl}$.

In these historical captures, no numerical residual or conditioning cutoff
was enabled in the map replay. The current policy comparison keeps the same
initial eligibility but has different final sample acceptance and weights.
Warning-only models remain eligible; zero flags do not certify scientific
adequacy. The unusable state includes the different causes shown in Section 7.
""")
    for array in ARRAYS:
        sections.append("\n\\newpage\n\n")
        sections.append(
            figure(
                f"{array}-populations-print",
                f"{array}: group membership, fitted nonlinearity and model flags for the actual contributors. Exact counts and chunk identities remain in population-ledger.ecsv.",
            )
        )
    sections.append(r"""
The common-control and V3 maps share 1,871/643/498 contributors for the three
arrays. A further 12/4/0 contribute only in the legacy control and 11/5/0 only
in V3. This control retains initial eligibility, not identical final sample
masks or weights. It measures the model change together with downstream cuts.
""")
    for i, (array, m) in enumerate(zip(ARRAYS, metrics, strict=True)):
        sections.append(f"\n\\newpage\n\n# 152434: {array} map comparison\n\n")
        figpath = (evidence / f"{array}.pdf").as_posix()
        sections.append(
            f"![Unchanged sky-coordinate maps and V3 minus common-eligibility legacy. Intensity panels share a scale; residuals use a signed scale. No image registration or signal rescaling.]({figpath}){{width=100%}}\n\n"
        )
        sections.append(
            f"**Table {9 + i}. Native pointing-fit measurements retained for comparison.**\n\n"
        )
        p = m["pointing"]
        full = m["full_legacy_comparison"]["pointing"]
        rows = []
        for key, label in (
            ("amp", "Peak (mJy/beam)"),
            ("x_t", "Az offset (arcsec)"),
            ("y_t", "El offset (arcsec)"),
            ("a_fwhm", "Width a (arcsec)"),
            ("b_fwhm", "Width b (arcsec)"),
        ):
            rows.append(
                [
                    label,
                    f"{full[key]['left']:.3f}",
                    f"{p[key]['left']:.3f}",
                    f"{p[key]['right']:.3f}",
                ]
            )
        sections.append(
            table(["Quantity", "Legacy full", "Legacy eligible", "V3 models"], rows)
        )
        bg = m["background"]
        sections.append(
            f"Against legacy eligible: peak ratio **{p['amp']['right'] / p['amp']['left']:.3f}**; common-pixel correlation **{m['signal_correlation']:.3f}**; difference RMS **{m['difference_rms']:.2f} mJy/beam** within 40 arcsec. The 20--40 arcsec background RMS changes **{bg['legacy_rms']:.2f} to {bg['python_rms']:.2f} mJy/beam**.\n\n"
        )
        sections.append(
            "The a1400 fit reaches the default 12.6 arcsec upper width bound; these offsets and widths are not adopted as calibration.\n\n"
            if array == "a1400"
            else "Background fluctuations include correlated residual structure, not noise-only sensitivity or mapping speed.\n\n"
        )
        sections.append(
            "\n\\newpage\n\n## Representative retained joint sweep model\n\n"
        )
        sections.append(
            figure(
                f"{array}-sweep-example",
                "Measured sweep, original legacy prediction and full nonlinear V3 group prediction for both channels of a two-channel/two-resonator group. No fitting or rescaling is performed for the plot.",
            )
        )
        sections.append(
            "The example is chosen nearest the median paired residual ratio among APT-good selected M=2, N=2 responses excluded only from joint mapping, rather than as a best fit. The selection record gives network, group, channels and fitted nonlinearity. A retained sweep model is not evidence that its components have accepted APT identities or unique instantaneous solutions.\n\n"
        )
    sections.append(r"""
\newpage

# Handoff contents and remaining work

The 22 November local bundle includes coordinated Tolteca/Tollan wheels,
Python/C++ source snapshots, a comparison commit series, runtime constraints,
both 145609 block reports/stubs, compiled guides and validation receipts.
The 152434 work is a later local extension with native patches and receipts.
Its current engine additionally includes shared gap filling, mask propagation,
valid-sample PCA projection and explicit sample acceptance. These changes and
the current report recipes must accompany the refreshed package.

The full-array trial required a guard against estimating covariance from fewer
than two valid samples, and omission of unused joint-component outputs when
the APT has no component bindings. Independent channel results and flags were
retained; all seven previously completed control map files were numerically
unchanged after these fixes. These patches belong in the refreshed release,
not implicitly in the older wheel bundle. [R4]

## Reproduction

The frozen workflow guide gives the maintained 145609 replay and the exact
local 152434 corrected acceptance-control commands, followed by historical
coordinate/PCA controls. Trial directories are pinned;
repeating an invocation does not silently select its newest reduNN. The
strict native parity failure must remain visible.

Regenerate this PDF from the checkout containing the saved evidence:

~~~sh
python examples/tune_model_comparison/handoff/build_pdf.py \
  --root /path/to/tolteca \
  --output output/pdf/grant-tune-comparison-20261123.pdf
~~~

## Before a validated native release

1. Freeze the newer full-array code, native fixes and population audit in a
   refreshed source/wheel manifest, preserving unrelated work.
2. Resolve measured Python/native flags, validity and numerical-state differences
   without relaxing the verification gate.
3. Review the corrected gap/PCA path and explicit acceptance policy using the
   full-population controls in Section 6; validate signal transfer and the
   coordinate-dependent a1400 result before claiming sensitivity gains.
4. Establish accepted component-to-beam/APT bindings for multiplet maps and
   assess their transfer. Scalar probes in multiplet groups do not validate
   full component recovery.
5. Obtain Grant's target Citlali revision and NGC4449 observation/problem
   description; port and reproduce that specific comparison.

## Evidence references

**R1:** release-manifest.json, release-readme.md.
**R2:** full-replay.json, full-native-parity-{affine,delay}.json.
**R3:** populations.json, population-ledger.ecsv, native-coverage.json.
**R4:** metrics.json, per-array figures, native-build.json.
**R5:** sweep-model-quality.json, sweep-model-responses.ecsv,
sweep-model-channels.ecsv, sweep-example-selection.json and corresponding figures.
**R6:** timestream-examples/selection.json, summary.json, per-component spectra
and complete solved arrays and corresponding figures.
**R7:** xprime-control.json, xprime-rms.pdf, xprime-{array}.pdf;
xprime-control/ recipes, native patch and build record, and three execution
records with configs and input hashes.
**R8:** pca-trace.json, pca-flag-projection.json, pca-channel-chunks.ecsv;
PCA figures and NPZ spectra/trajectories; pca-control/ selection, shared APT,
recipes, regression tests, native source snapshot and four execution records.
**R9:** gap-control.json, map-quality-summary.json/pdf, gap-populations.pdf,
gap-maps.pdf, gap-trajectories.pdf/npz;
maintained native patches and build recipe, frozen build receipt, and the three
full-population execution records and configurations.

""")
    source = output.with_suffix(".md")
    gap = read("gap-control.json")
    rms = gap["maps"][0]["rms_mJy_beam"]
    gap_intro = (
        f"**Full-population correction:** with identical APT weights, a1100 RMS is "
        f"**{rms[0]:.2f}** for legacy numerical, **{rms[2]:.2f}** for V3 numerical, "
        f"and **{rms[1]:.2f} mJy/beam** for V3 physical acceptance. The "
        "numerical pair has identical sample masks, map weights and coverage. The corrected "
        "path fills gaps, preserves their map exclusion, and estimates PCA amplitudes "
        "using valid samples. Section 6 separates sample acceptance from the model comparison. [R9]\n\n"
    )
    summary = read("map-quality-summary.json")
    if summary["source"] != "gap-control.json":
        message = "Current map summary must use the corrected gap-control audit"
        raise ValueError(message)
    if (
        summary["source_sha256"]
        != hashlib.sha256((evidence / "gap-control.json").read_bytes()).hexdigest()
    ):
        message = "Current map summary is stale; regenerate gap_control_report.py"
        raise ValueError(message)
    content = "\n".join(sections).replace(
        "**Updated a1100 finding:**",
        gap_intro + "**Earlier matched-subset finding:**",
        1,
    )
    current = current_comparison(gap, summary, figure)
    if joint_path.exists():
        current = broader_comparison(
            read("joint-control.json"), figure
        ) + current.replace(
            "# 152434: current numerical and physical acceptance comparison",
            "## Corrected scalar controls: numerical and physical acceptance",
        ).replace("The current runs include", "These scalar controls include").replace(
            "They do not restore the missing joint-component APT",
            "They deliberately omit the later joint-component APT",
        ).replace(
            "These are\nthe current results", "These are\nthe scalar-control results"
        ).replace("Current controls, all using", "Scalar controls, all using").replace(
            "Current contributing channel/APT rows",
            "Scalar-control contributing channel/APT rows",
        )
        content = (
            content.replace(
                "The 1,140 joint-required rows retain fitted models but the comparison has not\nprovided accepted component-to-beam bindings for their joint outputs.",
                "The original scalar gate excluded 1,140 joint-required rows despite retaining fitted models. The broader comparison now admits 1,045 through saved component/channel correspondences; 95 remain unmatched.",
            )
            .replace(
                "**are not included in the saved maps**: accepted component-to-beam bindings\nhave not been supplied to that map trial.",
                "were absent from the original scalar maps. The broader comparison now supplies recorded component/channel/APT bindings where unambiguous; its ledger identifies exactly which components enter mapping.",
            )
            .replace(
                "4. Establish accepted component-to-beam/APT bindings for multiplet maps and\n   assess their transfer.",
                "4. Validate the inherited component beam/flux response and account for joint covariance in sensitivity estimates; investigate remaining unmatched identities.",
            )
            .replace(
                "the current report recipes must accompany the refreshed package.",
                "the mixed component/channel APT projection, selected-group solving and current report recipes must accompany the refreshed package.",
            )
        )
        content += "\n**R10:** joint-control.json, joint-map-summary.pdf, joint-maps.pdf; joint-control/ selection, channel disposition, component binding ledger, APT, native build patches and exact execution/configuration receipts.\n"
    content = assemble_document(content, current)
    source.write_text(content)
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        message = "Pandoc is required to build the review"
        raise RuntimeError(message)
    subprocess.run(  # noqa: S603 -- resolved executable, explicit local input, no shell
        [
            pandoc,
            str(source),
            "-o",
            str(output),
            "--pdf-engine=pdflatex",
            "--from=markdown+tex_math_single_backslash",
            "--number-sections",
            "--resource-path",
            str(evidence),
        ],
        check=True,
    )
    receipt["pdf"] = {
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    receipt["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    shutil.copy2(Path(__file__), evidence / "build_pdf.py")
    (evidence / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output)
    print(evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/grant-tune-comparison-20261123.pdf"),
    )
    build(**vars(parser.parse_args()))
