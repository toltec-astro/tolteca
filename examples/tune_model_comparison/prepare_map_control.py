#!/usr/bin/env python
"""Prepare two Citlali configurations with exactly the same legacy calibration.

Supply a fully resolved, local single-input Citlali configuration and its saved
v2 fit-report directory. The v3 config selects each model explicitly. Original
model coefficients are checked against the APT before adding channel labels.
No reduction or detector-model fit is implemented here.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table


def prepare(config_path, output, *, single_pass=False):
    """Write matched configs and an input-hash/model-equivalence receipt."""
    config = yaml.safe_load(config_path.read_text())
    if len(config["inputs"]) != 1:
        msg = "Provide one explicitly resolved observation input"
        raise ValueError(msg)
    entry = config["inputs"][0]
    [apt_item] = [c for c in entry["cal_items"] if c["type"] == "array_prop_table"]
    apt_path = Path(apt_item["filepath"])
    apt = Table.read(apt_path)
    fit_dir = Path(config["kids"]["solver"]["fitreportdir"])
    sources = [Path(d["filepath"]) for d in entry["data_items"]] + [apt_path]
    for path in sources:
        if not path.is_absolute() or not path.is_file():
            msg = f"Input must be an existing absolute local path: {path}"
            raise ValueError(msg)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    models = output / "legacy-models"
    models.mkdir()
    config["runtime"].update(n_threads=1, parallel_policy="seq")
    if single_pass:
        config["timestream"]["fruit_loops"]["enabled"] = False
    v3 = copy.deepcopy(config)
    checks, reports = label_reports(v3["inputs"][0], apt, fit_dir, models)
    sources.extend(reports)
    v3["kids"]["solver"].pop("fitreportdir")
    for name, value in [("v2", config), ("v3-legacy-model", v3)]:
        folder = output / name
        folder.mkdir()
        value["runtime"]["output_dir"] = str(folder)
        (folder / "config.yaml").write_text(yaml.safe_dump(value))
    hashes = {}
    for path in sources:
        with path.open("rb") as stream:
            hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "purpose": "legacy-model engine control",
        "source_config": str(config_path),
        "single_pass": single_pass,
        "serial": True,
        "model_equivalence": checks,
        "input_hashes": hashes,
    }
    (output / "control-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def label_reports(entry, apt, fit_dir, models):
    """Validate original coefficients and add explicit channel identities."""
    checks = []
    sources = []
    for item in entry["data_items"]:
        interface = item["meta"]["interface"]
        if not interface.startswith("toltec"):
            continue
        # The caller supplies the exact calibration directory. Multiple reports
        # are an error, never an implicit choice of latest or nearest sweep.
        reports = list(fit_dir.glob(f"{interface}_*_tune.txt"))
        if len(reports) != 1:
            msg = f"Expected one exact calibration report for {interface}: {reports}"
            raise ValueError(msg)
        report = Table.read(reports[0], format="ascii.ecsv")
        rows = apt[apt["nw"] == int(interface.removeprefix("toltec"))]
        for column in report.colnames:
            np.testing.assert_array_equal(report[column], rows["kids_" + column])
        channels = np.asarray(rows["kids_tone"], dtype=np.int64)
        np.testing.assert_array_equal(channels, rows["kids_tone"])
        if len(np.unique(channels)) != len(report):
            msg = "APT must contain one unique channel per legacy model row"
            raise ValueError(msg)
        report["chan"] = channels
        target = models / reports[0].name
        report.write(target, format="ascii.ecsv")
        item["model_params"] = {"filepath": str(target)}
        sources.append(reports[0])
        checks.append(
            {
                "interface": interface,
                "channels": len(report),
                "original": str(reports[0]),
                "labeled": str(target),
                "coefficient_values_identical": True,
            }
        )
    return checks, sources


def main():
    """Prepare configurations without running either engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--single-pass", action="store_true")
    args = parser.parse_args()
    prepare(args.config, args.output, single_pass=args.single_pass)


if __name__ == "__main__":
    main()
