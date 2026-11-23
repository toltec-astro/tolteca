"""Trace 152434 a1100 through native PCA with matched channels and weights.

Prepare captures, freeze their shared fully valid a1100 population, then run
both model paths with that same APT. No fit or native algorithm is changed.
"""

# ruff: noqa: INP001, E501 -- scientific diagnostic constants

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table
from build_xprime_control import digest
from netCDF4 import Dataset
from run_xprime_control import make_config

MODELS = {"legacy": "legacy-common", "v3": "python-full"}


def capture_config(source, output, apt_path=None):
    """Save native stages and eigenvectors; disable coordinate-dependent cuts."""
    config = make_config(source, output, variance_cuts=False)
    for stage in ("raw_time_chunk", "processed_time_chunk"):
        config["timestream"][stage]["output"]["enabled"] = True
    config["timestream"]["output"]["stats"]["eigenvalues"] = True
    if apt_path is not None:
        [apt] = [
            item
            for item in config["inputs"][0]["cal_items"]
            if item["type"] == "array_prop_table"
        ]
        apt["filepath"] = str(apt_path.resolve())
    return config


def prepare_capture(base):
    """Write the two native capture configs from their existing model controls."""
    for name, source in MODELS.items():
        folder = base / "pca-control/capture" / name
        folder.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load(
            (base / "prepared" / source / "config.yaml").read_text()
        )
        (folder / "source.yaml").write_text(
            yaml.safe_dump(capture_config(config, folder), sort_keys=False)
        )


def stage_path(folder, stage):
    """Require one native output, never silently choose a later realization."""
    [path] = (folder / "redu00/152434/raw").glob(f"*_{stage}_timestream.nc")
    return path


def channel_health(path):
    """Assess every saved sample, without interpolating or hiding rejected rows."""
    with Dataset(path) as data:
        nw = np.asarray(data["apt_nw"][:], dtype=int)
        channel = np.asarray(data["apt_kids_tone"][:], dtype=int)
        finite = np.ones(len(nw), dtype=bool)
        valid = finite.copy()
        # Slice rows to avoid retaining all networks' large diagnostic arrays.
        for start in range(0, data["signal"].shape[0], 512):
            section = slice(start, start + 512)
            signal = np.ma.filled(data["signal"][section], np.nan)
            flags = np.ma.filled(data["flags"][section], 1)
            finite &= np.isfinite(signal).all(axis=0)
            valid &= (flags == 0).all(axis=0)
        return {
            "nw": nw,
            "channel": channel,
            "finite": finite,
            "valid": valid,
            "n_samples": data["signal"].shape[0],
        }


def matched_selection(apt, health):
    """Select fully valid channels in both paths; retain exclusion evidence."""
    eligible = (np.asarray(apt["array"]) == 0) & (np.asarray(apt["flag"]) == 0)
    selected = eligible.copy()
    ledger = Table(
        {"nw": apt["nw"], "channel": apt["kids_tone"], "a1100_eligible": eligible}
    )
    for name, h in health.items():
        np.testing.assert_equal(h["nw"], apt["nw"])
        np.testing.assert_equal(h["channel"], apt["kids_tone"])
        ledger[f"{name}_finite"] = h["finite"]
        ledger[f"{name}_all_samples_valid"] = h["valid"]
        selected &= h["finite"] & h["valid"]
    ledger["selected"] = selected
    return selected, ledger


def prepare_matched(base):
    """Use one APT and freeze an a1100 sample-validity intersection."""
    root = base / "pca-control"
    apt = Table.read(base / "prepared/common-apt.ecsv")
    health = {}
    sources = {}
    for name in MODELS:
        folder = root / "capture" / name
        receipt = json.loads((folder / "redu00/execution.json").read_text())
        assert receipt["exit_code"] == 0
        for stage in ("rtc", "ptc"):
            path = stage_path(folder, stage)
            key = f"{name}_{stage}"
            health[key] = channel_health(path)
            sources[key] = {"path": str(path), "sha256": digest(path)}
    assert len({h["n_samples"] for h in health.values()}) == 1
    selected, ledger = matched_selection(apt, health)
    output = apt.copy(copy_data=True)
    # Keep the other arrays' common eligibility; this experiment selects a1100.
    target = np.asarray(apt["array"]) == 0
    output["flag"][target] = np.where(selected[target], 0, 1)
    apt_path = root / "matched-apt.ecsv"
    output.write(apt_path, overwrite=True)
    ledger.write(root / "channel-selection.ecsv", overwrite=True)
    counts = [
        {
            "network": nw,
            "eligible": int(((apt["nw"] == nw) & ledger["a1100_eligible"]).sum()),
            "selected": int(((apt["nw"] == nw) & selected).sum()),
        }
        for nw in range(6)
    ]
    receipt = {
        "sources": sources,
        "counts": counts,
        "samples": next(iter(health.values()))["n_samples"],
        "apt": {"path": str(apt_path), "sha256": digest(apt_path)},
        "rule": "a1100 common APT eligibility and finite, unflagged at every native pre/post-PCA sample in both captures; no residual or noise cut",
    }
    (root / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for name, source in MODELS.items():
        folder = root / "matched" / name
        folder.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load(
            (base / "prepared" / source / "config.yaml").read_text()
        )
        (folder / "source.yaml").write_text(
            yaml.safe_dump(capture_config(config, folder, apt_path), sort_keys=False)
        )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-capture", "prepare-matched"))
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    if args.action == "prepare-capture":
        prepare_capture(args.base.resolve())
    else:
        prepare_matched(args.base.resolve())
