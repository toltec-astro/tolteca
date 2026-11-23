"""Audit map contributors against the executed APT and selected fitted models.

Counts describe acquired channel/APT rows, not unique optical detectors. The
Python fit supplies diagnostic labels for *all* populations, including legacy
contributors; those labels do not describe the model used by the legacy solver.
"""

# ruff: noqa: INP001 -- shared implementation for standalone comparison recipes

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.raw_obs_context import ArtifactBinding
from tolteca_kids.product import open_kids_product
from tolteca_kidsproc.model_flags import ModelFitFlag

POPULATIONS = {
    "legacy": "Full legacy",
    "common": "Legacy with v3 eligibility",
    "python": "Python models",
}
GROUP_KINDS = {
    "singlet_one_probe": "One resonator, one probe",
    "singlet_multiple_probes": "One resonator, several probes",
    "multiplet": "Several resonators in selected group",
    "unresolved": "No selected group",
}
NONLINEARITY_BINS = ("[0,0.1)", "[0.1,0.5)", "[0.5,1)", "[1,infinity)")


def a_bin(value):
    """Describe nonlinearity without imposing a scientific acceptance cut."""
    if not np.isfinite(value) or value < 0:
        return "unresolved"
    return NONLINEARITY_BINS[np.searchsorted([0.1, 0.5, 1.0], value, side="right")]


def describe_channels(report, quality):
    """Distinguish group multiplicity from components coupled to each probe."""
    lookup = {}
    for group in report.meta["joint_models"]["groups"]:
        for i, channel in enumerate(group["channels"]):
            if channel in lookup:
                msg = f"Channel {channel} belongs to multiple selected groups"
                raise ValueError(msg)
            lookup[channel] = group, i
    rows = {}
    for source in report:
        channel = int(source["chan"])
        row = {
            "group_id": "",
            "group_kind": "unresolved",
            "n_readout_probes": 0,
            "n_group_resonators": 0,
            "n_channel_components": 0,
            "channel_kind": "unresolved",
            "resonator_id": "",
            "scalar_a_nl": float(source["a_nl"]),
            "a_nl_bin": "unresolved",
            "group_residual_fraction": np.nan,
            "group_reduced_chi2": np.nan,
        }
        rows[channel] = row
        if channel not in lookup:
            continue
        group, i = lookup[channel]
        coupling = (
            np.asarray(group["coupling_re"])[i]
            + 1j * np.asarray(group["coupling_im"])[i]
        )
        active = np.flatnonzero(coupling != 0)
        n_res = len(group["resonator_ids"])
        n_probes = len(group["channels"])
        row.update(
            group_id=group["group_id"],
            group_kind=(
                "multiplet"
                if n_res > 1
                else "singlet_multiple_probes"
                if n_probes > 1
                else "singlet_one_probe"
            ),
            n_readout_probes=n_probes,
            n_group_resonators=n_res,
            n_channel_components=len(active),
        )
        if len(active) > 1:
            row.update(channel_kind="blended", a_nl_bin="blended")
        elif len(active) == 1:
            j = active[0]
            # Check the independent row is the actual component, not the first
            # component of an arbitrary multi-resonator group.
            if not int(source["flag"]) & int(ModelFitFlag.fit_joint_required):
                np.testing.assert_equal(source["a_nl"], group["a_nl"][j])
                np.testing.assert_equal(source["fr"], group["fr"][j])
            row.update(
                channel_kind="single_component",
                resonator_id=group["resonator_ids"][j],
                a_nl_bin=a_bin(float(source["a_nl"])),
            )
        metrics = quality[group["group_id"]]
        row.update(
            group_residual_fraction=metrics["residual_fraction"],
            group_reduced_chi2=metrics["reduced_chi2"],
        )
    return rows


def model_details(receipt):
    """Read the exact calibration block selected by the prepared raw context."""
    index = DataProdIndex.read(receipt["model_index"])
    descriptions, sources = {}, []
    for item in receipt["context"]["data_items"]:
        [sweep] = [c for c in item["cal_items"] if c["type"] == "sweep"]
        [cal] = sweep["data_items"]
        interface, block_id = item["interface"], cal["block"]
        binding = ArtifactBinding(
            **cal["model"], interface=interface, scope=f"blocks/{block_id}"
        )
        path = binding.resolve_path(index)
        model = Table.read(path)
        with open_kids_product(index, interface=interface) as tree:
            [block] = [b for b in tree.toltec_kids.blocks if b.meta.block == block_id]
            quality = block.tree["model_fit/quality"].to_dataset(inherit=False).load()
        metrics = {
            str(gid): {
                field: float(quality[field].values[i])
                for field in ("residual_fraction", "reduced_chi2")
            }
            for i, gid in enumerate(quality.group_id.values)
        }
        nw = int(interface.removeprefix("toltec"))
        descriptions.update(
            {(nw, c): d for c, d in describe_channels(model, metrics).items()}
        )
        sources.append(
            {
                "interface": interface,
                "block": block_id,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "thresholds": model.meta["quality_thresholds"],
                "acceptance": model.meta["calibration_acceptance"],
            }
        )
    return descriptions, sources


def changed_columns(left, right):
    """Compare values, units and masks without treating NaN as a difference."""
    if left.colnames != right.colnames or len(left) != len(right):
        msg = "APT columns or row count changed"
        raise ValueError(msg)
    changed = []
    for name in left.colnames:
        a, b = left[name], right[name]
        equal = np.array_equal(
            np.asarray(a), np.asarray(b), equal_nan=a.dtype.kind in "fc"
        )
        equal &= np.array_equal(np.ma.getmaskarray(a), np.ma.getmaskarray(b))
        if not equal or a.unit != b.unit:
            changed.append(name)
    return changed


def apt_audit(folders, ledger):
    """Verify source APT equality and the exact common-eligibility mutation."""
    tables, sources = {}, {}
    for name, folder in folders.items():
        config_path = folder.parents[1] / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        [entry] = config["inputs"]
        [cal] = [c for c in entry["cal_items"] if c["type"] == "array_prop_table"]
        path = Path(cal["filepath"])
        apt = Table.read(path)
        tables[name] = apt
        sources[name] = {
            "config": str(config_path.resolve()),
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(apt),
            "flag_zero": int(sum(apt["flag"] == 0)),
        }
    base, common = tables["legacy"], tables["common"]
    if sources["legacy"]["sha256"] != sources["python"]["sha256"]:
        msg = "Full legacy and Python reductions did not use the identical APT"
        raise ValueError(msg)
    changed = changed_columns(base, common)
    if set(changed) - {"flag"} or base.meta != common.meta:
        msg = "Common APT changed more than its eligibility flag"
        raise ValueError(msg)
    lookup = {(int(r["nw"]), int(r["channel"])): r for r in ledger}
    if len(lookup) != len(ledger) or len(ledger) != len(base):
        msg = "APT/channel ledger has missing or duplicate identities"
        raise ValueError(msg)
    for i, apt_row in enumerate(base):
        row = lookup[int(apt_row["nw"]), int(apt_row["kids_tone"])]
        assert row["apt_flag"] == apt_row["flag"]
        assert row["apt_uid"] == apt_row["uid"]
        assert bool(row["eligible"]) == (common["flag"][i] == 0)
    return {"sources": sources, "common_changed_columns": changed}


def count_values(values):
    """Count disjoint labels, retaining zero-count categories at presentation."""
    labels, counts = np.unique(values, return_counts=True)
    return dict(zip(map(str, labels), map(int, counts), strict=True))


def summarize_population(rows):
    """Describe actual channel rows, with fit metrics counted once per group."""
    flags = np.asarray(rows["model_flag"], dtype=np.uint32)
    state = np.full(len(rows), "warnings_only", dtype="U20")
    state[flags == 0] = "no_flags"
    state[(flags & int(ModelFitFlag.fit_joint_required)) != 0] = "joint_required"
    state[(flags & int(ModelFitFlag.fit_unusable)) != 0] = "unusable"
    groups = {(int(r["nw"]), str(r["group_id"])): r for r in rows if r["group_id"]}
    metrics = {}
    for field in ("group_residual_fraction", "group_reduced_chi2"):
        values = np.asarray([r[field] for r in groups.values()])
        values = values[np.isfinite(values)]
        metrics[field] = {
            "groups_with_finite_value": len(values),
            "median_p90": np.percentile(values, [50, 90]).tolist()
            if len(values)
            else [],
        }
    return {
        "channels": len(rows),
        "apt_good": int(sum(rows["apt_flag"] == 0)),
        "apt_bad": int(sum(rows["apt_flag"] != 0)),
        "group_kind": count_values(rows["group_kind"]),
        "channel_kind": count_values(rows["channel_kind"]),
        "a_nl_bin": count_values(rows["a_nl_bin"]),
        "quality_state": count_values(state),
        "flag_bits": {
            f.name: int(sum((flags & int(f)) != 0))
            for f in ModelFitFlag
            if np.any(flags & int(f))
        },
        "unique_selected_groups": len(groups),
        "group_metrics": metrics,
    }


def population_audit(ledger, descriptions, run_stats):
    """Join channel identities, model properties and saved map contributions."""
    rows = []
    used = {
        name: {
            (r["nw"], c): chunks
            for r in stats
            for c, chunks in zip(
                r["contributing_channel_ids"], r["contributing_chunk_ids"], strict=True
            )
        }
        for name, stats in run_stats.items()
    }
    for row in ledger:
        key = int(row["nw"]), int(row["channel"])
        record = {name: row[name] for name in ledger.colnames}
        record.update(descriptions[key])
        for name in POPULATIONS:
            chunks = used[name].get(key, [])
            record[f"{name}_used"] = bool(chunks)
            record[f"{name}_chunks"] = len(chunks)
            record[f"{name}_chunk_ids"] = json.dumps(chunks)
        rows.append(record)
    result = Table(rows=rows)
    summary = {}
    for array in np.unique(ledger["array"]):
        subset = result[result["array"] == array]
        common, python = subset["common_used"], subset["python_used"]
        summary[str(int(array))] = {
            "acquired": len(subset),
            "apt_good": int(sum(subset["apt_flag"] == 0)),
            "apt_bad": int(sum(subset["apt_flag"] != 0)),
            "eligibility": count_values(subset["status"]),
            "populations": {
                name: summarize_population(subset[subset[f"{name}_used"]])
                for name in POPULATIONS
            },
            "common_python_overlap": {
                "both": int(sum(common & python)),
                "common_only": int(sum(common & ~python)),
                "python_only": int(sum(~common & python)),
            },
        }
    return result, summary
