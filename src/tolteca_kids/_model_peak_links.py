"""Exact finding membership behind fitted components; no frequency inference."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np
import xarray as xr
from astropy.table import MaskedColumn, Table


def finding_signature(finding):
    """Fingerprint the measured peaks, configuration and explicit membership."""
    peaks = finding.d21_peaks.peaks.copy()
    peaks["bitmask"] = finding.d21_peaks.bitmask
    content = {"config": finding.config.model_dump(mode="json")}
    for name, table in (
        ("peaks", peaks),
        ("membership", finding.det_grouped),
        ("detected", finding.detected),
    ):
        content[name] = {
            key: [
                str(getattr(table[key], "unit", None)),
                np.asarray(table[key]).tolist(),
            ]
            for key in table.colnames
        }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def make_peak_links(finding, fit):
    """Record all D21 peaks against components of an automatic finding plan."""
    membership, detected = finding.det_grouped, finding.detected
    if not {"subdet", "peak_id"}.issubset(membership.colnames):
        msg = "Finding lacks original peak membership; rerun KidsFind before ModelFit"
        raise ValueError(msg)
    components = {}
    for group, rid in zip(fit.group_id.values, fit.resonator_id.values, strict=True):
        pair = (str(group), str(rid))
        if str(rid) in components and components[str(rid)] != pair:
            msg = "A finding component has multiple fitted group owners"
            raise ValueError(msg)
        components[str(rid)] = pair
    candidates = {int(row["idx"]): components.get(str(row["idx"])) for row in detected}
    by_peak, by_detection = defaultdict(set), defaultdict(set)
    for row in membership:
        if row["subdet"] == "d21":
            peak, detection = int(row["peak_id"]), int(row["group"])
            by_peak[peak].add(detection)
            by_detection[detection].add(peak)
    rows = []
    for peak in finding.d21_peaks.peaks["idx"]:
        detections = by_peak[int(peak)]
        if not detections:
            rows.append((int(peak), "", "", "unresolved", "peak not retained"))
        for detection in sorted(detections):
            pair = candidates.get(detection)
            if pair is None:
                rows.append((int(peak), "", "", "unresolved", "no fitted component"))
                continue
            reason = (
                "peak belongs to multiple detections"
                if len(detections) > 1
                else "multiple D21 peaks in detection"
                if len(by_detection[detection]) > 1
                else ""
            )
            rows.append(
                (int(peak), *pair, "unresolved" if reason else "linked", reason)
            )
    names = ("peak_id", "group_id", "resonator_id", "status", "reason")
    table = Table(rows=rows, names=names, dtype=(int, str, str, str, str))
    return xr.DataTree(
        dataset=xr.Dataset(
            {name: ("row", np.asarray(table[name])) for name in names},
            attrs={"finding_signature": finding_signature(finding)},
        )
    )


def read_peak_links(tree):
    """Materialize persisted links, masking absent component coordinates."""
    table = Table({name: var.values for name, var in tree.data_vars.items()})
    for name in ("group_id", "resonator_id"):
        table[name] = MaskedColumn(table[name], mask=table[name] == "")
    table.meta["finding_signature"] = tree.attrs["finding_signature"]
    return table
