"""Bind fitted components through saved KidsFind correspondences to the existing APT.

This comparison reuses an observation's accepted channel-to-APT relation. It
does not create beam positions or match new component frequencies to an APT.
Ambiguous saved finding/channel correspondences remain explicit exclusions.
"""

# ruff: noqa: INP001, E501 -- maintained local comparison recipe

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_kids.model_joint import joint_solver
from tolteca_kids.product import open_kids_product
from tolteca_kidsproc.model_flags import ModelFitFlag


def component_correspondences(detected, plan, pairs):
    """Follow a component's recorded finding ID and require a unique saved match."""
    if len(np.unique(detected["idx"])) != len(detected):
        message = "Finding IDs must be unique within the selected block"
        raise ValueError(message)
    findings = {str(int(r["idx"])): r for r in detected}
    result = {}
    for row in plan:
        rid = str(row["resonator_id"])
        finding = findings[rid]
        if (
            row["fr_initial"] != finding["f"]
            or row["r_initial"] != 1 / (2 * finding["Qr"])
            or str(row["finding_group_id"]) != str(int(finding["group"]))
        ):
            message = "Model plan is not bound to the saved finding"
            raise ValueError(message)
        edges = pairs[np.asarray(pairs["idx_det"]) == int(rid)]
        unique = (
            len(edges) == 1
            and edges[0]["n_matched_query"] == 1
            and edges[0]["n_matched_ref"] == 1
        )
        item = {
            "finding_id": rid,
            "status": "unique_saved_match" if unique else "ambiguous_saved_match",
        }
        if unique:
            match = edges[0]
            if (
                int(match["idx_chan"]) != int(finding["idx_chan"])
                or match["f_det"] != finding["f"]
            ):
                message = "Saved finding and match pair disagree"
                raise ValueError(message)
            item.update(
                channel=int(match["idx_chan"]),
                finding_frequency_hz=float(finding["f"]),
                probe_frequency_hz=float(match["f_chan"]),
                distance_hz=float(finding["f"] - match["f_chan"]),
            )
        result[(str(row["group_id"]), rid)] = item
    return result


def native_group(group):
    """Freeze the gain evaluated at the actual probes for native binding checks."""
    solver = joint_solver(group)
    result = copy.deepcopy(group)
    # The native APT contract validates the evaluated complex gain, while the
    # model stub retains its complete frequency-dependent representation.
    result["gain_re"] = solver.gain.real.tolist()
    result["gain_im"] = solver.gain.imag.tolist()
    return result


def bind_network(apt, common, report, detected, plan, pairs):  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917 -- ordered, audited eligibility decisions
    """Add supported components while preserving every scalar-control row."""
    np.testing.assert_array_equal(apt["kids_tone"], report["chan"])
    np.testing.assert_array_equal(common["kids_tone"], apt["kids_tone"])
    links = component_correspondences(detected, plan, pairs)
    groups = report.meta["joint_models"]["groups"]
    candidates, exclusions = [], []
    for group in groups:
        if len(group["resonator_ids"]) == 1:
            continue
        for rid in group["resonator_ids"]:
            row = {
                "group_id": group["group_id"],
                "resonator_id": rid,
                "n_channels": len(group["channels"]),
                "n_resonators": len(group["resonator_ids"]),
            }
            link = links.get((group["group_id"], rid))
            if link is None or link["status"] != "unique_saved_match":
                row["status"] = "finding_match_missing_or_ambiguous"
            else:
                row.update(link)
                channel = row["channel"]
                positions = np.flatnonzero(np.asarray(apt["kids_tone"]) == channel)
                if len(positions) != 1:
                    message = "APT must have one row per acquired channel"
                    raise ValueError(message)
                index = int(positions[0])
                row.update(
                    apt_row=index,
                    uid=int(apt["uid"][index]),
                    original_apt_flag=int(apt["flag"][index]),
                    model_flag=int(report["flag"][index]),
                )
                if apt["flag"][index] != 0:
                    row["status"] = "original_apt_excluded"
                elif common["flag"][index] == 0:
                    row["status"] = "retained_scalar_control"
                elif group["fit_flag"] & int(ModelFitFlag.fit_unusable):
                    row["status"] = "group_fit_unusable"
                elif len(group["channels"]) < len(group["resonator_ids"]):
                    row["status"] = "insufficient_simultaneous_probes"
                elif channel not in group["channels"]:
                    row["status"] = "probe_routed_to_another_group"
                elif report["flag"][index] & int(ModelFitFlag.fit_unusable):
                    row["status"] = "channel_model_unusable"
                elif not report["flag"][index] & int(ModelFitFlag.fit_joint_required):
                    row["status"] = "not_joint_required"
                else:
                    row["status"] = "candidate"
                    candidates.append(row)
            if row["status"] != "candidate":
                exclusions.append(row)
    # A candidate or APT identity cannot be independently reused in two groups.
    owners = Counter(r["uid"] for r in candidates)
    component_owners = Counter(r["resonator_id"] for r in candidates)
    accepted = []
    for row in candidates:
        if owners[row["uid"]] > 1 or component_owners[row["resonator_id"]] > 1:
            row["status"] = "competing_group_binding"
            exclusions.append(row)
        else:
            row["status"] = "joint_added"
            accepted.append(row)
    chosen = {r["group_id"] for r in accepted}
    payload = {
        "nw": int(apt["nw"][0]),
        "rows": [
            {k: r[k] for k in ("uid", "group_id", "resonator_id", "channel")}
            for r in accepted
        ],
        "channel_rows": [
            {"apt_row": i, "uid": int(r["uid"]), "channel": int(r["kids_tone"])}
            for i, r in enumerate(apt)
            if int(r["uid"]) not in {a["uid"] for a in accepted}
        ],
        "groups": [native_group(g) for g in groups if g["group_id"] in chosen],
    }
    return payload, accepted + exclusions


def prepare(base, output):  # noqa: PLR0915 -- preparation and provenance receipt
    """Create the broader APT, policy configs and complete correspondence receipt."""
    base, output = base.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((base / "prepared/prepared.json").read_text())
    model = DataProdIndex.read(receipt["model_index"])
    source = yaml.safe_load((base / "prepared/python-full/config.yaml").read_text())
    [item] = [
        r for r in source["inputs"][0]["cal_items"] if r["type"] == "array_prop_table"
    ]
    original_path = Path(item["filepath"])
    original = Table.read(original_path)
    apt = Table.read(base / "prepared/common-apt.ecsv")
    apt["uid"] = np.asarray(apt["uid"], dtype=np.int64)
    networks, ledger = [], []
    for nw in np.unique(apt["nw"]).astype(int):
        mask = np.asarray(apt["nw"]) == nw
        with open_kids_product(model, interface=f"toltec{nw}") as product:
            block = product.toltec_kids.blocks[1]

            def table(key, tree=block.tree):
                return Table.from_pandas(
                    tree[key]
                    .to_dataset(inherit=False)
                    .to_dataframe()
                    .reset_index(drop=True)
                )

            report_path = (
                model.base_path
                / f"artifacts/toltec{nw}/blocks/1/tolteca_kids.model_fit:ModelFit/kids_model.ecsv"
            )
            report = Table.read(report_path)
            network, rows = bind_network(
                original[mask],
                apt[mask],
                report,
                table("kids_find/detected"),
                table("model_fit/plan"),
                table("kids_find/matched/pairs"),
            )
        networks.append(network)
        indices = np.flatnonzero(mask)
        for row in rows:
            row.update(nw=int(nw), array=int(apt["array"][indices[0]]))
            if row["status"] == "joint_added":
                i = indices[row["apt_row"]]
                apt["flag"][i] = 0
                apt["tone_freq"][i] = row["probe_frequency_hz"]
        ledger.extend(rows)
    apt.meta["kids_readout"] = yaml.safe_dump(
        {"schema": "kids.apt_readout/1", "networks": networks}, sort_keys=False
    )
    apt.meta["joint_binding_evidence"] = (
        "Saved ModelFit plan -> KidsFind finding -> unique saved channel match -> existing observation APT UID; no new beam positions."
    )
    apt_path = output / "joint-apt.ecsv"
    apt.write(apt_path)
    (output / "component-bindings.json").write_text(json.dumps(ledger, indent=2) + "\n")
    Table(rows=ledger).write(output / "component-bindings.ecsv")
    legacy_apt = apt.copy(copy_data=True)
    legacy_apt.meta.pop("kids_readout")
    legacy_apt.meta.pop("joint_binding_evidence")
    legacy_apt.meta["comparison_eligibility"] = (
        "Same accepted APT rows as the joint-component comparison; legacy per-channel solving."
    )
    legacy_path = output / "legacy-apt.ecsv"
    legacy_apt.write(legacy_path)
    legacy_config = yaml.safe_load(
        (base / "gap-control/legacy/source.yaml").read_text()
    )
    for item in legacy_config["inputs"][0]["cal_items"]:
        if item["type"] == "array_prop_table":
            item["filepath"] = str(legacy_path)
    legacy_config["runtime"]["output_dir"] = str(output / "legacy")
    (output / "legacy.yaml").write_text(yaml.safe_dump(legacy_config, sort_keys=False))
    configurations = {"legacy": str(output / "legacy.yaml")}
    for policy in ("numerical", "physical"):
        config = yaml.safe_load(
            (base / f"gap-control/v3-{policy}/source.yaml").read_text()
        )
        [item] = [
            r
            for r in config["inputs"][0]["cal_items"]
            if r["type"] == "array_prop_table"
        ]
        item["filepath"] = str(apt_path)
        config["runtime"]["output_dir"] = str(output / policy)
        config_path = output / f"{policy}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        configurations[policy] = str(config_path)
    summary = {
        "model_index": receipt["model_index"],
        "stub_index": receipt["stub_index"],
        "original_apt": str(original_path),
        "original_apt_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "configurations": configurations,
        "networks": [],
        "status": dict(Counter(r["status"] for r in ledger)),
    }
    for n in networks:
        rows = [r for r in ledger if r["nw"] == n["nw"]]
        summary["networks"].append(
            {
                "nw": n["nw"],
                "added_components": len(n["rows"]),
                "selected_groups": len(n["groups"]),
                "status": dict(Counter(r["status"] for r in rows)),
            }
        )
    (output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    prepare(**vars(parser.parse_args()))
