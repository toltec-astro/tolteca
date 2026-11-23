#!/usr/bin/env python
"""Prepare a complete acquired-network map comparison and its coverage ledger.

The input manifest explicitly names each Tune and target file. Both Tune blocks
are retained in one fitted product. The target headers select the calibration
block, and ordinary raw-context preparation writes one model stub per network.
The original APT is retained for the full reduction. A second legacy control
uses the same model-eligible channels, separating changed coverage from changed
coefficients. Joint components without accepted APT bindings remain explicit
identity gaps; this recipe never assigns them beam positions by proximity.
"""

import argparse
import copy
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

from tolteca_datamodels.data_prod import (
    DataProdIndex,
    DataProdStore,
    Location,
    WorkflowRun,
    build_raw_obs_batch,
)
from tolteca_datamodels.raw_obs_context import (
    ArtifactBinding,
    RawObsContext,
    RawObsDataItem,
    SweepCalItem,
    prepare_run_inputs,
)
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata
from tolteca_kids.model_binding import prepare_model_binding
from tolteca_kids.model_fit import ModelFit
from tolteca_kids.product import open_kids_product
from tolteca_kidsproc.model_flags import ModelFitFlag
from tolteca_reduce.stubs import materialize_model_stubs


def channel_ledger(apt, report):
    """Separate original APT exclusions, unusable fits and missing joint bindings.

    Warning-only fits remain eligible. The counts of individual flag bits can
    overlap; ``status`` is a mutually exclusive accounting of every channel.
    """
    np.testing.assert_array_equal(apt["kids_tone"], report["chan"])
    flags = np.asarray(report["flag"], dtype=np.uint32)
    unusable = (flags & int(ModelFitFlag.fit_unusable)) != 0
    joint = (flags & int(ModelFitFlag.fit_joint_required)) != 0
    good = np.asarray(apt["flag"]) == 0
    eligible = good & ~unusable & ~joint
    status = np.full(len(apt), "eligible", dtype="U40")
    status[joint] = "joint_apt_binding_missing"
    status[unusable] = "fit_unusable"
    status[~good] = "original_apt_excluded"
    ledger = Table(
        {
            "array": apt["array"],
            "nw": apt["nw"],
            "channel": report["chan"],
            "apt_uid": apt["uid"],
            "apt_flag": apt["flag"],
            "model_flag": flags,
            "fit_unusable": unusable,
            "joint_required": joint,
            "eligible": eligible,
            "status": status,
        }
    )
    summary = {
        "channels": len(apt),
        "apt_good": int(good.sum()),
        "fit_unusable": int(unusable.sum()),
        "joint_required": int(joint.sum()),
        "eligible": int(eligible.sum()),
        "eligible_with_warnings": int((eligible & (flags != 0)).sum()),
        "status": {str(s): int((status == s).sum()) for s in np.unique(status)},
        "flag_bits": {
            flag.name: int(((flags & int(flag)) != 0).sum())
            for flag in ModelFitFlag
            if ((flags & int(flag)) != 0).any()
        },
    }
    return ledger, summary


def prepare_context(manifest, model, store):
    """Use the maintained raw-context and artifact machinery for all networks."""
    files = [row[key] for row in manifest for key in ("tune", "timestream")]
    batch = build_raw_obs_batch(
        files, locations={"raw": Location(root="/", location_type="local")}
    )
    indexes = {}
    for index in batch.items:
        saved = DataProdIndex.read(store.write_index(index))
        indexes[saved.product.reference] = saved
    products = {
        i.product.uid: i.product
        for i in indexes.values()
        if i.product.meta.data_prod_type == "dp_raw_obs"
    }
    uids = set()
    for row in manifest:
        with open_toltec_kids(row["timestream"]) as raw:
            meta = raw.toltec_kids.meta
            if not isinstance(meta, ToltecRawObsMetadata):
                msg = "Manifest requires raw KIDs acquisitions"
                raise TypeError(msg)
            if meta.interface != row["interface"]:
                msg = "Manifest interface differs from acquisition header"
                raise ValueError(msg)
            uids.add(meta.uid_raw_obs)
    if len(uids) != 1:
        msg = "This comparison requires networks from one target acquisition"
        raise ValueError(msg)
    context = RawObsContext.from_product(
        products[uids.pop()],
        data_items=tuple(
            RawObsDataItem(interface=row["interface"], block=0) for row in manifest
        ),
        cal_products=products,
    )
    by_interface = {row["interface"]: row for row in manifest}
    for item in context.data_items:
        [sweep] = item.cal_items
        if not isinstance(sweep, SweepCalItem) or item.block is None:
            msg = "Each selected timestream block requires its recorded sweep"
            raise ValueError(msg)
        bound = prepare_model_binding(
            model,
            sweep=sweep,
            source=by_interface[item.interface]["tune"],
            producer="tolteca_kids.model_fit:ModelFit",
            name="model",
        )
        context = context.with_model(bound, interface=item.interface, block=item.block)
    indexes[model.product.reference] = model
    inputs = prepare_run_inputs([context], raw_products=products, model_indexes=indexes)
    with ExitStack() as stack, WorkflowRun.create(context={"inputs": inputs}) as run:
        stack.callback(store.write_run, run)
        context, stub_index = materialize_model_stubs(
            context, indexes=indexes, store=store, run=run
        )
    indexes[stub_index.product.reference] = stub_index
    inputs = prepare_run_inputs(
        [context],
        raw_products=products,
        model_indexes=indexes,
        artifact_indexes=indexes,
    )
    return context, stub_index, inputs


def plan_records(model, interface, array):
    """Read every proposed group, including those with insufficient fit data."""
    plans = []
    with open_kids_product(model, interface=interface) as tree:
        for block in tree.toltec_kids.blocks:
            quality = (
                block.tree["model_fit/plan_quality"].to_dataset(inherit=False).load()
            )
            for gid in np.unique(quality.group_id.values):
                selected = quality.isel(
                    row=np.flatnonzero(quality.group_id.values == gid)
                )
                plans.append(
                    {
                        "interface": interface,
                        "array": array,
                        "block": block.meta.block,
                        "group_id": str(gid),
                        "residual_dof": int(selected.fit_residual_dof.values[0]),
                        "status": str(selected.fit_plan_status.values[0]),
                        "n_channels": len(np.unique(selected.chan)),
                        "n_resonators": len(np.unique(selected.resonator_id)),
                    }
                )
    return plans


def fit_records(block, report, interface, array):
    """Count all fitted groups and components independently of fixed-probe routing."""
    view = ModelFit.view(block.tree)
    fit = view.fit.load()
    quality = block.tree["model_fit/quality"].to_dataset(inherit=False).load()
    routed = {g["group_id"]: g for g in report.meta["joint_models"]["groups"]}
    groups, components = [], []
    for gid in np.unique(fit.group_id.values):
        rows = fit.isel(row=np.flatnonzero(fit.group_id.values == gid))
        ids = np.unique(rows.resonator_id.values)
        group = {
            "interface": interface,
            "array": array,
            "block": block.meta.block,
            "group_id": str(gid),
            "n_channels": len(np.unique(rows.chan.values)),
            "n_resonators": len(ids),
            "fit_status": str(rows.fit_status.values[0]),
            "fit_quality_flag": int(
                quality.group_flag.values[
                    np.flatnonzero(quality.group_id.values == gid)[0]
                ]
            ),
            "routed": str(gid) in routed,
            "n_readout_channels": len(routed.get(str(gid), {}).get("channels", [])),
        }
        groups.append(group)
        for rid in ids:
            k = np.flatnonzero(rows.resonator_id.values == rid)[0]
            components.append(
                dict(
                    **group,
                    resonator_id=str(rid),
                    fr_hz=float(rows.fr.values[k]),
                    r=float(rows.r.values[k]),
                    a_nl=float(rows.a_nl.values[k]),
                )
            )
    return groups, components


def prepare(  # noqa: PLR0915 -- sequential preparation and saved coverage receipts
    config_path, manifest_path, model_index, output, *, dataprod_path=None
):
    """Prepare full-array configurations without fitting or solving again."""
    from astropy.table import vstack  # noqa: PLC0415

    output = output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    base = yaml.safe_load(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    model = DataProdIndex.read(model_index)
    store = DataProdStore(
        output / "dataprod_toltec" if dataprod_path is None else dataprod_path,
        namespace="python-affine",
    )
    context, stub_index, inputs = prepare_context(manifest, model, store)
    [entry] = base["inputs"]
    manifest_files = {r["interface"]: r["timestream"] for r in manifest}
    actual_files = {
        d["meta"]["interface"]: d["filepath"]
        for d in entry["data_items"]
        if d["meta"]["interface"].startswith("toltec")
    }
    if actual_files != manifest_files:
        msg = "Config and manifest must describe the same acquired networks"
        raise ValueError(msg)
    [apt_item] = [c for c in entry["cal_items"] if c["type"] == "array_prop_table"]
    apt = Table.read(apt_item["filepath"])
    common = apt.copy()
    ledgers, summaries, groups, components, plans = [], [], [], [], []
    for item in context.data_items:
        [sweep] = item.cal_items
        [cal] = sweep.data_items
        binding = ArtifactBinding(
            **cal.model.model_dump(),
            interface=item.interface,
            scope=f"blocks/{cal.block}",
        )
        report = Table.read(binding.resolve_path(model))
        nw = int(item.interface.removeprefix("toltec"))
        rows = np.flatnonzero(np.asarray(apt["nw"]) == nw)
        ledger, summary = channel_ledger(apt[rows], report)
        common["flag"][rows[~np.asarray(ledger["eligible"])]] = 1
        ledgers.append(ledger)
        summaries.append(
            dict(interface=item.interface, array=int(apt["array"][rows[0]]), **summary)
        )
        plans.extend(plan_records(model, item.interface, int(apt["array"][rows[0]])))
        with open_kids_product(model, interface=item.interface) as tree:
            for block in tree.toltec_kids.blocks:
                source = binding.model_copy(
                    update={"scope": f"blocks/{block.meta.block}"}
                )
                block_report = Table.read(source.resolve_path(model))
                gg, cc = fit_records(
                    block, block_report, item.interface, int(apt["array"][rows[0]])
                )
                groups.extend(gg)
                components.extend(cc)
    common_path = output / "common-apt.ecsv"
    common.write(common_path)
    vstack(ledgers).write(output / "channel-coverage.ecsv")
    Table(rows=groups).write(output / "group-coverage.ecsv")
    Table(rows=components).write(output / "component-coverage.ecsv")
    Table(rows=plans).write(output / "plan-coverage.ecsv")
    for name in ("python-full", "legacy-common"):
        config = copy.deepcopy(base)
        folder = output / name
        folder.mkdir()
        config["runtime"]["output_dir"] = str(folder)
        if name == "legacy-common":
            [target] = [
                c
                for c in config["inputs"][0]["cal_items"]
                if c["type"] == "array_prop_table"
            ]
            target["filepath"] = str(common_path)
        else:
            for data in config["inputs"][0]["data_items"]:
                interface = data["meta"]["interface"]
                if interface not in manifest_files:
                    continue
                data.pop("model_params", None)
                data["stubs"] = [
                    {
                        "filepath": str(s.artifact.resolve_path(stub_index)),
                        "mode": s.mode,
                    }
                    for s in context.stubs
                    if s.interface == interface
                ]
        (folder / "config.yaml").write_text(yaml.safe_dump(config))
    receipt = {
        "model_index": str(model_index.resolve()),
        "stub_index": str(stub_index.base_path / "index.yaml"),
        "context": context.model_dump(mode="json"),
        "inputs": inputs,
        "networks": summaries,
        "joint_identity_policy": (
            "No accepted model-component APT associations supplied; "
            "retain joint groups in calibration and report them as identity "
            "gaps, never approximate them as independent channels."
        ),
    }
    (output / "prepared.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    """Prepare explicit full-network reduction inputs and coverage receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config_path", "manifest_path", "model_index", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--dataprod-path", type=Path)
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
