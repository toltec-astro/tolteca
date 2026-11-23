#!/usr/bin/env python
"""Fit a recorded Tune, prepare model stubs and map an explicit observation.

The recipient supplies a working Citlali configuration and its matched APT.
All science calculation is delegated to maintained workflow/native steps.
"""

# ruff: noqa: E501, S603, T201 -- standalone handoff recipe

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table, vstack

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from prepare_array_map_control import channel_ledger, prepare_context  # noqa: E402
from prepare_joint_map_control import bind_network  # noqa: E402
from run_xprime_control import run as run_native  # noqa: E402

from tolteca_datamodels.data_prod import (  # noqa: E402
    DataProdIndex,
    DataProdStore,
    resolve_index,
)
from tolteca_datamodels.raw_obs_context import ArtifactBinding  # noqa: E402
from tolteca_datamodels.toltec import open_toltec_kids  # noqa: E402
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata  # noqa: E402
from tolteca_kids.product import open_kids_product  # noqa: E402


def digest(path):
    """Hash a source without loading an acquisition into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_filepaths(value, directory):
    """Resolve explicit Citlali filepath entries relative to the source config."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "filepath":
                path = Path(item).expanduser()
                path = path if path.is_absolute() else directory / path
                if not path.is_file():
                    raise FileNotFoundError(path)
                value[key] = str(path.resolve())
            else:
                resolve_filepaths(item, directory)
    elif isinstance(value, list):
        for item in value:
            resolve_filepaths(item, directory)


def acquisition_manifest(config, tunes, obsnum=None):  # noqa: C901, PLR0912 -- sequential header validation
    """Validate header identities and pair each network with its supplied Tune."""
    if len(config.get("inputs", [])) != 1:
        msg = "This comparison recipe requires one acquisition input"
        raise ValueError(msg)
    indexed = {}
    tune_uids, target_uids = set(), set()
    for source in tunes:
        path = source.expanduser().resolve()
        with open_toltec_kids(path) as raw:
            meta = raw.toltec_kids.meta
            if not isinstance(meta, ToltecRawObsMetadata) or meta.uid_raw_obs is None:
                msg = f"Expected raw calibration acquisition metadata: {path}"
                raise ValueError(msg)
            key = meta.interface
            if key in indexed:
                msg = f"More than one supplied calibration file for {key}"
                raise ValueError(msg)
            if meta.obs_type not in ("tune", "targ"):
                msg = f"Expected a recorded Tune/target sweep: {path}"
                raise ValueError(msg)
            tune_uids.add(meta.uid_raw_obs)
            indexed[key] = str(path)
    rows = []
    for item in config["inputs"][0]["data_items"]:
        interface = item["meta"]["interface"]
        if not interface.startswith("toltec"):
            continue
        if any(r["interface"] == interface for r in rows):
            msg = f"Repeated acquisition interface: {interface}"
            raise ValueError(msg)
        with open_toltec_kids(item["filepath"]) as raw:
            meta = raw.toltec_kids.meta
            if not isinstance(meta, ToltecRawObsMetadata) or meta.uid_raw_obs is None:
                msg = "Expected raw target acquisition metadata"
                raise ValueError(msg)
            if meta.interface != interface or (
                obsnum is not None and meta.obsnum != obsnum
            ):
                msg = "Config/obsnum disagrees with acquisition header"
                raise ValueError(msg)
            target_uids.add(meta.uid_raw_obs)
        if interface not in indexed:
            msg = f"Missing recorded calibration file for {interface}"
            raise ValueError(msg)
        rows.append(
            {
                "interface": interface,
                "tune": indexed[interface],
                "timestream": item["filepath"],
            }
        )
    if len(tune_uids) != 1 or len(target_uids) != 1 or not rows:
        msg = "Supply one calibration acquisition and one target acquisition"
        raise ValueError(msg)
    if set(indexed) != {r["interface"] for r in rows}:
        msg = "Supplied Tune networks must equal the selected target networks"
        raise ValueError(msg)
    return rows, next(iter(tune_uids)), next(iter(target_uids))


def validate_reference(config, *, also_legacy=False):
    """Reject unsupported map quantities and misleading legacy comparisons."""
    quantity = config.get("timestream", {}).get("type", "xs")
    if quantity not in ("xs", "rs"):
        msg = "Component mapping requires a solved timestream type: xs or rs"
        raise ValueError(msg)
    if also_legacy and any(
        item.get("stubs")
        for entry in config.get("inputs", [])
        for item in entry.get("data_items", [])
        if item.get("meta", {}).get("interface", "").startswith("toltec")
    ):
        msg = "--also-legacy requires the original legacy-model input config, without KIDs stubs"
        raise ValueError(msg)


def bound_apt(original, context, model):
    """Use recorded calibration blocks and saved finding/channel correspondences."""
    apt = original.copy(copy_data=True)
    apt["uid"] = np.asarray(apt["uid"], dtype=np.int64)
    networks, components, channels = [], [], []
    expected = {int(i.interface.removeprefix("toltec")) for i in context.data_items}
    if set(np.unique(apt["nw"]).astype(int)) != expected:
        msg = "APT networks must match the selected acquired networks"
        raise ValueError(msg)
    for item in context.data_items:
        nw = int(item.interface.removeprefix("toltec"))
        [sweep] = item.cal_items
        [cal] = sweep.data_items
        binding = ArtifactBinding(
            **cal.model.model_dump(),
            interface=item.interface,
            scope=f"blocks/{cal.block}",
        )
        report = Table.read(binding.resolve_path(model))
        indices = np.flatnonzero(np.asarray(apt["nw"]) == nw)
        source = original[indices]
        channel_rows, _ = channel_ledger(source, report)
        common = source.copy(copy_data=True)
        common["flag"][
            (np.asarray(common["flag"]) == 0) & ~np.asarray(channel_rows["eligible"])
        ] = 1
        apt["flag"][indices] = common["flag"]
        with open_kids_product(model, interface=item.interface) as product:
            block = product.toltec_kids.blocks[cal.block]

            def table(key, tree=block.tree):
                return Table.from_pandas(
                    tree[key]
                    .to_dataset(inherit=False)
                    .to_dataframe()
                    .reset_index(drop=True)
                )

            network, ledger = bind_network(
                source,
                common,
                report,
                table("kids_find/detected"),
                table("model_fit/plan"),
                table("kids_find/matched/pairs"),
            )
        for row in ledger:
            row.update(
                nw=nw, array=int(source["array"][0]), calibration_block=int(cal.block)
            )
            if row["status"] == "joint_added":
                index = indices[row["apt_row"]]
                apt["flag"][index] = 0
                apt["tone_freq"][index] = row["probe_frequency_hz"]
                channel_rows["status"][row["apt_row"]] = "joint_added"
                channel_rows["eligible"][row["apt_row"]] = True
        networks.append(network)
        components.extend(ledger)
        channels.append(channel_rows)
    apt.meta["kids_readout"] = yaml.safe_dump(
        {"schema": "kids.apt_readout/1", "networks": networks}, sort_keys=False
    )
    apt.meta["joint_binding_evidence"] = (
        "Saved ModelFit plan -> KidsFind finding -> unique saved channel match -> existing observation APT UID; no new beam positions."
    )
    return apt, components, vstack(channels)


def prepare(  # noqa: C901, PLR0913, PLR0915 -- explicit preparation and provenance
    config_path,
    tunes,
    output,
    *,
    model_index=None,
    apt_path=None,
    obsnum=None,
    workers=4,
    gain_family="affine",
    acceptance="physical",
    variance_cuts=True,
):
    """Prepare a portable product tree and exact native config in a fresh folder."""
    config_path, output = config_path.resolve(), output.resolve()
    config = yaml.safe_load(config_path.read_text())
    validate_reference(config)
    resolve_filepaths(config.get("inputs", []), config_path.parent)
    manifest, tune_uid, target_uid = acquisition_manifest(config, tunes, obsnum)
    [apt_item] = [
        c for c in config["inputs"][0]["cal_items"] if c["type"] == "array_prop_table"
    ]
    if apt_path is not None:
        apt_item["filepath"] = str(apt_path.expanduser().resolve(strict=True))
    original_apt = Path(apt_item["filepath"])
    original = Table.read(original_apt)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "input-config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "acquisitions.json").write_text(json.dumps(manifest, indent=2) + "\n")
    store = DataProdStore(output / "dataprod_toltec", namespace="comparison")
    if model_index is None:
        cli = Path(sys.executable).with_name("tolteca_kids")
        if not cli.is_file():
            msg = "Install the coordinated tolteca_kids command in this Python environment"
            raise FileNotFoundError(msg)
        fit = yaml.safe_load((HERE.parent / "fit.yaml").read_text())
        fit["kids"]["model_fit"]["analysis"]["gain_family"] = gain_family
        fit["kids"]["output"].update(
            path=str(output / "dataprod_toltec"), namespace="comparison"
        )
        fit_path = output / "fit.yaml"
        fit_path.write_text(yaml.safe_dump(fit, sort_keys=False))
        argv = [
            str(cli),
            "--config_paths",
            str(fit_path),
            "--",
            *[r["tune"] for r in manifest],
        ]
        env = dict(
            os.environ,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            DASK_SCHEDULER="processes",
            DASK_NUM_WORKERS=str(workers),
        )
        with (output / "logs/fit.log").open("w") as log:
            subprocess.run(
                argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        model_index = resolve_index(store.get_index_path(tune_uid, "dp_raw_obs_proc"))
    model = DataProdIndex.read(model_index)
    context, stub, inputs = prepare_context(manifest, model, store)
    apt, components, channels = bound_apt(original, context, model)
    apt_path = output / "apt.ecsv"
    apt.write(apt_path)
    channels.write(output / "channels.ecsv")
    (output / "components.json").write_text(json.dumps(components, indent=2) + "\n")
    if components:
        Table(rows=components).write(output / "components.ecsv")
    legacy_apt = apt.copy(copy_data=True)
    legacy_apt.meta.pop("kids_readout")
    legacy_apt.meta.pop("joint_binding_evidence")
    legacy_apt.write(output / "legacy-apt.ecsv")
    legacy = copy.deepcopy(config)
    for entry in legacy["inputs"][0]["cal_items"]:
        if entry["type"] == "array_prop_table":
            entry["filepath"] = str(output / "legacy-apt.ecsv")
    (output / "legacy.yaml").write_text(yaml.safe_dump(legacy, sort_keys=False))
    apt_item["filepath"] = str(apt_path)
    for item in config["inputs"][0]["data_items"]:
        interface = item["meta"]["interface"]
        if not interface.startswith("toltec"):
            continue
        item.pop("model_params", None)
        item["stubs"] = [
            {"filepath": str(s.artifact.resolve_path(stub)), "mode": s.mode}
            for s in context.stubs
            if s.interface == interface
        ]
        if not item["stubs"]:
            msg = f"Missing prepared model stub for {interface}"
            raise ValueError(msg)
    config.setdefault("kids", {}).setdefault("solver", {})["sample_acceptance"] = (
        acceptance
    )
    config["runtime"].update(output_dir=str(output / "maps"), use_subdir=True)
    if not variance_cuts:
        for stage in ("raw_time_chunk", "processed_time_chunk"):
            flags = config["timestream"][stage]["flagging"]
            flags.update(lower_tod_inv_var_factor=0.0, upper_tod_inv_var_factor=0.0)
    (output / "citlali.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    receipt = {
        "schema": "tune-comparison/1",
        "target_uid": target_uid,
        "tune_uid": tune_uid,
        "model_index": str(Path(model_index).resolve()),
        "stub_index": str(stub.base_path / "index.yaml"),
        "inputs": inputs,
        "context": context.model_dump(mode="json"),
        "acceptance": acceptance,
        "variance_cuts": variance_cuts,
        "apt_source": {"path": str(original_apt), "sha256": digest(original_apt)},
        "apt_eligible": int((np.asarray(apt["flag"]) == 0).sum()),
        "channel_status": dict(Counter(map(str, channels["status"]))),
        "components": dict(Counter(r["status"] for r in components)),
        "config_source": {"path": str(config_path), "sha256": digest(config_path)},
        "status": "prepared",
    }
    (output / "handoff-run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    """Run on user-supplied paths; no built-in observation or host paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Working native Citlali config with matched APT and explicit raw files",
    )
    parser.add_argument(
        "--tune",
        type=Path,
        nargs="+",
        required=True,
        help="Recorded calibration files for every selected network",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New comparison directory"
    )
    parser.add_argument(
        "--model-index",
        type=Path,
        help="Reuse an exact fitted product instead of refitting",
    )
    parser.add_argument(
        "--apt",
        type=Path,
        help="Original observation-matched beammap APT, if the source config used a restricted comparison APT",
    )
    parser.add_argument(
        "--obsnum",
        type=int,
        help="Optional assertion against target acquisition headers",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gain-family", choices=["affine", "delay"], default="affine")
    parser.add_argument(
        "--acceptance", choices=["physical", "numerical"], default="physical"
    )
    parser.add_argument(
        "--no-variance-cuts",
        action="store_true",
        help="Reproduce the matched-weight 152434 comparison setting",
    )
    parser.add_argument(
        "--citlali", type=Path, help="Coordinated comparison executable"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--also-legacy",
        action="store_true",
        help="Run legacy coefficients from the input config with the same broader eligibility",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    validate_reference(
        yaml.safe_load(args.config.read_text()), also_legacy=args.also_legacy
    )
    if not args.prepare_only and (args.citlali is None or not args.citlali.is_file()):
        parser.error("Supply --citlali, or use --prepare-only")
    receipt = prepare(
        args.config,
        args.tune,
        args.output,
        model_index=args.model_index,
        apt_path=args.apt,
        obsnum=args.obsnum,
        workers=args.workers,
        gain_family=args.gain_family,
        acceptance=args.acceptance,
        variance_cuts=not args.no_variance_cuts,
    )
    if not args.prepare_only:
        try:
            run_native(
                args.citlali,
                args.output / "citlali.yaml",
                args.output / "maps",
                variance_cuts=not args.no_variance_cuts,
            )
            if args.also_legacy:
                run_native(
                    args.citlali,
                    args.output / "legacy.yaml",
                    args.output / "legacy-maps",
                    variance_cuts=not args.no_variance_cuts,
                )
        except subprocess.CalledProcessError:
            receipt["status"] = "native_failed"
            (args.output / "handoff-run.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            raise
        receipt["status"] = "completed"
        (args.output / "handoff-run.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("inputs", "context")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
