#!/usr/bin/env python
"""Hold the map inputs fixed while changing the selected Tune model.

This first comparison uses only one-channel/one-resonator groups with an
existing usable APT row and zero model flags. It does not assign a multiplet
component to a beam position. Both maps use exactly the same channel mask,
constant weights, and the original flux conversion; absolute flux calibration
of the new model is not established by this experiment.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_datamodels.raw_obs_context import ArtifactBinding


def prepare(config, model_index, prepared, output):
    """Resolve the exact model artifact and write a common-selection map pair."""
    base = yaml.safe_load(config.read_text())
    index = DataProdIndex.read(model_index)
    receipt = json.loads(prepared.read_text())
    [item] = receipt["context"]["data_items"]
    [sweep] = item["cal_items"]
    [cal] = sweep["data_items"]
    binding = ArtifactBinding.model_validate(
        {
            **cal["model"],
            "interface": item["interface"],
            "scope": f"blocks/{cal['block']}",
        }
    )
    report_path = binding.resolve_path(index)
    report = Table.read(report_path)
    [entry] = base["inputs"]
    [apt_item] = [c for c in entry["cal_items"] if c["type"] == "array_prop_table"]
    apt = Table.read(apt_item["filepath"])
    nw = int(item["interface"].removeprefix("toltec"))
    apt = apt[apt["nw"] == nw]
    np.testing.assert_array_equal(apt["kids_tone"], report["chan"])
    one = {
        g["channels"][0]
        for g in report.meta["joint_models"]["groups"]
        if len(g["channels"]) == len(g["resonator_ids"]) == 1
    }
    good = np.asarray(apt["flag"]) == 0
    single = np.isin(report["chan"], list(one))
    unflagged = np.asarray(report["flag"]) == 0
    selected = good & single & unflagged
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    apt["flag"][~selected] = 1
    apt.write(output / "shared-singlet-apt.ecsv")
    apt_item["filepath"] = str(output / "shared-singlet-apt.ecsv")
    entry["data_items"] = [
        d
        for d in entry["data_items"]
        if d["meta"]["interface"] in ["lmt", item["interface"]]
    ]
    [old] = [
        d for d in entry["data_items"] if d["meta"]["interface"] == item["interface"]
    ]
    if old["filepath"] != receipt["data_item"]["filepath"]:
        msg = "Prepared model targets a different raw acquisition"
        raise ValueError(msg)
    base["timestream"]["polarimetry"]["ignore_hwpr"] = True
    for name in ["raw_time_chunk", "processed_time_chunk"]:
        for key in ["lower_tod_inv_var_factor", "upper_tod_inv_var_factor"]:
            base["timestream"][name]["flagging"][key] = 0.0
    base["timestream"]["processed_time_chunk"]["weighting"].update(
        type="const",
        lower_map_weight_factor=0.0,
        upper_map_weight_factor=0.0,
        median_map_weight_factor=0.0,
    )
    for name in ["legacy-singlets", "python-singlets"]:
        config_data = copy.deepcopy(base)
        folder = output / name
        folder.mkdir()
        config_data["runtime"]["output_dir"] = str(folder)
        if name == "python-singlets":
            config_data["inputs"][0]["data_items"] = [
                receipt["data_item"]
                if d["meta"]["interface"] == item["interface"]
                else d
                for d in config_data["inputs"][0]["data_items"]
            ]
        (folder / "config.yaml").write_text(yaml.safe_dump(config_data))
    selection = {
        "model_index": str(model_index),
        "report": str(report_path),
        "prepared": str(prepared),
        "acquired_channels": len(apt),
        "legacy_apt_good": int(good.sum()),
        "single_channel_single_resonator": int(single.sum()),
        "single_with_zero_model_flags": int((single & unflagged).sum()),
        "common_selected": int(selected.sum()),
        "selected_channels": np.asarray(report["chan"])[selected].tolist(),
        "calibration": sweep,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2))


def main():
    """Prepare an explicit matched model comparison without refitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "model_index", "prepared", "output"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
