"""Run a full map control without changing fitted models or source artifacts."""

# ruff: noqa: INP001, S603, T201 -- reproducible local command receipt

import argparse
import copy
import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml
from build_xprime_control import digest


def make_config(source, output, *, variance_cuts=True):
    """Keep native settings; optionally disable both variance-dependent cuts."""
    config = copy.deepcopy(source)
    config["runtime"]["output_dir"] = str(output.resolve())
    if not variance_cuts:
        for stage in ("raw_time_chunk", "processed_time_chunk"):
            flags = config["timestream"][stage]["flagging"]
            flags["lower_tod_inv_var_factor"] = 0.0
            flags["upper_tod_inv_var_factor"] = 0.0
    return config


def run(binary, config, output, *, variance_cuts=True):
    """Preserve config, command, input hashes, output directory and exit status."""
    binary, config, output = binary.resolve(), config.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contents = make_config(
        yaml.safe_load(config.read_text()), output, variance_cuts=variance_cuts
    )
    target = output / "config.yaml"
    # Existing runs must retain their exact configuration.
    if target.exists() and yaml.safe_load(target.read_text()) != contents:
        msg = "Existing run configuration differs; choose a new output directory"
        raise ValueError(msg)
    target.write_text(yaml.safe_dump(contents, sort_keys=False))
    paths = set()

    def collect(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "filepath":
                    paths.add(Path(v))
                else:
                    collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)

    collect(contents["inputs"])
    previous = set(output.glob("redu*"))
    receipt = {
        "argv": [str(binary), str(target)],
        "binary_sha256": digest(binary),
        "source_config_sha256": digest(config),
        "executed_config_sha256": digest(target),
        "variance_cuts": variance_cuts,
        "input_hashes": {str(p): digest(p) for p in sorted(paths)},
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    with (output / "citlali.log").open("w") as log:
        result = subprocess.run(
            receipt["argv"], stdout=log, stderr=subprocess.STDOUT, check=False
        )
    new = set(output.glob("redu*")) - previous
    receipt.update(
        exit_code=result.returncode,
        completed_at=dt.datetime.now(dt.UTC).isoformat(),
        realizations=sorted(str(p) for p in new),
    )
    destination = next(iter(new)) if len(new) == 1 else output
    (destination / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in receipt.items() if k != "input_hashes"}, indent=2)
    )
    result.check_returncode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "config", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument(
        "--no-variance-cuts", dest="variance_cuts", action="store_false"
    )
    run(**vars(parser.parse_args()))
