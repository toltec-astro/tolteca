#!/usr/bin/env python
"""Check saved population accounting, signal projection and immutable inputs."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

UNIT_RESPONSE_ATOL = 1e-12
SUMMARY_BAND_HZ = (5, 20)


def verify(root, output=None, *, direction_column="beta_at_probe"):
    """Verify the real comparison independently of its rendered figures."""
    output = root / "active-response" if output is None else output
    sources = json.loads((output / "sources.json").read_text())
    for record in sources.values():
        assert (
            hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
            == record["sha256"]
        )
    hashes = json.loads((root / "notation-before/science-sha256.json").read_text())
    source_root = root
    if (root / "replay-source.json").exists():
        source_root = Path(
            json.loads((root / "replay-source.json").read_text())["source_root"]
        )
    changed = [
        path
        for path, checksum in hashes.items()
        if hashlib.sha256((source_root / path).read_bytes()).hexdigest() != checksum
    ]
    assert not changed, changed
    table = pd.read_csv(
        output / "signal-noise.csv", dtype={"group_id": str, "resonator_id": str}
    )
    assert not table.duplicated(["family", "group_id", "resonator_id"]).any()
    selected = table[table.population == "active_suitable"]
    assert not selected.duplicated(["family", "apt_uid"]).any()
    excluded = table[table.population != "active_suitable"]
    assert excluded[direction_column].isna().all()
    assert excluded.projected_to_x_asd.isna().all()
    assert (selected.assumed_signal_transfer.sub(1).abs() < UNIT_RESPONSE_ATOL).all()
    methods = runpy.run_path(str(Path(__file__).with_name("network_response.py")))
    max_error = 0.0
    checked = 0
    for family, folder in (("Affine", "coverage"), ("Delay", "coverage-delay")):
        measured = root / folder / "measured"
        lookup = selected[selected.family == family].set_index(
            ["group_id", "resonator_id"]
        )
        rate = json.loads((measured / "output-quality-method.json").read_text())[
            "sample_rate_hz"
        ]
        for base, x, r in methods["component_records"](measured):
            key = base["group_id"], base["resonator_id"]
            if key not in lookup.index:
                continue
            row = lookup.loc[key]
            assert x is not None
            assert r is not None
            split = len(x) // 2
            # Project the held-out time series itself, independently of the
            # report's quadratic form of its spectral covariance.
            projected = row.weight_x * x[split:] + row.weight_r * r[split:]
            freq, power = welch(
                projected,
                fs=rate,
                window="hann",
                nperseg=1024,
                noverlap=512,
                detrend="constant",
                scaling="density",
            )
            band = (freq >= SUMMARY_BAND_HZ[0]) & (freq <= SUMMARY_BAND_HZ[1])
            actual = np.sqrt(power[band].mean())
            error = abs(actual / row.projected_asd_5_20_hz - 1)
            max_error = max(error, max_error)
            np.testing.assert_allclose(actual, row.projected_asd_5_20_hz, rtol=1e-11)
            checked += 1
    result = {
        "scientific_files_checked": len(hashes),
        "scientific_files_changed": changed,
        "source_files_checked": len(sources),
        "components": len(table),
        "independent_time_domain_projections": checked,
        "maximum_relative_projection_error": max_error,
        "excluded_components_have_no_optical_projection": True,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(verify(parser.parse_args().root.resolve()), indent=2))  # noqa: T201
