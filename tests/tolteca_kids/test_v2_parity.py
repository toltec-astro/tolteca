"""Exact reader-level parity through the configured pipeline and persisted view."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from tollan.config import RuntimeContext

from tolteca_datamodels.toltec.filestore import build_input_catalog
from tolteca_kids.product import open_sweep_product
from tolteca_kids.runner import run_source
from tolteca_kids.sweep_reduce import SweepReduceView


@pytest.mark.parametrize(
    ("suffix", "block"),
    [
        ("targsweep", -1),
        ("vnasweep", -1),
        ("tune", -1),
        ("tune", 0),
    ],
)
def test_persisted_values_equal_v2(data_lmt_path, tmp_path, suffix, block):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON to enable cross-environment parity tests")
    sources = sorted((data_lmt_path / "toltec/tcs/toltec0").glob(f"*_{suffix}.nc"))
    if not sources:
        pytest.skip(f"No {suffix} fixture")
    reference = tmp_path / "v2.npz"
    subprocess.run(  # noqa: S603 -- explicit local comparison interpreter
        [
            interpreter,
            str(Path(__file__).with_name("v2_sweep_reference.py")),
            str(sources[0]),
            str(reference),
            str(block),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    runtime = RuntimeContext({"kids": {"sweep_reduce": {"block": block}}})
    runtime = runtime.with_config(
        overrides={
            "kids": {
                "output": {"enabled": True, "path": str(tmp_path), "name": "product"}
            }
        }
    )
    info = build_input_catalog([sources[0]]).toltec_file.to_info_list()[0]
    product = run_source(info, runtime)
    assert product is not None
    with np.load(reference) as expected, open_sweep_product(product) as reopened:
        view = SweepReduceView(reopened).selected_block
        for name in ("I", "Q", "unc_I", "unc_Q", "f_chan", "sweep", "frequency"):
            actual = getattr(view, name)
            assert actual is not None, name
            np.testing.assert_array_equal(actual.values, expected[name], err_msg=name)
        for name, unit_v2 in zip(
            ("I", "Q", "unc_I", "unc_Q"), expected["units"], strict=True
        ):
            unit = getattr(view, name).u.unit
            assert (u.dimensionless_unscaled if unit is None else unit) == u.Unit(
                unit_v2
            )
        frequency = view.frequency
        assert frequency is not None
        assert frequency.attrs["units"] == "Hz"
