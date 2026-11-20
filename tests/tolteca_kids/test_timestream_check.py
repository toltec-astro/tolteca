"""Raw-noise parity and shared single-block workflow, persistence and figures."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.signal import welch
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids import (
    BlockCreate,
    TimestreamCheck,
    TimestreamCheckPlot,
    TimestreamCheckView,
)
from tolteca_kids.runner import run_data
from tolteca_kids.timestream_check_viz import TimestreamNoisePSDViz

from .test_sweep_check_plot_parity import _assert_equal, _decode


def _allow_roundoff(actual, expected):
    """Check numeric plot arrays at the same tolerance as saved spectra."""
    if not isinstance(expected, dict):
        return
    for key, value in expected.items():
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            np.testing.assert_allclose(actual[key], value, rtol=2e-13, atol=0)
            actual[key] = value
        elif isinstance(value, dict):
            _allow_roundoff(actual[key], value)
        elif isinstance(value, list):
            for a, e in zip(actual[key], value, strict=True):
                _allow_roundoff(a, e)


@pytest.mark.parametrize(
    ("obsnum", "suffix", "block"),
    [
        ("145605", "vnasweep", 0),
        ("145485", "targsweep", 0),
        ("145609", "tune", 0),
        ("145609", "tune", 1),
    ],
)
def test_noise_values_and_figure_equal_v2(
    data_lmt_path, tmp_path, obsnum, suffix, block
):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON for production-v2 comparison")
    source = next(
        (data_lmt_path / "toltec/tcs/toltec0").glob(f"*{obsnum}*_{suffix}.nc")
    )
    subprocess.run(  # noqa: S603 -- selected local production interpreter
        [
            interpreter,
            str(Path(__file__).with_name("v2_noise_psd_reference.py")),
            str(source),
            str(tmp_path),
            str(block),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with open_toltec_kids(source) as raw:
        tree = BlockCreate()(ensure_datatree(raw))
        scope = tree.toltec_kids.blocks[block].tree
        TimestreamCheck()(scope)
        result = TimestreamCheck.view(scope)
        result.data_source.to_zarr(
            tmp_path / "check.zarr", zarr_format=2, consolidated=True
        )
    with xr.open_datatree(
        tmp_path / "check.zarr", engine="zarr", consolidated=True
    ) as saved:
        view = TimestreamCheckView(saved)
        assert view.config == result.config
        assert view.psd.psd_analysis == result.config.psd
        actual = {
            "f_psd": view.psd.f_psd,
            "f_psd_mask": view.normalized.band_mask,
            "I_psd": view.psd.psd.I,
            "Q_psd": view.psd.psd.Q,
            "Sphi": view.normalized.Sphi,
            "chan_I_psd_med": view.psd.psd_median.I,
            "chan_I_psd_mad_std": view.psd.psd_mad_std.I,
            "chan_Q_psd_med": view.psd.psd_median.Q,
            "chan_Q_psd_mad_std": view.psd.psd_mad_std.Q,
            "chan_Sphi_med": view.normalized.Sphi_median,
            "chan_Sphi_mad_std": view.normalized.Sphi_mad_std,
        }
        with np.load(tmp_path / "v2.npz") as expected:
            for name, values in actual.items():
                np.testing.assert_allclose(
                    values, expected[name], rtol=2e-13, atol=0, err_msg=name
                )
        fig = TimestreamNoisePSDViz()(view)
        fig.write_json(tmp_path / "v3.json")
        fig.write_html(tmp_path / "noise_psd.html")
    actual = _decode(json.loads(fig.to_json()))
    expected = _decode(json.loads((tmp_path / "v2.json").read_text()))
    for a, e in zip(actual["data"], expected["data"], strict=True):
        if a["type"] == "scattergl":
            assert e["type"] == "scatter"
            a["type"] = "scatter"
    for obj in (actual, expected):
        template = obj["layout"].pop("template")
        obj["layout"]["colorway"] = template["layout"]["colorway"]
    _allow_roundoff(actual, expected)
    _assert_equal(actual, expected)


def test_timestream_route_publication_and_alias(timestream_path, tmp_path):
    with open_toltec_kids(timestream_path) as raw:
        selected = raw.isel(time=slice(0, 512)).toltec_kids.select_channels([10, 0, 5])
        tree = run_data(
            selected,
            timestream_path,
            RuntimeContext(
                {
                    "kids": {
                        "timestream_check": {"enabled": True},
                        "timestream_check_plot": {"enabled": True},
                        "output": {
                            "enabled": True,
                            "path": str(tmp_path),
                            "namespace": "test",
                        },
                    }
                }
            ),
        )
        scope = tree.toltec_kids.blocks[0].tree
        check = TimestreamCheck.view(scope)
        _, expected = welch(
            selected["Data.Toltec.Is"].values.T, fs=122.0703125, nperseg=512
        )
        np.testing.assert_array_equal(check.psd.psd.I, expected)
        alias = TimestreamCheck(alias="short", psd={"nperseg": 128})
        alias(scope)
        assert TimestreamCheck.view(scope, alias="short").psd.f_psd.size == 65
        assert check.psd.f_psd.size == 257
        TimestreamCheckPlot(check_alias="short", save_path=tmp_path / "plots")(scope)
        assert (tmp_path / "plots/noise_psd.html").is_file()
    stores = list(tmp_path.rglob("array.zarr"))
    assert len(stores) == 1
    with xr.open_datatree(stores[0], engine="zarr") as saved:
        restored = TimestreamCheck.view(saved["toltec0/blocks/0"])
        np.testing.assert_array_equal(restored.psd.psd.I, expected)
        fig = TimestreamNoisePSDViz()(restored, channels=[0, 2])
        np.testing.assert_array_equal(fig.data[0].x, [10, 5])
    assert any(tmp_path.glob("test/**/artifacts/**/noise_psd.html"))
