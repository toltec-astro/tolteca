"""Figure contracts against production v2, including pages and persisted input."""

import base64
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree

from tolteca_kids import SweepCheck, SweepCheckPlot, SweepReduce
from tolteca_kids.sweep_check_view import SweepCheckView
from tolteca_kids.sweep_check_viz import (
    ChannelSummaryViz,
    IQGridViz,
    RoachReadoutPowerViz,
    S21FrequencyGridViz,
    S21FrequencyViz,
    SweepCheckArtifacts,
)

FIGURES = {
    "chan_summary": ChannelSummaryViz,
    "s21_f": S21FrequencyViz,
    "s21_f_grid": S21FrequencyGridViz,
    "i_q_grid": IQGridViz,
}


def _decode(value):
    if isinstance(value, dict):
        if "bdata" in value:
            array = np.frombuffer(
                base64.b64decode(value["bdata"]), dtype=value["dtype"]
            )
            if "shape" in value:
                array = array.reshape(tuple(map(int, value["shape"].split(","))))
            return array
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _assert_equal(actual, expected, path="figure"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            _assert_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list) and any(
        isinstance(item, (list, dict)) for item in expected
    ):
        assert len(actual) == len(expected), path
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _assert_equal(a, e, f"{path}[{index}]")
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=path)


def _figure_contract(figure):
    result = _decode(figure)
    # Plotly's template contains unrelated defaults (including removed mapbox
    # types). Compare the color cycle used by these figures, not that entire API.
    template = result["layout"].pop("template")
    result["layout"]["colorway"] = template["layout"]["colorway"]
    # V3 trace labels retain stored channel IDs and name formerly anonymous D21
    # traces. These are tested separately; all other trace properties are exact.
    for trace in result["data"]:
        trace.pop("name", None)
    for frame in result.get("frames", []):
        for trace in frame["data"]:
            trace.pop("name", None)
    return result


@pytest.fixture(
    scope="module",
    params=[
        ("145605", "vnasweep", 0),
        ("145485", "targsweep", 0),
        ("145609", "tune", 0),
        ("145609", "tune", 1),
    ],
)
def figures(request, tmp_path_factory):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON for production figure comparison")
    data_root = request.getfixturevalue("data_lmt_path")
    obsnum, suffix, block = request.param
    source = next((data_root / "toltec/tcs/toltec0").glob(f"*{obsnum}*_{suffix}.nc"))
    destination = tmp_path_factory.mktemp(f"figures-{obsnum}-{block}")
    subprocess.run(  # noqa: S603 -- explicitly selected local production interpreter
        [
            interpreter,
            str(Path(__file__).with_name("v2_sweep_check_plot_reference.py")),
            str(source),
            str(destination),
            str(block),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with xr.open_dataset(source) as raw:
        tree = ensure_datatree(raw)
        SweepReduce()(tree)
        scope = tree.children["blocks"].children[str(block)]
        SweepCheck()(scope)
        check = SweepCheck.view(scope)
        check.data_source.to_zarr(
            destination / "check.zarr", zarr_format=2, consolidated=True
        )
    # Exercise all figures from saved data with no live computation context.
    with xr.open_datatree(
        destination / "check.zarr", engine="zarr", consolidated=True
    ) as saved:
        reopened = SweepCheckView(saved)
        artifacts = SweepCheckArtifacts()(reopened)
    return destination, artifacts


@pytest.mark.parametrize("name", ["s21_f", "s21_f_grid", "i_q_grid"])
def test_full_figure_and_every_page_equal_production_v2(figures, name):
    destination, artifacts = figures
    expected = _figure_contract(json.loads((destination / f"{name}.json").read_text()))
    actual = _figure_contract(json.loads(artifacts[FIGURES[name]].data.to_json()))
    _assert_equal(actual, expected)


def test_full_summary_equal_production_v2(figures):
    destination, artifacts = figures
    expected = _figure_contract(
        json.loads((destination / "chan_summary.json").read_text())
    )
    actual = _figure_contract(json.loads(artifacts[ChannelSummaryViz].data.to_json()))
    for a, e in zip(actual["data"], expected["data"], strict=True):
        if "x" not in e:
            np.testing.assert_array_equal(a.pop("x"), np.arange(e["z"].shape[-1]))
    _assert_equal(actual, expected)


def test_readout_power_figure_equal_production_v2(figures):
    destination, artifacts = figures
    expected = _decode(json.loads((destination / "roach_tone_power.json").read_text()))
    actual = _decode(json.loads(artifacts[RoachReadoutPowerViz].data.to_json()))
    # Only the two channel scatter traces change renderer to WebGL. Chain nodes,
    # shading, annotations, all values/names, axis ranges and styling stay exact.
    for a, e in zip(actual["data"], expected["data"], strict=True):
        if e.get("xaxis") == "x2":
            assert a["type"] == "scattergl" and e["type"] == "scatter"
            a["type"] = "scatter"
    for fig in (actual, expected):
        template = fig["layout"].pop("template")
        fig["layout"]["colorway"] = template["layout"]["colorway"]
    _assert_equal(actual, expected)


def test_selected_channels_custom_pages_and_step_export(target_sweep_path, tmp_path):
    with xr.open_dataset(target_sweep_path) as raw:
        tree = ensure_datatree(raw.toltec_kids.select_channels(range(0, 135, 5)).load())
    SweepReduce()(tree)
    block = tree.children["blocks"].children["0"]
    SweepCheck()(block)
    check = SweepCheck.view(block)
    selected = [0, 2, 4, 6, 8, 10, 12]
    amp = S21FrequencyGridViz(page_size=4, columns=3, height=650)(
        check, channels=selected
    )
    iq = IQGridViz(page_size=4, height=600)(check, channels=selected)
    assert amp.layout.height == 650 and iq.layout.height == 600
    assert [frame.name for frame in amp.frames] == ["0", "4"]
    assert [frame.name for frame in iq.frames] == ["0", "1"]
    for figure in (amp, iq):
        assert figure.data[0].name == "Channel 0"
        assert figure.frames[-1].data[0].name == "Channel 40"
        assert len(figure.frames[-1].data[3].x) == 0
    assert len(amp.frames[0].data[4].x) == 0  # Unused slot in a nonrectangular page.
    np.testing.assert_array_equal(iq.frames[-1].data[0].x, check.sweep.I.values[8])
    np.testing.assert_array_equal(
        iq.frames[-1].data[0].error_x.array, check.sweep.unc_I.values[8]
    )
    _, context = SweepCheckPlot(
        artifacts=SweepCheckArtifacts(save_formats=["json"]),
        save_path=tmp_path,
    )(block, return_context=True)
    assert context.completed and len(context.files) == 5
    for name, cls in FIGURES.items():
        _assert_equal(
            _decode(json.loads((tmp_path / f"{name}.json").read_text())),
            _decode(json.loads(context.artifacts[cls].data.to_json())),
        )
    assert IQGridViz()(check, channels=[0]).layout.height == 1200
    for spec in (IQGridViz(), S21FrequencyGridViz()):
        with pytest.raises(ValueError, match="Select at least one channel"):
            spec(check, channels=[])
