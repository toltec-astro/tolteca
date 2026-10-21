"""Production parity, persisted diagnostics and standalone plot publication."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_kids import SweepCheck, SweepCheckPlot, SweepReduce
from tolteca_kids.core import Kids
from tolteca_kids.product import (
    open_sweep_product,
    read_product_provenance,
    write_kids_product,
)
from tolteca_kids.runner import run_data
from tolteca_kids.sweep_check_viz import (
    ChannelSummaryViz,
    IQGridViz,
    S21FrequencyGridViz,
    S21FrequencyViz,
)


@pytest.mark.parametrize("despike", [True, False])
@pytest.mark.parametrize(
    ("interface", "suffix", "block"),
    [
        ("toltec0", "vnasweep", 0),
        ("toltec0", "targsweep", 0),
        ("toltec0", "tune", 0),
        ("toltec0", "tune", 1),
        ("toltec7", "tune", 0),
        ("toltec7", "tune", 1),
        ("toltec11", "tune", 0),
        ("toltec11", "tune", 1),
    ],
)
def test_all_core_results_equal_production_v2(  # noqa: PLR0913 -- parametrized production cases
    data_lmt_path, tmp_path, *, interface, suffix, block, despike
):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON for production comparison")
    obs_pattern = "1456*" if interface == "toltec0" else "156755"
    source = next(
        (data_lmt_path / "toltec/tcs" / interface).glob(f"*{obs_pattern}*_{suffix}.nc")
    )
    reference = tmp_path / "v2.npz"
    subprocess.run(  # noqa: S603 -- explicitly selected local production interpreter
        [
            interpreter,
            str(Path(__file__).with_name("v2_sweep_check_reference.py")),
            str(source),
            str(reference),
            str(block),
            str(despike).lower(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with xr.open_dataset(source) as raw:
        working = ensure_datatree(raw)
        SweepReduce()(working)
        scope = working.toltec_kids.blocks[block].tree
        original = scope.sweep.to_dataset().copy(deep=True)
        SweepCheck(despike=despike)(scope)
        xr.testing.assert_identical(scope.sweep.to_dataset(), original)
        # All scientific arrays survive export; live contexts are deliberately omitted.
        result = SweepCheck.view(scope)
        result.data_source.to_zarr(
            tmp_path / "check.zarr", zarr_format=2, consolidated=True
        )
        with xr.open_datatree(
            tmp_path / "check.zarr", engine="zarr", consolidated=True
        ) as saved:
            from tolteca_kids.sweep_check_view import SweepCheckView

            restored = SweepCheckView(saved)
            assert restored.config == result.config
            with np.load(reference) as expected:
                for name in expected.files:
                    if name.endswith("_unit"):
                        assert str(
                            getattr(restored, name.removesuffix("_unit")).unit
                        ) == str(expected[name])
                        continue
                    if name == "S21_checked":
                        actual = restored.sweep.I.values + 1j * restored.sweep.Q.values
                    elif name == "aS21_unc_db":
                        s = restored.sweep
                        actual = (
                            20
                            * np.log10(np.e)
                            * np.abs(
                                (s.unc_I.values + 1j * s.unc_Q.values)
                                / (s.I.values + 1j * s.Q.values)
                            )
                        )
                    else:
                        actual = getattr(restored, name)
                        if hasattr(actual, "unit"):
                            actual = actual.value
                        if name.endswith("_slices"):
                            actual = [(s.start, s.stop) for s in actual]
                    np.testing.assert_array_equal(actual, expected[name], err_msg=name)


def test_configured_product_contains_checks_and_html(target_sweep_path, tmp_path):
    runtime = RuntimeContext(
        {
            "kids": {
                # This test selects only three adjacent channels; use shorter
                # D21 windows for their narrow combined frequency grid.
                "sweep_check": {"enabled": True, "chunk_size": 5},
                "sweep_check_plot": {"enabled": True},
            }
        }
    )
    kids = Kids(runtime)
    with xr.open_dataset(target_sweep_path) as native:
        working = ensure_datatree(native.toltec_kids.select_channels([8, 9, 10]).load())
    assert run_data(working, target_sweep_path, runtime) is working
    checked = SweepCheck.get_context(working["blocks/0"])
    assert checked.completed
    path = write_kids_product(working, tmp_path, workflow_config=kids.config)
    index = DataProdIndex.read(path)
    artifacts = [item for item in index.data_items if item.meta["format"] == "html"]
    assert len(artifacts) == 4
    for item in artifacts:
        html = (path / item.source_uri).read_text()
        assert "Plotly.newPlot" in html
        assert 'src="https://cdn.plot.ly/plotly-' not in html
        assert "Plotly.animate(" not in html
    assert read_product_provenance(path).inputs["toltec0"].config == kids.config
    with open_sweep_product(path) as reopened:
        result = SweepCheck.view(reopened)
        np.testing.assert_array_equal(result.bitmask_chan, checked.data.bitmask_chan)
        fig = S21FrequencyGridViz()(result, channels=[1])
        expected = 20 * np.log10(
            np.abs(result.sweep.I.values[1] + 1j * result.sweep.Q.values[1])
        )
        np.testing.assert_array_equal(fig.data[0].y, expected)
        assert fig.data[0].name == "Channel 9"


def test_guide_and_paged_diagnostics_include_uncertainties(target_sweep_path):
    import re

    guide = Path(__file__).parents[2] / "docs/kids-reduction.md"
    section = guide.read_text().split("### Check sweeps and publish diagnostic plots")[
        1
    ]
    example = re.search(r"```python\n(.*?)```", section, re.DOTALL)[1]
    scope = {"raw_path": target_sweep_path}
    exec(compile(example, str(guide), "exec"), scope)  # noqa: S102 -- execute public guide
    check = scope["check"]
    tree = scope["tree"]
    with pytest.raises(KeyError):
        _ = tree.toltec_kids.sweep_check
    block = tree.toltec_kids.blocks[0]
    assert block.sweep_check.data_source is check.data_source
    assert block.tree.toltec_kids.sweep_check.data_source is check.data_source
    figure = IQGridViz()(check, channels=range(27))
    assert [frame.name for frame in figure.frames] == ["0", "25"]
    last_page = figure.frames[-1]
    sweep = check.sweep
    np.testing.assert_array_equal(last_page.data[1].x, sweep.I.values[26])
    np.testing.assert_array_equal(last_page.data[1].y, sweep.Q.values[26])
    np.testing.assert_array_equal(
        last_page.data[1].error_x.array, sweep.unc_I.values[26]
    )
    np.testing.assert_array_equal(
        last_page.data[1].error_y.array, sweep.unc_Q.values[26]
    )
    assert last_page.data[1].name == "Channel 26"
    assert len(last_page.data[2].x) == 0


def test_checks_and_plots_follow_selected_block_scope(data_lmt_path):
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    with xr.open_dataset(source) as native:
        working = ensure_datatree(
            native.toltec_kids.select_channels([0, 5, 10, 15]).load()
        )
    SweepReduce()(working)
    first = working["blocks/0"]
    second = working["blocks/1"]
    original = second.sweep.to_dataset().copy(deep=True)

    SweepCheck()(first)
    assert SweepCheck.get_context(first).completed
    assert "sweep_check" not in second.children
    xr.testing.assert_identical(second.sweep.to_dataset(), original)

    SweepCheckPlot()(first)
    plots = SweepCheckPlot.get_context(first)
    assert plots.completed
    assert len(plots.artifacts) == 4
    assert not SweepCheckPlot.has_context(second)
    check = SweepCheck.view(first)
    for viz in (
        ChannelSummaryViz(),
        S21FrequencyViz(),
        S21FrequencyGridViz(),
        IQGridViz(),
    ):
        assert plots.artifacts[viz].data.to_json() == viz(check).to_json()

    # Acquisition scope checks all blocks; block selects the published view.
    SweepCheck(block=0)(working)
    assert "sweep_check" in second.children
    assert SweepCheck.get_context(working).data.data_source is first.sweep_check
    SweepCheckPlot()(second)
    assert len(SweepCheckPlot.get_context(second).artifacts) == 4
    SweepCheckPlot()(working)
    assert not SweepCheckPlot.get_context(working).completed
    assert not SweepCheckPlot.get_context(working).artifacts
