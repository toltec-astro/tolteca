"""Finding plots and tables survive publication without live step contexts."""

import copy
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from astropy.table import QTable
from tollan.accessor.xarray import ensure_datatree
from tollan.pipeline import PIPELINE_CONTEXT_KEY, ArtifactContext

from tolteca_kids import BlockCreate, KidsFind, KidsFindPlot, SweepCheck, SweepReduce
from tolteca_kids.kids_find_artifacts import FIND_COLUMNS
from tolteca_kids.kids_plot import make_kids_find_figs

from .test_sweep_check_plot_parity import _assert_equal, _decode

FIGURES = (
    "peaks",
    "peak_props",
    "d21_summary",
    "s21_summary",
    "det_summary",
    "matched",
    "matched_ref",
)


def _contract(figure):
    result = _decode(figure)
    # Renderer versions carry different unused template entries (e.g. mapbox).
    # All explicit figure settings and the template's color cycle are compared.
    template = result["layout"].pop("template")
    result["layout"]["colorway"] = template["layout"]["colorway"]
    traces = []
    for original in result["data"]:
        trace = copy.deepcopy(original)
        if trace.get("mode") == "lines" and trace.get("line", {}).get("color") == "red":
            # V3 batches independent match lines into one trace. Compare every
            # segment against v2, including order and style, without requiring
            # hundreds of separate browser traces. Both figures hide legends.
            trace.pop("showlegend", None)
            if any(value is None for value in trace["x"]):
                traces.extend(
                    trace
                    | {
                        "x": trace["x"][start : start + 2],
                        "y": trace["y"][start : start + 2],
                    }
                    for start in range(0, len(trace["x"]), 3)
                )
                continue
        traces.append(trace)
    result["data"] = traces
    return result


@pytest.fixture(
    scope="module",
    params=[
        ("145485", "targsweep", 0, "chan"),
        ("145485", "targsweep", 0, "d21"),
        ("145609", "tune", 1, "s21"),
    ],
)
def saved_finding(request, tmp_path_factory):
    obsnum, suffix, block_id, match_ref = request.param
    data_root = request.getfixturevalue("data_lmt_path")
    source = next((data_root / "toltec/tcs/toltec0").glob(f"*{obsnum}*_{suffix}.nc"))
    destination = tmp_path_factory.mktemp(f"finding-{obsnum}-{block_id}-{match_ref}")
    with xr.open_dataset(source) as raw:
        tree = ensure_datatree(raw)
        BlockCreate()(tree)
        block = tree.children["blocks"].children[str(block_id)]
        SweepReduce()(block)
        SweepCheck()(block)
        KidsFind(match_ref=match_ref)(block)
        context = KidsFind.get_context(block)
        assert context.completed
        tables = {
            name: table[list(FIND_COLUMNS)].copy()
            for name, table in (
                ("kids_find", context.data.detected_matched),
                ("chan_prop", context.data.chan_matched),
            )
        }
        # Close all original resources and remove the in-memory execution state.
        saved = block.copy(deep=False)
        for node in saved.subtree:
            node.attrs.pop(PIPELINE_CONTEXT_KEY, None)
            node.dataset = node.dataset.drop_encoding()
        saved.to_zarr(destination / "block.zarr", zarr_format=2, consolidated=True)
    return source, destination, block_id, match_ref, tables


@pytest.fixture(scope="module")
def reference(saved_finding):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON for production-v2 comparison")
    source, destination, block_id, match_ref, _ = saved_finding
    subprocess.run(  # noqa: S603 -- explicitly selected production interpreter
        [
            interpreter,
            str(Path(__file__).with_name("v2_kids_find_plot_reference.py")),
            str(source),
            str(destination / "v2"),
            str(block_id),
            match_ref,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return destination / "v2"


@pytest.fixture(scope="module")
def saved_figures(saved_finding):
    _, destination, _, _, _ = saved_finding
    with xr.open_datatree(
        destination / "block.zarr", engine="zarr", consolidated=True
    ) as tree:
        view = KidsFind.view(tree)
        return make_kids_find_figs(SweepCheck.view(tree).sweep.data_source, view)


@pytest.mark.parametrize("name", FIGURES)
def test_saved_figures_match_production_v2(reference, saved_figures, name):
    actual = _contract(json.loads(saved_figures[name].to_json()))
    expected = _contract(json.loads((reference / f"{name}.json").read_text()))
    _assert_equal(actual, expected, name)


def _assert_table_equal(actual, expected):
    assert actual.colnames == expected.colnames
    for name in expected.colnames:
        assert getattr(actual[name], "unit", None) == getattr(
            expected[name], "unit", None
        )
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    np.testing.assert_equal(actual.meta["shift"], expected.meta["shift"])


def test_saved_tables_retain_values_units_order_and_identity(saved_finding, tmp_path):
    source, destination, _, _, expected = saved_finding
    with xr.open_datatree(
        destination / "block.zarr", engine="zarr", consolidated=True
    ) as tree:
        view = KidsFind.view(tree)
        tables = view.export_tables()
        for name, table in tables.items():
            _assert_table_equal(table, expected[name])
            obsnum = table.meta["Header.Toltec.ObsNum"]
            assert str(obsnum) in source.name
            assert table.meta["uid_raw_obs"] == f"tcs-{obsnum}-0-1"
            assert table.meta["uid_raw_obs_file"] == f"tcs-{obsnum}-0-1-toltec0"
            assert table.meta["Header.Toltec.RoachIndex"] == 0
            assert table.meta["Header.Toltec.SubObsNum"] == 0
            assert table.meta["Header.Toltec.ScanNum"] == 1
        artifacts = ArtifactContext()
        for name, table in tables.items():
            artifacts.add(
                view.config.artifacts.table.model_copy(update={"alias": name}), table
            )
        receipts = view.config.artifacts.write(artifacts, path=tmp_path)
        assert len(receipts) == 2
    for name, expected_table in tables.items():
        table = QTable.read(tmp_path / f"{name}.ecsv")
        _assert_table_equal(table, expected_table)
        assert table.meta == expected_table.meta


def test_exported_tables_match_production_v2(reference, saved_finding):
    _, destination, _, _, _ = saved_finding
    with xr.open_datatree(
        destination / "block.zarr", engine="zarr", consolidated=True
    ) as tree:
        for name, table in KidsFind.view(tree).export_tables().items():
            _assert_table_equal(table, QTable.read(reference / f"{name}.ecsv"))


def test_saved_plot_step_never_runs_finding_or_matching(saved_finding, monkeypatch):
    from tolteca_kids.match1d import Match1D
    from tolteca_kids.peaks1d import Peaks1D

    def forbidden(*args, **kwargs):
        pytest.fail("Plotting invoked scientific computation")

    for cls, name in (
        (KidsFind, "run"),
        (SweepCheck, "run"),
        (Match1D, "__call__"),
        (Peaks1D, "__call__"),
    ):
        monkeypatch.setattr(cls, name, forbidden)
    _, destination, _, match_ref, _ = saved_finding
    with xr.open_datatree(
        destination / "block.zarr", engine="zarr", consolidated=True
    ) as tree:
        assert PIPELINE_CONTEXT_KEY not in tree.attrs
        view = KidsFind.view(tree)
        assert view.config.match_ref == match_ref
        assert ("matched_ref" in view.data_source.children) == (match_ref != "chan")
        _, context = KidsFindPlot()(tree, return_context=True)
        assert context.completed
        assert len(context.artifacts) == 7
        assert not KidsFind.has_context(tree)


def test_product_publication_includes_finding_datasets_and_artifacts(
    data_lmt_path, tmp_path, monkeypatch
):
    from tolteca_datamodels.data_prod import DataProdIndex
    from tolteca_kids.product import open_sweep_product, write_kids_product

    source = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145485*_targsweep.nc"))
    with xr.open_dataset(source) as raw:
        tree = ensure_datatree(raw)
        SweepReduce()(tree)
        block = tree.children["blocks"].children["0"]
        SweepCheck()(block)
        KidsFind()(block)
        KidsFindPlot(artifacts={"figure": {"save_formats": ["json"]}})(block)
        expected = KidsFind.view(block).export_tables()
        path = write_kids_product(tree, tmp_path)
    index = DataProdIndex.read(path)
    items = index.product.data_items
    assert sum(item.meta.get("format") == "json" for item in items) == 7
    assert sum(item.meta.get("format") == "ecsv" for item in items) == 2
    with open_sweep_product(path) as reopened:
        block = reopened.children["blocks"].children["0"]
        assert not KidsFind.has_context(block)
        view = KidsFind.view(block)
        for name, table in view.export_tables().items():
            _assert_table_equal(table, expected[name])
            exported = next(
                item for item in items if item.source.path.name == f"{name}.ecsv"
            )
            _assert_table_equal(QTable.read(path / exported.source.path), table)
        figures = make_kids_find_figs(SweepCheck.view(block).sweep.data_source, view)
        for name, fig in figures.items():
            exported = next(
                item for item in items if item.source.path.name == f"{name}.json"
            )
            _assert_equal(
                _decode(json.loads(fig.to_json())),
                _decode(json.loads((path / exported.source.path).read_text())),
            )
    guide = Path(__file__).parents[2] / "docs/kids-reduction.md"
    section = guide.read_text().split(
        "Recreate figures or export tables from a saved product:"
    )[1]
    match = re.search(r"```python\n(.*?)```", section, re.DOTALL)
    assert match is not None
    example = match[1]
    monkeypatch.chdir(tmp_path)
    scope = {"product_path": path}
    exec(compile(example, str(guide), "exec"), scope)  # noqa: S102 -- execute public guide
    assert (tmp_path / "peaks.html").is_file()
    _assert_table_equal(QTable.read(tmp_path / "kids_find.ecsv"), expected["kids_find"])
