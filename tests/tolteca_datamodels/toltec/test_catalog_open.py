"""Catalog resource ownership and row alignment."""

import pandas as pd
import pytest
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper, NetCDF4DataStore

from tests.source_helpers import patch_metadata_read
from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.toltec.file import ToltecFileAccessor, guess_info_from_sources


@pytest.fixture
def paths(data_lmt_path):
    return [
        data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        data_lmt_path
        / "toltec/tcs/toltec0"
        / "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
    ]


@pytest.fixture
def resource_calls(monkeypatch):
    opened, closed = [], []
    open_dataset, close_store = xr.open_dataset, NetCDF4DataStore.close

    def open_data(path):
        opened.append(path)
        return open_dataset(path)

    def close(store, **kwargs):
        closed.append(store)
        return close_store(store, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", open_data)
    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    return opened, closed


def test_open_metadata_query_alignment_and_cleanup(paths, resource_calls, monkeypatch):
    catalog = guess_info_from_sources([paths[0], paths[1], paths[0]])
    catalog.index = pd.Index([8, 3, 8], name="input_row")
    catalog["batch"] = ["first", "second", "third"]
    catalog.attrs["label"] = "selected inputs"
    before = catalog.copy(deep=True)
    read = NetCDF4ArrayWrapper._getitem

    def header_only(array, key):
        if array.variable_name.startswith("Data."):
            assert array.variable_name == "Data.TelescopeBackend.TelTime"
            assert key in ((0,), (array.shape[0] - 1,))
        return read(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", header_only)
    opened, closed = resource_calls
    assert catalog.get("io_obj") is None
    with catalog.toltec_file.open() as live:
        live.toltec_file.update_from_source_metadata(inplace=True)
        assert live is not catalog
        pd.testing.assert_frame_equal(catalog, before)
        pd.testing.assert_index_equal(live.index, catalog.index)
        pd.testing.assert_series_equal(live["batch"], catalog["batch"])
        assert live.attrs == catalog.attrs
        objects = live.io_obj
        assert len(objects) == 3
        assert objects.iloc[0] is objects.iloc[2]
        for row, obj in zip(live.itertuples(index=False), objects, strict=True):
            assert row.io_obj is obj
        selected = live.toltec_file.query('interface == "tel"').iloc[::-1]
        assert selected["batch"].tolist() == ["third", "first"]
        assert selected.io_obj.iloc[0] is objects.iloc[2]
        assert len(live.toltec_file.query("atten_drive == 7.25")) == 1
        assert "io_obj" in live
        assert opened == paths and not closed
        # Inner scopes borrow; they do not reopen or close outer resources.
        with selected.toltec_file.open() as nested:
            assert nested.io_obj.iloc[0] is objects.iloc[0]
        assert nested.get("io_obj") is None
        assert opened == paths and not closed
        assert live.get("io_obj") is not None
        # Dropping runtime columns must not interfere with resource cleanup.
        del live["io_obj"]
    assert live.get("io_obj") is None
    assert len(closed) == 2
    assert live["filepath_orig"].notna().all()
    pd.testing.assert_frame_equal(catalog, before)


@pytest.mark.parametrize("as_tree", [False, True])
def test_borrowed_remote_resource_is_not_reopened_or_closed(
    paths, resource_calls, as_tree
):
    remote = f"archive:/data_lmt/tel/{paths[0].name}"
    catalog = guess_info_from_sources([remote])
    opened, closed = resource_calls
    with xr.open_dataset(paths[0]) as data:
        obj = xr.DataTree(data) if as_tree else data
        with catalog.toltec_file.open(io_objs=[obj]) as live:
            assert live.io_obj.iloc[0] is obj
            assert live["obsnum"].tolist() == [145609]
            enriched = live.toltec_file.update_from_source_metadata()
            assert enriched.io_obj.iloc[0] is obj
            assert enriched["obs_goal"].iloc[0] == "pointing"
        assert opened == [paths[0]]
        assert not closed
        assert data.lmt_tel.az.isel(time=0).compute().size == 1
    assert len(closed) == 1


@pytest.mark.parametrize("failure", ["open", "metadata", "body"])
def test_failure_closes_owned_resources_and_preserves_input(
    paths, resource_calls, monkeypatch, tmp_path, failure
):
    if failure == "open":
        paths = [paths[0], tmp_path / "missing.nc"]
    catalog = guess_info_from_sources(paths)
    before = catalog.copy(deep=True)
    opened, closed = resource_calls
    if failure == "metadata":

        def fail(*args, **kwargs):
            msg = "Header read failure"
            raise RuntimeError(msg)

        patch_metadata_read(monkeypatch, ToltecFileAccessor._reader_rules, fail)
    with (  # noqa: PT012 -- test failures entering and using the resource scope.
        pytest.raises((FileNotFoundError, RuntimeError)),
        catalog.toltec_file.open(on_error="raise") as live,
    ):
        live.toltec_file.update_from_source_metadata(inplace=True, on_error="raise")
        assert failure == "body"
        msg = "Caller failure"
        raise RuntimeError(msg)
    assert len(closed) == (1 if failure == "open" else len(opened))
    pd.testing.assert_frame_equal(catalog, before)
    if failure == "body":
        assert live.get("io_obj") is None


def test_open_defers_metadata_until_explicit_update(paths, monkeypatch):
    read = []

    def traced_read(read_source):
        def read_data(data):
            read.append(data)
            return read_source(data)

        return read_data

    for accessor_cls in ToltecFileAccessor._reader_rules.accessor_classes:
        monkeypatch.setattr(
            accessor_cls, "meta", property(traced_read(accessor_cls.meta.func))
        )
    catalog = guess_info_from_sources(paths)
    assert not read
    with catalog.toltec_file.open() as live:
        assert len(live.io_obj) == len(paths)
        assert not read
        live.toltec_file.update_from_source_metadata(inplace=True)
        assert len(read) == len(paths)
        assert live["filepath_orig"].notna().all()


@pytest.mark.parametrize("objects", [[], [object()]])
def test_bad_borrowed_objects_rejected_before_opening(paths, objects, resource_calls):
    catalog = guess_info_from_sources(paths[:1])
    with (
        pytest.raises((ValueError, TypeError)),
        catalog.toltec_file.open(io_objs=objects, on_error="raise"),
    ):
        pytest.fail("Invalid IO objects were accepted")
    assert resource_calls == ([], [])


def test_empty_catalog_open():
    catalog = guess_info_from_sources([])
    with catalog.toltec_file.open() as live:
        assert live.empty
        assert live.io_obj.empty
        assert live.source_trace.map(source_trace_issues).empty
    pd.testing.assert_frame_equal(live, catalog)


def test_mixed_owned_and_borrowed_resources_on_error(paths, resource_calls):
    catalog = guess_info_from_sources(paths)
    opened, closed = resource_calls
    with xr.open_dataset(paths[0]) as borrowed:
        msg = "Caller failure"
        with (
            pytest.raises(RuntimeError, match=msg),
            catalog.toltec_file.open(io_objs=[borrowed, None]),
        ):
            raise RuntimeError(msg)
        assert opened == paths
        assert len(closed) == 1
        assert borrowed.lmt_tel.az.isel(time=0).compute().size == 1
    assert len(closed) == 2
