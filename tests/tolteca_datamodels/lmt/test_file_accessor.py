"""Telescope file-table construction and header-based selection."""

from contextlib import contextmanager
from pathlib import Path

import astropy.units as u
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.utils.fileloc import fileloc
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper, NetCDF4DataStore

from tolteca_datamodels.lmt import open_lmt_tel
from tolteca_datamodels.lmt.file import LmtFileAccessor


@pytest.fixture
def tel_paths(data_lmt_path):
    return sorted((data_lmt_path / "tel").glob("*.nc"))


def test_read_and_query_reference_observations(tel_paths, lmtmc_records):
    assert len(tel_paths) == 7
    records = {Path(row["FileName"]).name: row for row in lmtmc_records}
    table = LmtFileAccessor.from_sources(tel_paths, mode="read")
    assert table.source.tolist() == tel_paths
    assert table.instru.unique().tolist() == ["lmt"]
    assert not {"data_kind", "roach", "metadata", "filepath_info"} & set(table)
    for row in table.itertuples(index=False):
        assert isinstance(row.source, Path)
        expected = records[row.source.name]
        assert row.obs_goal == expected["ObsGoal"].lower()
        assert row.obs_pgm == expected["ObsPgm"].lower()
        assert (row.obsnum, row.subobsnum, row.scannum) == tuple(
            map(int, expected["ObsNum"].split("."))
        )
        assert row.t_int_s == pytest.approx(
            float(expected["IntegrationTime"]), abs=0.005
        )
        assert row.filepath_orig == expected["FileName"]
        assert (
            row.uid_raw_obs_file
            == f"tcs-{row.obsnum}-{row.subobsnum}-{row.scannum}-tel"
        )
        with open_lmt_tel(row.source) as data:
            meta = data.lmt_tel.meta
            assert row.t0_mjd == meta.t0.mjd
            assert row.t1_mjd == meta.t1.mjd
            assert row.t_elapsed_s == meta.t_elapsed.to_value(u.s)
            assert row.target_ra_deg == meta.target.ra.deg
            assert row.target_ref_dec_deg == meta.target_ref.dec.deg
            assert row.target_frame == meta.target.frame.name == "icrs"
            assert row.target_ref_frame == meta.target_ref.frame.name == "icrs"

    pointing = table.lmt_file.query('obs_goal == "pointing" and obs_pgm == "lissajous"')
    assert pointing.obsnum.tolist() == [145609, 145610, 145611]
    assert pointing.scannum.tolist() == [2, 2, 2]
    selected = pointing.lmt_file.query(
        'target_frame == "icrs" and 70 <= target_ra_deg < 80 '
        "and 60 <= target_dec_deg < 70 and t_int_s > 60 and t_elapsed_s > 65"
    )
    assert selected.uid_raw_obs_file.tolist() == ["tcs-145609-0-2-tel"]
    assert table.lmt_file.query('obs_pgm == "tune"').obs_goal.tolist() == [
        "calibration",
        "pointing",
    ]
    assert table.lmt_file.query('obs_goal == "calibration"').obsnum.tolist() == [
        145605,
        145606,
        145608,
    ]


def test_guess_never_opens_and_does_not_classify_instrument_files(
    tel_paths, data_lmt_path, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("Filename guesses opened a file")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    monkeypatch.setattr(LmtFileAccessor, "_source_opener", xr.open_dataset)
    kids_path = next((data_lmt_path / "toltec").rglob("*.nc"))
    table = LmtFileAccessor.from_sources([tel_paths[0], kids_path, "unknown.nc"])
    assert table.iloc[0].uid_raw_obs_file == "tcs-145605-0-1-tel"
    assert table.iloc[0].file_prefix == "tel_toltec"
    assert table.iloc[1:].obsnum.isna().all()
    assert table.iloc[1:].uid_raw_obs_file.isna().all()
    assert "obs_goal" not in table


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_mode_controls_reads_and_recovers_renamed_file(
    tel_paths, tmp_path, monkeypatch, mode
):
    renamed = tmp_path / "renamed.nc"
    renamed.symlink_to(tel_paths[-1])
    open_dataset = xr.open_dataset
    opened = []

    @contextmanager
    def tracked(path):
        opened.append(path)
        with open_dataset(path) as data:
            yield data

    monkeypatch.setattr(xr, "open_dataset", tracked)
    monkeypatch.setattr(LmtFileAccessor, "_source_opener", xr.open_dataset)
    table = LmtFileAccessor.from_sources([tel_paths[0], renamed], mode=mode)
    expected = {
        "guess": [],
        "guess_or_read": [renamed],
        "read": [tel_paths[0], renamed],
    }
    assert opened == expected[mode]
    assert pd.isna(table.file_obsnum.iloc[1])
    if mode == "guess":
        assert pd.isna(table.obsnum.iloc[1])
    else:
        assert table.obsnum.iloc[1] == 145611
        assert table.obs_goal.iloc[1] == "pointing"
        assert table.uid_raw_obs_file.iloc[1] == "tcs-145611-0-2-tel"
    if mode == "guess_or_read":
        # A recognized filename's missing goal does not trigger a read.
        assert pd.isna(table.obs_goal.iloc[0])


def test_reads_local_source_not_remote_identification_location(tel_paths, tmp_path):
    cached = tmp_path / "download.cache"
    cached.symlink_to(tel_paths[0])
    # Deliberately conflicting identification name tests header precedence.
    remote = fileloc(f"archive:/data_lmt/tel/{tel_paths[-1].name}")
    table = LmtFileAccessor.from_sources([{"source": cached, "source_loc": remote}])
    table.index = pd.Index([8], name="input_row")
    table["batch"] = ["first"]
    table.attrs["label"] = "cached observation"
    before = table.copy(deep=True)
    enriched = table.lmt_file.read_metadata()
    pd.testing.assert_frame_equal(table, before)
    assert enriched.obsnum.iloc[0] == 145605
    assert enriched.file_obsnum.iloc[0] == 145611
    assert enriched.uid_raw_obs_file.iloc[0] == "tcs-145605-0-1-tel"
    pd.testing.assert_frame_equal(
        enriched[["source", "source_loc", "batch"]],
        table[["source", "source_loc", "batch"]],
    )
    assert enriched.attrs == table.attrs
    assert enriched.source_loc.iloc[0] is remote


@pytest.mark.parametrize("as_tree", [False, True])
def test_borrowed_objects_stay_open(tel_paths, monkeypatch, as_tree):
    def unexpected(*args, **kwargs):
        pytest.fail("Borrowed dataset was reopened")

    with open_lmt_tel(tel_paths[0]) as dataset:
        obj = xr.DataTree(dataset) if as_tree else dataset
        closed = []
        close = obj._close

        def track_close():
            closed.append(True)
            if close is not None:
                close()

        obj.set_close(track_close)
        monkeypatch.setattr(xr, "open_dataset", unexpected)
        monkeypatch.setattr(LmtFileAccessor, "_source_opener", xr.open_dataset)
        table = LmtFileAccessor.from_sources(
            [{"source": obj, "source_loc": tel_paths[0]}], mode="read"
        )
        assert table.source.iloc[0] is obj
        assert table.obsnum.iloc[0] == 145605
        assert not closed
        assert obj.lmt_tel.az.isel(time=0).item() is not None


def test_header_only_reads_close_each_owned_file(tel_paths, monkeypatch):
    original = NetCDF4ArrayWrapper.__getitem__
    open_dataset = xr.open_dataset
    closed = []
    read = []

    def header_only(array, key):
        read.append(array.variable_name)
        if array.variable_name.startswith("Data."):
            assert array.variable_name == "Data.TelescopeBackend.TelTime"
            assert key.tuple in [(0,), (array.shape[0] - 1,)]
        return original(array, key)

    @contextmanager
    def tracked(path):
        with open_dataset(path) as data:
            yield data
        closed.append(path)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", header_only)
    monkeypatch.setattr(xr, "open_dataset", tracked)
    monkeypatch.setattr(LmtFileAccessor, "_source_opener", xr.open_dataset)
    table = LmtFileAccessor.from_sources(tel_paths, mode="read")
    assert closed == tel_paths
    assert len(read) == len(tel_paths) * 12
    assert len(table) == len(tel_paths)


def test_open_queries_headers_and_loads_pointing_without_reopening(
    tel_paths, monkeypatch
):
    table = LmtFileAccessor.from_sources([*tel_paths, tel_paths[-1]])
    table.index = pd.Index([*range(len(tel_paths)), 0], name="input_row")
    before = table.copy()
    opened, closed = [], []
    open_dataset, close_store = xr.open_dataset, NetCDF4DataStore.close
    getitem = NetCDF4ArrayWrapper._getitem
    metadata_only = True

    def open_data(path):
        opened.append(path)
        return open_dataset(path)

    def close(store, **kwargs):
        closed.append(store)
        return close_store(store, **kwargs)

    def read(array, key):
        if metadata_only and array.variable_name.startswith("Data."):
            assert array.variable_name == "Data.TelescopeBackend.TelTime"
            assert key in ((0,), (array.shape[0] - 1,))
        return getitem(array, key)

    monkeypatch.setattr(xr, "open_dataset", open_data)
    monkeypatch.setattr(LmtFileAccessor, "_source_opener", xr.open_dataset)
    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    with table.lmt_file.open() as live:
        assert opened == tel_paths and not closed
        assert live.io_obj.iloc[-1] is live.io_obj.iloc[-2]
        pd.testing.assert_index_equal(live.index, table.index)
        selected = live.lmt_file.query(
            'obs_goal == "pointing" and obs_pgm == "lissajous"'
        )
        assert selected.obsnum.tolist() == [145609, 145610, 145611, 145611]
        with selected.iloc[::-1].lmt_file.open() as nested:
            assert nested.io_obj.iloc[0] is selected.io_obj.iloc[-1]
            assert opened == tel_paths and not closed
            metadata_only = False
            az = nested.io_obj.iloc[0].lmt_tel.az.isel(time=slice(0, 10)).load()
        assert not closed
    assert len(closed) == len(tel_paths)
    assert az.size == 10
    assert "io_obj" not in live and "io_obj" not in nested
    assert live.obs_goal.notna().all()
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("as_tree", [False, True])
@pytest.mark.parametrize("supplied_as", ["source", "io_objs", "column"])
def test_open_borrows_remote_resource_and_refreshes_from_it(
    tel_paths, monkeypatch, as_tree, supplied_as
):
    location = fileloc(f"archive:/data_lmt/tel/{tel_paths[0].name}")
    with xr.open_dataset(tel_paths[0]) as data:
        obj = xr.DataTree(data) if as_tree else data
        source = obj if supplied_as == "source" else location
        table = LmtFileAccessor.from_sources(
            [{"source": source, "source_loc": location}]
        )
        if supplied_as == "column":
            table["io_obj"] = pd.Series([obj], dtype=object).array
        before = table.copy()

        def unexpected(*args, **kwargs):
            pytest.fail("A borrowed telescope object was opened or closed")

        with monkeypatch.context() as patch:
            patch.setattr(xr, "open_dataset", unexpected)
            patch.setattr(LmtFileAccessor, "_source_opener", xr.open_dataset)
            patch.setattr(xr.Dataset, "close", unexpected)
            patch.setattr(xr.DataTree, "close", unexpected)
            io_objs = [obj] if supplied_as == "io_objs" else None
            with table.lmt_file.open(io_objs=io_objs) as live:
                assert live.io_obj.iloc[0] is obj
                assert live.obsnum.tolist() == [145605]
                cached = obj.lmt_tel.meta
                obj["Header.Dcs.ObsGoal"] = "Pointing"
                refreshed = live.lmt_file.read_metadata()
                assert refreshed.io_obj.iloc[0] is obj
                assert refreshed.obs_goal.tolist() == ["pointing"]
                assert obj.lmt_tel.meta is cached
                assert cached.obs_goal == "calibration"
            assert "io_obj" not in live
            pd.testing.assert_frame_equal(table, before)
            assert data.lmt_tel.az.isel(time=0).compute().size == 1


def test_partial_refresh_preserves_remote_rows_and_duplicate_indices(
    tel_paths, monkeypatch
):
    table = LmtFileAccessor.from_sources([tel_paths[0], tel_paths[-1]], mode="read")
    table["source"] = [f"archive:{tel_paths[0]}", tel_paths[-1]]
    table.index = [8, 8]
    open_dataset = xr.open_dataset

    @contextmanager
    def changed(path):
        assert path == tel_paths[-1]
        with open_dataset(path) as data:
            yield data.assign({"Header.Dcs.IntegrationTime": 12.5})

    monkeypatch.setattr(xr, "open_dataset", changed)
    monkeypatch.setattr(LmtFileAccessor, "_source_opener", xr.open_dataset)
    assert table.lmt_file.read_metadata(inplace=True) is None
    assert table.t_int_s.tolist() == [0.0, 12.5]
    assert table.index.tolist() == [8, 8]


def test_bad_header_propagates_and_leaves_table_unchanged(tel_paths, monkeypatch):
    paths = tel_paths[:2]
    table = LmtFileAccessor.from_sources(paths)
    before = table.copy(deep=True)
    open_dataset = xr.open_dataset
    closed = []

    @contextmanager
    def corrupt_second(path):
        try:
            with open_dataset(path) as data:
                yield data if path == paths[0] else data.drop_vars("Header.Dcs.ObsGoal")
        finally:
            closed.append(path)

    monkeypatch.setattr(xr, "open_dataset", corrupt_second)
    monkeypatch.setattr(LmtFileAccessor, "_source_opener", xr.open_dataset)
    with pytest.raises(KeyError):
        table.lmt_file.read_metadata(inplace=True)
    assert closed == paths
    pd.testing.assert_frame_equal(table, before)


def test_unsupported_and_non_telescope_resources(data_lmt_path, tel_paths):
    kids = next((data_lmt_path / "toltec").rglob("*.nc"))
    sources = [kids, "no-reader.ecsv", f"archive:{tel_paths[0]}"]
    guessed = LmtFileAccessor.from_sources(sources)
    refreshed = guessed.lmt_file.read_metadata()
    pd.testing.assert_frame_equal(
        refreshed.drop(columns="diagnostics"), guessed.drop(columns="diagnostics")
    )
    assert [[item.stage for item in cell] for cell in refreshed.diagnostics] == [
        ["filepath", "metadata"],
        ["filepath"],
        [],
    ]
    assert refreshed.diagnostics.iloc[0][-1].exception_type is None
    assert len(guessed.diagnostics.iloc[0]) == 1
    with (
        pytest.raises(ValueError, match="No supported dataset recognized"),
        refreshed.iloc[:1].lmt_file.open(on_error="raise"),
    ):
        pytest.fail("KIDs data was accepted by the telescope accessor")


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_empty_input(mode):
    table = LmtFileAccessor.from_sources([], mode=mode)
    assert table.empty
    assert str(table.obsnum.dtype) == "Int64"
    pd.testing.assert_frame_equal(table.lmt_file.read_metadata(), table)
    with table.lmt_file.open() as opened:
        assert opened.io_obj.empty
        assert opened.diagnostics.empty
    pd.testing.assert_frame_equal(opened.drop(columns="diagnostics"), table)


def test_invalid_mode():
    with pytest.raises(ValidationError):
        LmtFileAccessor.from_sources([], mode="invalid")  # ty: ignore[invalid-argument-type]


def test_obs_spec_uses_only_lmt_context(tel_paths):
    table = LmtFileAccessor.from_sources(tel_paths, mode="read")
    assert table.lmt_file.query_obs_spec("145609/tel").scannum.tolist() == [1, 2]
    assert table.lmt_file.query_obs_spec("145609/{tel,1}").scannum.tolist() == [1, 2]
    assert table.lmt_file.query_obs_spec("145609/1").empty
    assert table.lmt_file.query_obs_spec("tcs-[-1]").obsnum.tolist() == [145611]
