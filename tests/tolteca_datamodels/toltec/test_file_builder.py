"""File table construction, in-place stages and authoritative metadata."""

from contextlib import contextmanager

import pandas as pd
import pytest
import xarray as xr
from pydantic import TypeAdapter
from tollan.utils.typing import get_literal_values
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import file as file_module
from tolteca_datamodels.toltec.file import (
    FileTableModeT,
    ToltecFileAccessor,
    get_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_population_choices():
    choices = ["guess", "guess_or_read", "read"]
    assert get_literal_values(FileTableModeT) == choices
    adapter = TypeAdapter(FileTableModeT)
    for choice in choices:
        assert adapter.validate_python(choice) == choice


@pytest.mark.parametrize("mode", ["guess", "read"])
def test_factory_enriches_one_table_with_one_accessor(data_lmt_path, monkeypatch, mode):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    working_tables = []
    instances = []
    stages = []
    make_uid_columns = file_module.make_uid_columns

    def trace_uids(master, obsnum, subobsnum, scannum, interface):
        stages.append("uids")
        table = working_tables[0]
        for name, column in (
            ("master", master),
            ("obsnum", obsnum),
            ("subobsnum", subobsnum),
            ("scannum", scannum),
            ("interface", interface),
        ):
            pd.testing.assert_series_equal(column, table[name])
        return make_uid_columns(master, obsnum, subobsnum, scannum, interface)

    class Files(ToltecFileAccessor):
        def __init__(self, table):
            working_tables.append(table)
            instances.append(self)
            super().__init__(table)

        def _guess_data_kinds(self):
            stages.append("kind")
            return super()._guess_data_kinds()

        def _read_metadata(self, rows, *, trace):
            stages.append("metadata")
            return super()._read_metadata(rows, trace=trace)

    monkeypatch.setattr(file_module, "make_uid_columns", trace_uids)
    result = Files.from_sources(iter([path]), mode=mode)
    assert result is working_tables[0]
    assert len(instances) == 1
    assert stages == ["kind", "uids"] + (["metadata"] if mode == "read" else [])
    assert result["uid_raw_obs_file"].tolist() == ["tcs-145609-0-2-tel"]
    if mode == "read":
        assert result.obs_goal.tolist() == ["pointing"]
    else:
        assert "obs_goal" not in result


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_factory_metadata_policy_and_resource_lifetime(
    data_lmt_path, tmp_path, monkeypatch, mode
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    cached = tmp_path / "download.nc"
    cached.symlink_to(path)
    opened, closed = [], []
    open_dataset = xr.open_dataset
    read = NetCDF4ArrayWrapper._getitem

    @contextmanager
    def open_data(source, **kwargs):
        opened.append(source)
        try:
            with open_dataset(source, **kwargs) as data:
                yield data
        finally:
            closed.append(source)

    def header_only(array, key):
        if array.variable_name.startswith("Data."):
            assert array.variable_name == "Data.TelescopeBackend.TelTime"
            assert key in ((0,), (array.shape[0] - 1,))
        return read(array, key)

    monkeypatch.setattr(xr, "open_dataset", open_data)
    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", header_only)
    table = ToltecFileAccessor.from_sources([path, cached, path], mode=mode)
    expected = [] if mode == "guess" else [cached]
    if mode == "read":
        expected = [path, cached, path]
    assert opened == closed == expected
    assert table["source"].tolist() == [path, cached, path]
    assert pd.isna(table["file_prefix"].iloc[1])
    assert "io_obj" not in table
    if mode != "guess":
        assert table["uid_raw_obs_file"].iloc[1] == "tcs-145609-0-2-tel"
        assert table["obs_goal"].iloc[1] == "pointing"
    else:
        assert pd.isna(table["uid_raw_obs_file"].iloc[1])
        assert "obs_goal" not in table


def test_factory_headers_override_guesses_without_recomputing(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    location = path.with_name(
        "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc"
    )
    with xr.open_dataset(path) as raw:
        table = ToltecFileAccessor.from_sources(
            [{"source": raw, "source_loc": location}], mode="read"
        )
        assert table["source"].iloc[0] is raw
        assert table["file_obsnum"].iloc[0] == 145606
        assert table["obsnum"].iloc[0] == 145609
        assert table["file_suffix"].iloc[0] == "targsweep"
        assert table["data_kind_toltec"].iloc[0] == ToltecDataKind.tune.value
        assert table["uid_raw_obs_file"].iloc[0] == "tcs-145609-0-1-toltec0"
        assert raw.toltec_kids.meta.obsnum == 145609


def test_guess_helper_and_single_default_use_guess_mode(monkeypatch):
    modes = []
    from_sources = ToltecFileAccessor.from_sources

    def build(cls, sources, *, mode, on_error):
        modes.append(mode)
        return from_sources(sources, mode=mode, on_error=on_error)

    monkeypatch.setattr(ToltecFileAccessor, "from_sources", classmethod(build))
    name = "tel_toltec_2025-10-31_145609_00_0002.nc"
    assert get_info_from_source(name).obsnum == 145609
    assert guess_info_from_sources([name])["obsnum"].tolist() == [145609]
    assert modes == ["guess", "guess"]


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_factory_empty_table(mode):
    table = ToltecFileAccessor.from_sources([], mode=mode)
    assert table.empty
    assert str(table["data_kind_toltec"].dtype) == "Int64"
    assert str(table["uid_raw_obs_file"].dtype) == "string"


def test_inplace_metadata_refresh_preserves_table_identity(data_lmt_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    table = guess_info_from_sources([path])
    accessor = table.toltec_file
    assert accessor.update_from_source_metadata(inplace=True) is None
    assert accessor._obj is table
    assert table["obs_goal"].tolist() == ["pointing"]


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_remote_and_unsupported_sources_keep_their_guesses(monkeypatch, mode):
    sources = [
        "archive:/data_lmt/tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        "archive:/download.nc",
        "notes.txt",
    ]

    def unexpected(*args, **kwargs):
        pytest.fail("Unavailable metadata reader attempted I/O")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    guessed = guess_info_from_sources(sources)
    table = ToltecFileAccessor.from_sources(sources, mode=mode)
    pd.testing.assert_frame_equal(
        table.drop(columns="source_trace"), guessed.drop(columns="source_trace")
    )
    assert table.source_trace.iloc[-1][-1].outcome == (
        "skipped" if mode == "guess" else "error"
    )


@pytest.mark.parametrize("mode", ["guess_or_read", "read"])
def test_attempted_local_read_errors_propagate(tmp_path, mode):
    with pytest.raises(FileNotFoundError):
        ToltecFileAccessor.from_sources(
            [tmp_path / "missing.nc"], mode=mode, on_error="raise"
        )


def test_default_factory_only_guesses(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Default factory attempted I/O")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    sources = ["tel_toltec_2025-10-31_145609_00_0002.nc", "unidentified.nc"]
    pd.testing.assert_frame_equal(
        ToltecFileAccessor.from_sources(sources), guess_info_from_sources(sources)
    )
