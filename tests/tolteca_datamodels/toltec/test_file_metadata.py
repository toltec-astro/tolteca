"""Metadata queries and resource ownership for recorded TolTEC files."""

import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper, NetCDF4DataStore

from tests.file_helpers import patch_metadata_read
from tolteca_datamodels.toltec.file import ToltecFileAccessor, ToltecFileWorkflow


def test_explicit_metadata_read_enriches_recorded_telescope_and_kids_files(
    data_lmt_path,
):
    tel = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    tune = (
        data_lmt_path
        / "toltec/tcs/toltec0/toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    file_accessor = ToltecFileAccessor.from_sources([tel, tune]).read_metadata()
    table = file_accessor.table
    assert table.master.tolist() == ["tcs", "tcs"]
    assert table.obsnum.tolist() == table.file_obsnum.tolist() == [145609, 145609]
    assert file_accessor.query("obs_pgm.notna()").interface.tolist() == ["tel"]
    assert file_accessor.query("atten_drive == 7.25").interface.tolist() == ["toltec0"]


def test_catalog_failure_closes_the_open_file(data_lmt_path, monkeypatch):
    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    closed = []
    close_store = NetCDF4DataStore.close

    def close(data, **kwargs):
        close_store(data, **kwargs)
        closed.append(data)

    def fail(*args, **kwargs):
        msg = "Test header failure"
        raise ValueError(msg)

    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    patch_metadata_read(monkeypatch, ToltecFileWorkflow._source_reader_rules, fail)
    with pytest.raises(ValueError, match="Test header failure"):
        ToltecFileAccessor.from_sources([source]).read_metadata(on_error="raise")
    assert len(closed) == 1


def test_invalid_file_table_mode_rejected_before_reading(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid mode reached the source opener")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    with pytest.raises(ValidationError):
        ToltecFileAccessor.from_sources(["nonexistent.nc"], mode="invalid")  # ty: ignore[invalid-argument-type] -- Exercise runtime validation.


@pytest.mark.parametrize("mode", ["guess", "guess_or_read"])
def test_recognized_filename_with_missing_metadata_does_not_open(
    data_lmt_path, tmp_path, monkeypatch, mode
):
    (source,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    # Relocation preserves the filename but removes its acquisition directories.
    path = tmp_path / source.name
    path.symlink_to(source)

    def unexpected(*args, **kwargs):
        pytest.fail("Missing catalog metadata triggered a source read")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", unexpected)
    catalog = ToltecFileAccessor.from_sources([path], mode=mode).table
    [record] = list(catalog.itertuples(index=False))
    assert catalog["file_prefix"].iloc[0] == "toltec0"
    assert pd.isna(record.master)
    assert "filepath_orig" not in catalog
    assert len(catalog.toltec_file.query("roach == 0")) == 1
    with pytest.raises(pd.errors.UndefinedVariableError, match="atten_drive"):
        catalog.toltec_file.query("atten_drive == 7.25")
    assert "filepath_orig" not in catalog


def test_header_query_uses_catalog_snapshot_without_opening(data_lmt_path, monkeypatch):
    (source,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    catalog = ToltecFileAccessor.from_sources([source], mode="read").table

    def unexpected(*args, **kwargs):
        pytest.fail("Catalog query accessed the source")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", unexpected)
    selected = catalog.toltec_file.query("atten_drive == 7.25 & roach == 0")
    assert len(selected) == 1
    assert list(selected.itertuples(index=False)) == list(
        catalog.itertuples(index=False)
    )
