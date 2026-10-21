"""Automatic catalog snapshots use recorded headers without signal reads."""

from collections import Counter
from contextlib import contextmanager

import pandas as pd
import pytest
import xarray as xr
from tollan.accessor.pandas.model_table import ModelTableAccessor
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper, NetCDF4DataStore

from tolteca_datamodels.toltec.file import ToltecFileAccessor, read_info_from_sources


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_catalog_reads_complete_headers_once_without_array_scans(
    data_lmt_path, monkeypatch, suffix
):
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob(pattern)
    getitem = NetCDF4ArrayWrapper._getitem
    reads = []

    def read(array, key):
        assert not array.variable_name.startswith("Data.")
        assert array.variable_name not in {
            "Header.Toltec.Xloc",
            "Header.Toltec.Yloc",
            "Header.Toltec.AdcSnapData",
        }
        reads.append(array.variable_name)
        return getitem(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    catalog = read_info_from_sources([path])
    [record] = list(catalog.itertuples(index=False))
    assert record.atten_drive == 7.25
    assert record.master_code == 0
    assert record.readout_code == 0
    assert "metadata" not in catalog
    assert record.master == "tcs"
    assert len(reads) == len(set(reads)) == (19 if suffix else 18)
    assert reads.count("Header.Toltec.NumSamplesPerSweepStep") == bool(suffix)
    if suffix:
        assert record.n_sweepreps == 10
    else:
        assert pd.isna(record.n_sweepreps)
    assert len(catalog.toltec_file.query("atten_drive == 7.25 & roach == 0")) == 1
    assert len(catalog.toltec_file.query("readout_code == 0")) == 1
    assert len(catalog.toltec_file.query("atten_drive == 7.25")) == 1
    assert catalog.toltec_file.query("atten_drive < 0").empty
    assert "n_sweeps" not in catalog
    assert pd.isna(record.n_sweepsteps)
    assert pd.isna(record.n_blocks)


@pytest.mark.parametrize("header", ["Master", "DriveAtten", "ObsType"])
def test_catalog_missing_required_header_fails_and_closes(
    data_lmt_path, monkeypatch, header
):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
    open_dataset = xr.open_dataset
    close_store = NetCDF4DataStore.close
    closed = []

    @contextmanager
    def incomplete(*args, **kwargs):
        with open_dataset(*args, **kwargs) as data:
            yield data.drop_vars(f"Header.Toltec.{header}")

    def close(store, **kwargs):
        close_store(store, **kwargs)
        closed.append(store)

    monkeypatch.setattr(xr, "open_dataset", incomplete)
    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    with pytest.raises(ValueError, match=f"Header.Toltec.{header}"):
        read_info_from_sources([path], on_error="raise")
    assert len(closed) == 1


def test_projection_does_not_repeat_uncached_header_reads(data_lmt_path, monkeypatch):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc")
    getitem = NetCDF4ArrayWrapper._getitem
    reads = Counter()

    def read(array, key):
        reads[array.variable_name] += 1
        return getitem(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    with xr.open_dataset(path, cache=False) as data:
        metadata = data.toltec_kids.meta
        counts_before = reads.copy()
        ModelTableAccessor.from_models(
            [metadata], columns=ToltecFileAccessor._metadata_columns
        )
        assert reads == counts_before
    # Classification and metadata each read ObsType with xarray caching disabled;
    # projecting the metadata does not reread any header.
    assert reads["Header.Toltec.ObsType"] == 2
    assert reads["Header.Toltec.RoachIndex"] == 1
    assert sum(reads.values()) == 20
