"""Automatic catalog snapshots use recorded headers without signal reads."""

from collections import Counter
from contextlib import contextmanager

import pytest
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper, NetCDF4DataStore

from tolteca_datamodels.file import (
    SourceInfoModel,
    ToltecFileAccessor,
    guess_info_from_source,
)
from tolteca_datamodels.filestore import identify_source
from tolteca_datamodels.lmt import LmtTelMetadata
from tolteca_datamodels.toltec import (
    ToltecSweepMetadataHeader,
    ToltecTimeStreamMetadata,
)
from tolteca_datamodels.toltec.filestore import build_input_catalog
from tolteca_datamodels.toltec.kids.source import (
    identify_source as identify_kids_source,
)


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_catalog_reads_complete_headers_once_without_array_scans(
    data_lmt_path, monkeypatch, suffix
):
    pattern = f"*_{suffix}.nc" if suffix else "*_09_27_57.nc"
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
    catalog = build_input_catalog([path])
    [record] = catalog.toltec_file.to_info_list()
    cls = ToltecSweepMetadataHeader if suffix else ToltecTimeStreamMetadata
    assert type(record.metadata) is cls
    assert record.metadata.atten_drive == 7.25
    assert record.metadata.master == 0
    assert record.master == "tcs"
    assert len(reads) == len(set(reads)) == (20 if suffix else 18)
    assert len(catalog.toltec_file.query("atten_drive == 7.25 & roach == 0")) == 1
    assert len(catalog.toltec_file.query("metadata.atten_drive == 7.25")) == 1
    assert catalog.toltec_file.query("atten_drive < 0").empty
    context = record.model_dump(mode="json", context={"flat": True})
    assert context["atten_drive"] == 7.25
    assert "n_sweeps" not in context
    assert "n_blocks" not in context


def test_identification_replaces_metadata_instead_of_reusing_cached_accessor(
    data_lmt_path,
):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*_targsweep.nc")
    record = guess_info_from_source(path)
    assert record is not None
    with xr.open_dataset(path) as data:
        # Read and cache a prior snapshot; deliberate in-memory mutation must
        # not affect that frozen record or leak it into a fresh identification.
        cached = data.toltec_kids.meta_header
        first = identify_source(data, record)
        mapper = data.toltec_kids.mapper
        name = mapper.get_name(mapper.schema.atten_drive)
        assert name is not None
        data[name] = 1.25
        second = identify_source(data, first)
        assert isinstance(first.metadata, ToltecSweepMetadataHeader)
        assert isinstance(second.metadata, ToltecSweepMetadataHeader)
        assert first.metadata.atten_drive == cached.atten_drive == 7.25
        assert second.metadata.atten_drive == 1.25
        assert identify_kids_source(data, first).metadata is None
        processed = data.assign_attrs({"Header.Kids.kind": "processed_sweep"})
        assert identify_source(processed, first).metadata is None


def test_header_projection_does_not_reread_identifiers_without_xarray_cache(
    data_lmt_path, monkeypatch
):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc")
    getitem = NetCDF4ArrayWrapper._getitem
    reads = Counter()

    def read(array, key):
        reads[array.variable_name] += 1
        return getitem(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    with xr.open_dataset(path, cache=False) as data:
        record = guess_info_from_source(path)
        assert record is not None
        identify_source(data, record)
    # Header-model construction already reads ObsType twice and RoachIndex for
    # both source/calibration identity. Dispatch reads ObsType once more; source
    # projection adds no I/O for the identifiers or original filename.
    assert reads["Header.Toltec.ObsType"] == 3
    assert reads["Header.Toltec.RoachIndex"] == 2
    assert all(
        count == 1
        for name, count in reads.items()
        if name not in {"Header.Toltec.ObsType", "Header.Toltec.RoachIndex"}
    )
    assert sum(reads.values()) == 23


def test_unknown_telescope_name_keeps_typed_metadata_without_guessing_stream(
    data_lmt_path,
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(path) as data:
        # Deliberately unrecognized header path, not a positive filename case.
        changed = data.assign({"Header.File.Name": "unrecognized.nc"})
        record = identify_source(changed, SourceInfoModel(source="downloaded.nc"))
        assert record.data_kind.name == "Unknown"
        assert type(record.metadata) is LmtTelMetadata
        assert record.metadata.filepath_info_orig is None
        frame = ToltecFileAccessor.from_info_list([record])
        [restored] = frame.toltec_file.to_info_list()
        assert type(restored.metadata) is LmtTelMetadata
        assert restored.filepath_orig == "unrecognized.nc"


@pytest.mark.parametrize("header", ["Master", "DriveAtten", "ObsType"])
def test_catalog_missing_required_header_fails_and_closes(
    data_lmt_path, monkeypatch, header
):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*_targsweep.nc")
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
        build_input_catalog([path])
    assert len(closed) == 1
