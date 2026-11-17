"""Acquisition metadata and inherited block records have explicit read costs."""

import pytest
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlocksMetadata,
    ToltecRawObsMetadata,
)
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
@pytest.mark.parametrize("chunks", [None, {}])
def test_block_index_inherits_acquisition_headers_without_reading_signals(
    data_lmt_path, monkeypatch, suffix, chunks
):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    original = NetCDF4ArrayWrapper.__getitem__
    reads = []

    def read(array, key):
        assert array.variable_name not in {
            "Data.Toltec.Is",
            "Data.Toltec.Qs",
            "Header.Toltec.Xloc",
            "Header.Toltec.Yloc",
            "Header.Toltec.AdcSnapData",
        }
        reads.append(array.variable_name)
        return original(array, key)

    with open_toltec_kids(path, chunks=chunks) as source:
        monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", read)
        acquisition = source.toltec_kids.meta
        assert type(acquisition) is ToltecRawObsMetadata
        assert acquisition is source.toltec_kids.meta
        assert not any(name.startswith("Data.") for name in reads)
        assert "Header.Toltec.ToneFreq" not in reads
        assert acquisition.n_sweepreps == (10 if suffix else None)
        assert acquisition.n_blocks_max == source.sizes["numSweeps"]
        reads.clear()
        identified = source.toltec_kids.block_index
        assert type(identified.meta) is ToltecRawObsBlocksMetadata
        for name in ToltecRawObsMetadata.model_fields:
            assert getattr(identified.meta, name) == getattr(acquisition, name), name
        expected_blocks = 2 if suffix == "tune" else 1
        assert identified.meta.n_blocks == expected_blocks
        assert identified.index.sample_end[-1].item() == acquisition.n_times
        assert set(reads) <= {
            "Header.Toltec.NumSweepSteps",
            "Header.Toltec.ToneFreq",
            "Header.Toltec.ToneMask",
            "Header.Toltec.ToneAmp",
            "Header.Toltec.TonePhase",
            "Data.Toltec.LoFreq",
        }
        assert reads.count("Data.Toltec.LoFreq") == (1 if suffix else 2)
        assert not source["Data.Toltec.Is"].variable._in_memory


@pytest.mark.parametrize(
    ("header", "error_type"),
    [("ObsType", ValueError), ("DriveAtten", ValueError), ("Filename", KeyError)],
)
def test_acquisition_metadata_requires_raw_fields(
    real_targsweep_file, header, error_type
):
    name = f"Header.Toltec.{header}"
    with (
        open_toltec_kids(real_targsweep_file) as data,
        pytest.raises(error_type, match=name),
    ):
        _ = data.drop_vars(name).toltec_kids.meta


def test_programmed_repetitions_are_required_in_sweep_acquisitions(real_targsweep_file):
    with open_toltec_kids(real_targsweep_file) as data:
        changed = data.drop_vars("Header.Toltec.NumSamplesPerSweepStep")
        with pytest.raises(ValueError, match="NumSamplesPerSweepStep"):
            _ = changed.toltec_kids.meta


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_capacity_required_for_all_raw_acquisitions(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec_kids(path) as data, pytest.raises(KeyError, match="numSweeps"):
        _ = data.drop_dims("numSweeps").toltec_kids.meta


def test_acquisition_metadata_rejects_a_block_indexed_raw_source(real_targsweep_file):
    with open_toltec_kids(real_targsweep_file) as data:
        indexed = data.assign_coords(block=[0])
        with pytest.raises(ValueError, match="unpartitioned acquisition"):
            _ = indexed.toltec_kids.meta


def test_processed_sweep_without_parent_opens_with_unavailable_metadata(tmp_path):
    path = tmp_path / "renamed.nc"
    xr.Dataset(
        attrs={"contract": "kids.reduction/1", "kind": "ReducedSweep"}
    ).to_netcdf(path)
    with open_toltec_kids(path) as source:
        assert source.toltec_kids.meta is None
    file_accessor = FileAccessor.from_sources([path], mode="guess_or_read")
    assert file_accessor.table.data_kind_toltec.tolist() == [
        ToltecDataKind.reduced_sweep.value
    ]
    kinds = {
        entry.kind.value for entry in file_accessor.source_trace.iloc[0].iter_entries()
    }
    assert "metadata_unavailable" in kinds


def test_open_toltec_kids_forwards_decoder_options(real_targsweep_file, monkeypatch):
    options = []
    open_dataset = xr.open_dataset

    def open_with_options(path, **kwargs):
        options.append((kwargs["decode_coords"], kwargs["decode_times"]))
        return open_dataset(path, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", open_with_options)
    with open_toltec_kids(
        real_targsweep_file, decode_coords=False, decode_times=False
    ) as source:
        assert source.toltec_kids.data_kind & ToltecDataKind.raw_sweep
    assert options == [(False, False)]
