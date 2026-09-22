"""Header-only acquisition records reuse reader values without scanning arrays."""

from dataclasses import fields

import pytest
from pydantic import TypeAdapter
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import (
    ToltecSweepMetadataHeader,
    ToltecTimeStreamMetadata,
    open_toltec,
)


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_header_metadata_matches_full_metadata_without_array_reads(
    data_lmt_path, monkeypatch, suffix
):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    original = NetCDF4ArrayWrapper.__getitem__
    reads = []

    def read(array, key):
        assert not array.variable_name.startswith("Data.")
        assert array.variable_name not in {
            "Header.Toltec.Xloc",
            "Header.Toltec.Yloc",
            "Header.Toltec.AdcSnapData",
        }
        reads.append(array.variable_name)
        return original(array, key)

    with open_toltec(path) as data:
        with monkeypatch.context() as patch:
            patch.setattr(NetCDF4ArrayWrapper, "__getitem__", read)
            header = data.toltec_kids.meta_header
            assert header is data.toltec_kids.meta_header
            assert reads.count("Header.Toltec.LoCenterFreq") == 1
        cls = ToltecSweepMetadataHeader if suffix else ToltecTimeStreamMetadata
        assert type(header) is cls
        assert not hasattr(header, "n_sweeps")
        assert not hasattr(header, "n_blocks")
        if suffix:
            assert header.n_blocks_max == data.sizes["numSweeps"]
        full = data.toltec_kids.meta
        for field in fields(header):
            assert getattr(header, field.name) == getattr(full, field.name), field.name
        assert not data["Data.Toltec.Is"].variable._in_memory
        adapter = TypeAdapter(cls)
        restored = adapter.validate_json(adapter.dump_json(header, round_trip=True))
        assert adapter.dump_python(restored, mode="json", round_trip=True) == (
            adapter.dump_python(header, mode="json", round_trip=True)
        )


@pytest.mark.parametrize(
    ("header", "error_type"),
    [
        ("ObsType", ValueError),
        ("DriveAtten", ValueError),
        ("Filename", KeyError),
        ("NumSamplesPerSweepStep", ValueError),
        ("AdcSnapData", KeyError),
    ],
)
def test_header_metadata_requires_raw_fields(real_targsweep_file, header, error_type):
    name = f"Header.Toltec.{header}"
    with open_toltec(real_targsweep_file) as data:
        changed = data.drop_vars(name)
        with pytest.raises(error_type, match=name):
            _ = changed.toltec_kids.meta_header


@pytest.mark.parametrize("layout", ["processed", "block"])
def test_header_metadata_requires_unpartitioned_raw_acquisition(
    real_targsweep_file, layout
):
    with open_toltec(real_targsweep_file) as data:
        changed = (
            data.assign_attrs({"Header.Kids.kind": "processed_sweep"})
            if layout == "processed"
            else data.expand_dims(block=[0])
        )
        with pytest.raises(ValueError, match="requires (an unpartitioned|a raw KIDs)"):
            _ = changed.toltec_kids.meta_header
