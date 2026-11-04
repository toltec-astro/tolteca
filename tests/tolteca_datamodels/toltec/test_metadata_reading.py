"""Metadata values, header representations, and reference-file I/O boundaries."""

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from pydantic import TypeAdapter
from tollan.accessor import FieldLocation
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor, ToltecKidsIOMapper
from tolteca_datamodels.toltec.metadata import (
    ToltecRawObsBlocksMetadata,
    ToltecRawObsMetadata,
)
from tolteca_datamodels.toltec.toltec_db import ToltecDBRawObsMasterCode


@pytest.fixture
def raw(kids_readout_settings, kids_design_data):
    """Toy geometry/unit representations with recorded calibration settings."""
    return xr.Dataset(
        {
            **kids_readout_settings,
            **kids_design_data.data_vars,
            "Header.Toltec.Master": 1,
            "Header.Toltec.RoachIndex": 7,
            "Header.Toltec.ObsNum": 123,
            "Header.Toltec.SubObsNum": 2,
            "Header.Toltec.ScanNum": 3,
            "Header.Toltec.ObsType": 3,
            "Header.Toltec.ObsStartTime": 1761902400,
            "Header.Toltec.ObsEndTime": 1761902460,
            "Header.Toltec.SampleFreq": ((), 0.5, {"units": "kHz"}),
            "Header.Toltec.LoCenterFreq": ((), 600.0, {"units": "MHz"}),
            "Header.Toltec.NumSweepSteps": 3,
            "Header.Toltec.NumSamplesPerSweepStep": 2,
            "Header.Toltec.DriveAtten": 7.25,
            "Header.Toltec.SenseAtten": 4.0,
            "Header.Toltec.Filename": b"original.nc\x00 ",
            "Header.Toltec.RepeatLevel": 0,
            "Header.Toltec.ToneAmp": ("iqlen", np.ones(2)),
            "Data.Toltec.Is": (("time", "iqlen"), np.zeros((6, 2))),
            "Data.Toltec.Qs": (("time", "iqlen"), np.zeros((6, 2))),
            "Data.Toltec.LoFreq": ("time", [599.0, 599.0, 600.0, 600.0, 601.0, 601.0]),
        },
        coords={"numSweeps": [0, 1]},  # Toy capacity, independent of acquired blocks.
    )


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("as_tree", [False, True])
def test_named_dimensions_and_headers(raw, transpose, as_tree):
    ds = raw.transpose(...) if not transpose else raw.transpose("iqlen", "time", ...)
    obj = xr.DataTree(ds) if as_tree else ds
    meta = ToltecKidsAccessor(obj).meta
    assert isinstance(meta, ToltecRawObsMetadata)
    assert (meta.n_times, meta.n_chans) == (6, 2)
    assert meta.master_code == ToltecDBRawObsMasterCode.ICS
    assert meta.readout_code == 7
    assert (meta.roach, meta.nw, meta.cal_roach) == (7, 7, 7)
    assert meta.array_name == "a1400"
    assert meta.array == 1
    assert meta.obs_type == "targ"
    assert meta.t0.unix == 1761902400
    assert meta.t1.unix == 1761902460
    assert meta.t_int == 6 / 500 * u.s
    assert meta.f_smp == 500 * u.Hz
    assert meta.f_lo_center == 600e6
    assert meta.atten_drive == 7.25
    assert meta.filepath_orig == "original.nc"
    assert meta.repeat == 0
    adapter = TypeAdapter(ToltecRawObsMetadata)
    restored = adapter.validate_json(adapter.dump_json(meta))
    assert restored.f_smp == meta.f_smp
    assert restored.t0 == meta.t0
    assert restored.readout_code == meta.readout_code


@pytest.mark.parametrize("readout_code", [7, 11])
def test_readout_code_header_in_attributes(raw, readout_code):
    ds = raw.drop_vars("Header.Toltec.RoachIndex")
    ds.attrs["Header.Toltec.RoachIndex"] = readout_code
    meta = ds.toltec_kids.meta
    assert meta.readout_code == readout_code
    assert meta.interface == f"toltec{readout_code}"
    assert (meta.roach, meta.nw, meta.cal_roach) == (readout_code,) * 3


@pytest.mark.parametrize("master_code", [0, 1, 2])
def test_master_header_in_attributes(raw, master_code):
    ds = raw.drop_vars("Header.Toltec.Master")
    ds.attrs["Header.Toltec.Master"] = master_code
    assert ds.toltec_kids.meta.master_code == master_code
    assert ds.toltec_kids.meta.master == ("tcs", "ics", "clip")[master_code]


@pytest.mark.parametrize(
    "location", [FieldLocation.attr, FieldLocation.var, FieldLocation.dim]
)
def test_counts_use_mapped_locations_not_coordinate_values(raw, monkeypatch, location):
    ds = raw.assign_coords(
        time=[101, 102, 103, 104, 105, 106],
        iqlen=[201, 202],
        sweep=[-10, 0, 10],
    )
    ds.attrs.update(time=999, iqlen=999, sweep=999)
    name = "Header.Toltec.NumSweepSteps"
    if location != FieldLocation.var:
        ds = ds.drop_vars(name)
    if location == FieldLocation.attr:
        ds.attrs[name] = 3

    mapper = ToltecKidsIOMapper.from_data_source(ds)
    assert mapper.mapped_fields[mapper.schema.n_sweepsteps].location is location
    get_var = mapper._get_var

    def get_header(data_source, name):
        assert name not in {
            "time",
            "iqlen",
            "sweep",
            "Data.Toltec.Is",
            "Data.Toltec.Qs",
        }
        return get_var(data_source, name)

    monkeypatch.setattr(mapper, "_get_var", get_header)
    identified = mapper.get_blocks(ds)
    assert isinstance(identified.meta, ToltecRawObsBlocksMetadata)
    assert (
        identified.meta.n_times,
        identified.meta.n_chans,
        int(identified.index.n_sweepsteps[0]),
    ) == (6, 2, 3)


def test_declared_sweep_count_candidate_order(raw):
    raw.attrs["Header.Toltec.NumSweepSteps"] = 99
    assert raw.toltec_kids.block_index.index.n_sweepsteps.item() == 3


def test_unqualified_header_rejects_ambiguous_location(raw):
    raw.attrs["Header.Toltec.ObsNum"] = 999
    with pytest.raises(ValueError, match="Ambiguous field 'Header.Toltec.ObsNum'"):
        ToltecKidsIOMapper.from_data_source(raw)


@pytest.mark.parametrize(
    "missing",
    [
        "Header.Toltec.Master",
        "Header.Toltec.ObsNum",
        "Header.Toltec.SubObsNum",
        "Header.Toltec.ScanNum",
        "Header.Toltec.ObsStartTime",
        "Header.Toltec.ObsEndTime",
    ],
)
def test_required_metadata_is_not_fabricated(raw, missing):
    with pytest.raises(ValueError, match=missing):
        _ = raw.drop_vars(missing).toltec_kids.meta


def test_missing_raw_lo_center_is_not_inferred_from_samples(raw):
    missing = raw.drop_vars("Header.Toltec.LoCenterFreq")
    assert "Data.Toltec.LoFreq" in missing
    with pytest.raises(ValueError, match="Header.Toltec.LoCenterFreq"):
        _ = missing.toltec_kids.meta


def test_reference_metadata(data_lmt_path, monkeypatch):
    paths = sorted((data_lmt_path / "toltec/tcs/toltec0").glob("*.nc"))
    assert len(paths) >= 4
    original = NetCDF4ArrayWrapper.__getitem__
    reads = []

    def read_metadata(array, key):
        assert array.variable_name not in {
            "Data.Toltec.Is",
            "Data.Toltec.Qs",
            "Header.Toltec.AdcSnapData",
        }
        if array.variable_name.startswith("Data."):
            assert array.variable_name == "Data.Toltec.LoFreq"
        reads.append(array.variable_name)
        return original(array, key)

    for path in paths:
        with xr.open_dataset(path) as ds:
            expected = {
                name: ds[f"Header.Toltec.{name}"].item()
                for name in (
                    "RoachIndex",
                    "ObsStartTime",
                    "ObsEndTime",
                    "SampleFreq",
                    "DriveAtten",
                    "SenseAtten",
                    "LoCenterFreq",
                    "ObsType",
                    "TargSweepObsNum",
                )
            }
            sizes = dict(ds.sizes)
            n_blocks = 1
            if expected["ObsType"] == 4:
                n_blocks = 1 + np.count_nonzero(
                    np.diff(ds["Data.Toltec.LoFreq"].values) < 0
                )
        reads.clear()
        with monkeypatch.context() as patch:
            patch.setattr(NetCDF4ArrayWrapper, "__getitem__", read_metadata)
            with open_toltec_kids(path) as ds:
                acc = ds.toltec_kids
                assert reads == ["Header.Toltec.ObsType"]
                meta = acc.meta
                assert isinstance(meta, ToltecRawObsMetadata)
                assert meta is acc.meta
                assert meta.master_code == ToltecDBRawObsMasterCode.TCS
                assert meta.readout_code == expected["RoachIndex"]
                assert meta.n_times == sizes["time"]
                assert meta.n_chans == sizes["iqlen"]
                assert meta.n_kids_design == sizes["loclen"]
                assert meta.cal_obsnum == expected["TargSweepObsNum"]
                assert meta.t0.unix == expected["ObsStartTime"]
                assert meta.t1.unix == expected["ObsEndTime"]
                assert meta.atten_drive == expected["DriveAtten"]
                assert meta.atten_sense == expected["SenseAtten"]
                assert meta.f_lo_center == expected["LoCenterFreq"]
                assert meta.t_int.to_value(u.s) == pytest.approx(
                    sizes["time"] / expected["SampleFreq"]
                )
                assert not ds["Data.Toltec.Is"].variable._in_memory
                assert meta.n_blocks_max == sizes["numSweeps"]
                assert reads.count("Data.Toltec.LoFreq") == 0
                identified = acc.block_index
                assert identified.index.sizes["block"] == n_blocks
                assert (identified.meta.n_blocks, identified.meta.readout_code) == (
                    n_blocks,
                    meta.readout_code,
                )
                expected_lo_reads = 2 if expected["ObsType"] in (0, 1) else 1
                assert reads.count("Data.Toltec.LoFreq") == expected_lo_reads
                reads_before = list(reads)
                assert meta.t_elapsed == (meta.t1 - meta.t0).to(u.s)
                parsed = meta.filepath_info_orig
                assert parsed is not None
                assert parsed.model_dump()["roach"] == 0
                assert reads == reads_before


def test_tune_blocks_use_recorded_lo_resets(raw):
    raw["Header.Toltec.ObsType"] = 4
    raw["Data.Toltec.LoFreq"] = ("time", [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    mapper = ToltecKidsIOMapper.from_data_source(raw)
    meta = mapper.get_metadata(raw)
    assert isinstance(meta, ToltecRawObsMetadata)
    index = mapper.get_blocks(raw).index
    assert meta.n_times == 6
    assert index.sample_start.values.tolist() == [0, 3]
    assert index.sample_end.values.tolist() == [3, 6]


@pytest.mark.parametrize("n_blocks", [0, 1, 2])
def test_explicit_block_dimension_is_distinct_from_flat_input(raw, n_blocks):
    # Synthetic indexed layouts exercise dimension handling, not file identity.
    indexed = raw.assign_coords(block=np.arange(n_blocks))
    mapper = ToltecKidsIOMapper.from_data_source(indexed)
    identified = mapper.get_blocks(indexed)
    assert identified.meta.n_blocks == n_blocks
    assert identified.meta.n_times == n_blocks * raw.sizes["time"]
    assert identified.index.sizes["block"] == n_blocks


def test_empty_tune_has_no_acquired_blocks(raw):
    empty = raw.isel(time=slice(0, 0))
    empty["Header.Toltec.ObsType"] = 4
    assert empty.toltec_kids.meta.n_times == 0
    assert empty.toltec_kids.meta.n_blocks_max == 2
    assert empty.toltec_kids.block_index.index.sizes["block"] == 0


def test_acquisition_reuse_requires_raw_unpartitioned_source(raw):
    acquisition = raw.toltec_kids.meta
    processed = raw.assign_attrs({"Header.Kids.kind": "processed_sweep"})
    mapper = ToltecKidsIOMapper.from_data_source(processed)
    with pytest.raises(ValueError, match="unpartitioned raw source"):
        mapper.get_blocks(processed, acquisition=acquisition)
