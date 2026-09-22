"""Metadata values, header representations, and reference-file I/O boundaries."""

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from pydantic import TypeAdapter
from tollan.accessor import FieldLocation
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec import open_toltec
from tolteca_datamodels.toltec.kids import ToltecKidsIOMapper
from tolteca_datamodels.toltec.metadata import ToltecSweepMetadata
from tolteca_datamodels.toltec.toltecdb import ToltecDBRawObsMaster


@pytest.fixture
def raw():
    return xr.Dataset(
        {
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
            "Data.Toltec.Is": (("time", "iqlen"), np.zeros((6, 2))),
            "Data.Toltec.Qs": (("time", "iqlen"), np.zeros((6, 2))),
            "Data.Toltec.LoFreq": ("time", [599.0, 599.0, 600.0, 600.0, 601.0, 601.0]),
        },
    )


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("as_tree", [False, True])
def test_named_dimensions_and_headers(raw, transpose, as_tree):
    ds = raw.transpose(...) if not transpose else raw.transpose("iqlen", "time", ...)
    obj = xr.DataTree(ds) if as_tree else ds
    meta = obj.toltec_kids.meta
    assert (meta.n_times, meta.n_chans) == (6, 2)
    assert meta.master == ToltecDBRawObsMaster.ICS
    assert meta.array_name == "a1400"
    assert meta.t0.unix == 1761902400
    assert meta.t1.unix == 1761902460
    assert meta.t_exp == 6 / 500 * u.s
    assert meta.f_smp == 500 * u.Hz
    assert meta.f_lo_center == 600e6
    assert meta.atten_drive == 7.25
    assert meta.filepath_orig == "original.nc"
    assert meta.repeatvar == "0"
    adapter = TypeAdapter(ToltecSweepMetadata)
    restored = adapter.validate_json(adapter.dump_json(meta))
    assert restored.f_smp == meta.f_smp
    assert restored.t0 == meta.t0


@pytest.mark.parametrize("master", [0, 1, 2])
def test_master_header_in_attributes(raw, master):
    ds = raw.drop_vars("Header.Toltec.Master")
    ds.attrs["Header.Toltec.Master"] = master
    assert ds.toltec_kids.meta.master == master
    assert ds.toltec_kids.meta.master_str == ("tcs", "ics", "clip")[master]


@pytest.mark.parametrize("location", list(FieldLocation))
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
    meta = mapper.get_metadata(ds)
    assert isinstance(meta, ToltecSweepMetadata)
    assert (meta.n_times, meta.n_chans, meta.n_sweepsteps) == (6, 2, 3)


def test_declared_sweep_count_candidate_order(raw):
    raw.attrs["Header.Toltec.NumSweepSteps"] = 99
    assert raw.toltec_kids.meta.n_sweepsteps == 3


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


def test_missing_optional_headers_remain_unknown(raw):
    meta = raw.drop_vars(
        ["Header.Toltec.LoCenterFreq", "Header.Toltec.DriveAtten"]
    ).toltec_kids.meta
    assert meta.f_lo_center is None
    assert meta.atten_drive is None


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
            with open_toltec(path) as ds:
                acc = ds.toltec_kids
                assert not reads
                meta = acc.meta
                assert meta is acc.meta
                assert meta.master == ToltecDBRawObsMaster.TCS
                assert meta.n_times == sizes["time"]
                assert meta.n_chans == sizes["iqlen"]
                assert meta.n_kids_design == sizes["loclen"]
                assert meta.cal_obsnum == expected["TargSweepObsNum"]
                assert meta.t0.unix == expected["ObsStartTime"]
                assert meta.t1.unix == expected["ObsEndTime"]
                assert meta.atten_drive == expected["DriveAtten"]
                assert meta.atten_sense == expected["SenseAtten"]
                assert meta.t_exp.to_value(u.s) == pytest.approx(
                    sizes["time"] / expected["SampleFreq"]
                )
                assert not ds["Data.Toltec.Is"].variable._in_memory
                if isinstance(meta, ToltecSweepMetadata):
                    assert meta.f_lo_center == expected["LoCenterFreq"]
                    assert meta.n_blocks == n_blocks
                    assert meta.n_sweeps == n_blocks
                    assert meta.n_blocks_max == sizes["numSweeps"]
                assert reads.count("Data.Toltec.LoFreq") == (expected["ObsType"] == 4)


def test_tune_blocks_use_recorded_lo_resets(raw):
    raw["Header.Toltec.ObsType"] = 4
    raw["Data.Toltec.LoFreq"] = ("time", [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    mapper = ToltecKidsIOMapper.from_data_source(raw)
    meta = mapper.get_metadata(raw)
    assert isinstance(meta, ToltecSweepMetadata)
    assert meta.n_blocks == meta.n_sweeps == 2
    assert meta.n_times == 6
    for block in (0, 1):
        selected = mapper.get_metadata(raw, block=block)
        assert isinstance(selected, ToltecSweepMetadata)
        assert selected.n_times == 3
        assert selected.n_blocks == 2
        assert selected.n_sweeps == 1
    with pytest.raises(ValueError, match="out of range"):
        mapper.get_metadata(raw, block=2)
