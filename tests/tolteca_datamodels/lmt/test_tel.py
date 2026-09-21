"""Telescope metadata and lazy pointing-array access."""

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from pydantic import TypeAdapter
from tollan.utils.fileloc import fileloc
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.lmt import (
    LmtObsGoalType,
    LmtTelAccessor,
    LmtTelIOMapper,
    LmtTelMetadata,
    open_lmt_tel,
)


@pytest.fixture
def telescope():
    return xr.Dataset(
        {
            "Header.File.Name": b"/data/tel_toltec2_2025-10-31_000010_00_0001.nc  ",
            "Header.Dcs.ObsNum": 10,
            "Header.Dcs.SubObsNum": 0,
            "Header.Dcs.ScanNum": 1,
            "Header.TelescopeBackend.Master": 0,
            "Header.Dcs.ObsPgm": b"Lissajous  \x00",
            "Header.Dcs.ObsGoal": b"Pointing  \x00",
            "Header.Dcs.IntegrationTime": 0.0,
            "Header.Source.Ra": ("position", [1.0, 1.1], {"units": "rad"}),
            "Header.Source.Dec": ("position", [0.5, 0.6], {"units": "rad"}),
            "Data.TelescopeBackend.TelTime": ("sample", [1761902400.0, 1761902401.0]),
            "Data.TelescopeBackend.TelAzAct": ("sample", [2.0, 2.1], {"units": "rad"}),
            "Data.TelescopeBackend.TelElAct": ("sample", [0.9, 0.8], {"units": "rad"}),
            "Data.TelescopeBackend.TelAzDes": ("sample", [2.2, 2.3], {"units": "rad"}),
            "Data.TelescopeBackend.TelElDes": ("sample", [0.7, 0.6], {"units": "rad"}),
            "Data.TelescopeBackend.TelAzMap": ("sample", [0.1, 0.2], {"units": "rad"}),
            "Data.TelescopeBackend.TelElMap": ("sample", [0.3, 0.4], {"units": "rad"}),
            "Data.TelescopeBackend.SourceRaAct": (
                "sample",
                [1.0, 1.01],
                {"units": "rad"},
            ),
            "Data.TelescopeBackend.SourceDecAct": (
                "sample",
                [0.5, 0.51],
                {"units": "rad"},
            ),
            "Data.TelescopeBackend.Hold": ("sample", [0.0, 1.0], {"units": "boolean"}),
        },
    )


@pytest.mark.parametrize("as_tree", [False, True])
def test_metadata_and_arrays(telescope, as_tree):
    obj = xr.DataTree(telescope) if as_tree else telescope
    acc = LmtTelAccessor(obj)
    meta = acc.meta
    assert acc.meta is meta
    assert isinstance(meta, LmtTelMetadata)
    assert (meta.master, meta.master_str, meta.interface, meta.instru) == (
        0,
        "tcs",
        "tel",
        "lmt",
    )
    assert meta.instru_component == "tcs"
    assert meta.obs_pgm == "Lissajous"
    assert (meta.obsnum, meta.subobsnum, meta.scannum) == (10, 0, 1)
    assert meta.obs_goal == LmtObsGoalType.pointing
    assert meta.t0.unix == 1761902400.0
    assert meta.t1.unix == 1761902401.0
    assert meta.t_exp == 0 * u.s  # Not replaced by the elapsed span.
    assert meta.t_elapsed == (meta.t1 - meta.t0).to(u.s)
    assert meta.t_elapsed != meta.t_exp
    assert meta.target is not None and meta.target_ref is not None
    assert meta.target.ra.rad == pytest.approx(1.0)
    assert meta.target_ref.ra.rad == pytest.approx(1.1)
    for name, variable in (
        ("tel_time", "TelTime"),
        ("az", "TelAzAct"),
        ("alt", "TelElAct"),
        ("az_des", "TelAzDes"),
        ("alt_des", "TelElDes"),
        ("az_map", "TelAzMap"),
        ("alt_map", "TelElMap"),
        ("ra", "SourceRaAct"),
        ("dec", "SourceDecAct"),
    ):
        xr.testing.assert_identical(
            getattr(acc, name), telescope[f"Data.TelescopeBackend.{variable}"]
        )
    np.testing.assert_array_equal(acc.t.to_numpy(), [0.0, 1.0])
    assert acc.t.attrs["units"] == "s"
    assert acc.t is acc.t
    np.testing.assert_array_equal(acc.hold_flag.to_numpy(), [0, 1])
    assert acc.hold_flag.dtype == np.dtype(int)
    assert "units" not in acc.hold_flag.attrs
    assert telescope["Data.TelescopeBackend.Hold"].dtype == np.dtype(float)
    assert telescope["Data.TelescopeBackend.Hold"].attrs["units"] == "boolean"
    adapter = TypeAdapter(LmtTelMetadata)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert restored.t0 == meta.t0
    assert restored.obs_goal == meta.obs_goal
    assert restored.master == 0
    assert restored.master_str == "tcs"
    assert restored.filepath_orig == meta.filepath_orig
    assert restored.filepath_info_orig == meta.filepath_info_orig
    assert restored.target is not None
    assert restored.target.separation(meta.target).to_value(u.rad) == pytest.approx(0)


@pytest.mark.parametrize(
    "name",
    [
        "Header.Source.Ra",
        "Header.Source.Dec",
        "Header.Dcs.ObsGoal",
        "Header.Dcs.ObsPgm",
        "Header.Dcs.ObsNum",
        "Header.Dcs.SubObsNum",
        "Header.Dcs.ScanNum",
    ],
)
def test_established_headers_are_required(telescope, name):
    with pytest.raises((KeyError, ValueError), match=name):
        _ = telescope.drop_vars(name).lmt_tel.meta


def test_header_attributes(telescope):
    ds = telescope
    for name in ("Header.Dcs.ObsPgm", "Header.Dcs.ObsGoal"):
        ds.attrs[name] = ds[name].item()
        ds = ds.drop_vars(name)
    assert ds.lmt_tel.meta.interface == "tel"
    assert ds.lmt_tel.meta.obsnum == 10
    assert ds.lmt_tel.meta.obs_goal == LmtObsGoalType.pointing


def test_headers_are_authoritative_and_check_is_opt_in(telescope):
    telescope["Header.Dcs.ObsNum"] = 999
    telescope.encoding["source"] = "/copied/tel_toltec_2025-10-30_000123_02_0003.nc"
    meta = telescope.lmt_tel.meta
    assert (meta.obsnum, meta.subobsnum, meta.scannum) == (999, 0, 1)
    assert meta.filepath_orig == "/data/tel_toltec2_2025-10-31_000010_00_0001.nc"
    assert meta.filepath_info_orig.obsnum == 10
    assert meta.filepath_info_orig.ut.isoformat() == "2025-10-31T00:00:00+00:00"
    with pytest.raises(ValueError, match="Conflicting obsnum.*Header.Dcs.ObsNum"):
        telescope.lmt_tel.mapper.get_metadata(telescope, check_filepath_orig=True)


@pytest.mark.parametrize(
    ("name", "header"),
    [
        ("obsnum", "Header.Dcs.ObsNum"),
        ("subobsnum", "Header.Dcs.SubObsNum"),
        ("scannum", "Header.Dcs.ScanNum"),
        ("master", "Header.TelescopeBackend.Master"),
    ],
)
def test_original_filename_consistency_check(telescope, name, header):
    telescope["Header.File.Name"] = (
        "/data_lmt/toltec/tcs/tel/tel_toltec_2025-10-31_000010_00_0001.nc"
    )
    telescope[header] = telescope[header] + 1
    acc = telescope.lmt_tel
    assert getattr(acc.meta, name) == telescope[header].item()
    with pytest.raises(ValueError, match=f"Conflicting {name}.*{header}"):
        acc.mapper.get_metadata(acc.data_source, check_filepath_orig=True)


def test_missing_master_is_explicitly_unknown(telescope):
    ds = telescope.drop_vars("Header.TelescopeBackend.Master")
    meta = ds.lmt_tel.mapper.get_metadata(ds, check_filepath_orig=True)
    assert meta.master is None
    assert meta.master_str is None


def test_invalid_master_is_not_missing(telescope):
    telescope["Header.TelescopeBackend.Master"] = 99
    with pytest.raises(ValueError):
        _ = telescope.lmt_tel.meta


def test_metadata_does_not_parse_original_filename_by_default(telescope):
    telescope["Header.File.Name"] = "unrecognized.nc"
    acc = telescope.lmt_tel
    assert acc.meta.filepath_orig == "unrecognized.nc"
    assert "filepath_info_orig" not in acc.meta.__dict__
    with pytest.raises(ValueError):
        acc.mapper.get_metadata(acc.data_source, check_filepath_orig=True)


@pytest.mark.parametrize("chars", [False, True])
def test_fixed_width_text_headers(telescope, chars):
    filename = b"/data/tel_toltec2_2025-10-31_000010_00_0001.nc \x00"
    goal = b"Pointing \x00"
    for name, text, dim in (
        ("Header.File.Name", filename, "filename_char"),
        ("Header.Dcs.ObsGoal", goal, "goal_char"),
    ):
        values = np.frombuffer(text, dtype="S1") if chars else np.array([text])
        telescope[name] = (dim, values)
    meta = telescope.lmt_tel.meta
    assert meta.interface == "tel"
    assert meta.filepath_orig == filename.decode().strip(" \x00")
    assert meta.obs_goal == LmtObsGoalType.pointing


def test_dask_time_and_flags_do_not_compute_or_modify_source(telescope):
    ds = telescope.chunk()
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        t = ds.lmt_tel.t
        flags = ds.lmt_tel.hold_flag
    assert not tasks
    assert t is ds.lmt_tel.t
    assert flags is ds.lmt_tel.hold_flag
    np.testing.assert_array_equal(t.compute(), [0.0, 1.0])
    np.testing.assert_array_equal(flags.compute(), [0, 1])
    xr.testing.assert_identical(ds.compute(), telescope)


@pytest.mark.parametrize("as_header", [False, True])
def test_remote_source_metadata_does_not_open_source(telescope, monkeypatch, as_header):
    source = (
        "https://archive.example/tel_toltec2_2025-10-31_000010_00_0001.nc?key=a%26b"
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Metadata parsing attempted to open a remote source")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    if as_header:
        telescope["Header.File.Name"] = source
    else:
        telescope.encoding["source"] = fileloc(source)
    meta = telescope.lmt_tel.meta
    assert meta.filepath_orig == (
        source if as_header else "/data/tel_toltec2_2025-10-31_000010_00_0001.nc"
    )
    assert meta.filepath_info_orig.obsnum == 10
    assert meta.obsnum == 10
    adapter = TypeAdapter(LmtTelMetadata)
    restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
    assert restored.filepath_orig == meta.filepath_orig


def test_current_source_is_separate_from_metadata(telescope, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = fileloc("tel_toltec_2025-10-31_000010_00_0001.nc")
    telescope.encoding["source"] = source
    meta = telescope.lmt_tel.meta
    assert telescope.encoding["source"] is source
    assert source.root.local_parent_path is None
    assert source.root.remote_parent_path is None
    assert not hasattr(meta, "source")

    adapter = TypeAdapter(LmtTelMetadata)
    payload = adapter.dump_json(meta, round_trip=True)
    monkeypatch.chdir(tmp_path.parent)
    restored = adapter.validate_json(payload)
    assert restored.filepath_orig == meta.filepath_orig
    assert restored.filepath_info_orig == meta.filepath_info_orig
    assert telescope.encoding["source"] is source
    assert source.root.local_parent_path is None
    assert source.root.remote_parent_path is None


@pytest.mark.parametrize("as_url", [False, True])
def test_open_local_fileloc(data_lmt_path, as_url):
    path = next((data_lmt_path / "tel").glob("tel*.nc"))
    source = path.as_uri() if as_url else fileloc(path)
    with open_lmt_tel(source) as ds:
        assert ds.encoding["source"] == str(path)
        assert ds.lmt_tel.meta.filepath_orig == f"/data_lmt/tel/{path.name}"


@pytest.mark.parametrize(
    "source", ["host:/data/test.nc", "https://example.org/test.nc"]
)
def test_remote_open_requires_explicit_staging(monkeypatch, source):
    def unexpected(*args, **kwargs):
        pytest.fail("Remote source was passed to the local reader")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    with pytest.raises(ValueError, match="staged locally"):
        open_lmt_tel(source)
    with pytest.raises(ValueError, match="staged locally"):
        open_lmt_tel(fileloc(source))


def test_pointing_variable_is_not_shadowed_by_attribute(telescope):
    name = "Data.TelescopeBackend.TelAzAct"
    telescope.attrs[name] = "not pointing data"
    xr.testing.assert_identical(telescope.lmt_tel.az, telescope[name])


@pytest.mark.parametrize("chunks", [None, {}])
def test_reference_native_arrays_are_lazy(data_lmt_path, monkeypatch, chunks):
    def unexpected(*args, **kwargs):
        pytest.fail("Accessing a native telescope array read its values")

    for path in sorted((data_lmt_path / "tel").glob("tel*.nc")):
        with open_lmt_tel(path, chunks=chunks) as ds, monkeypatch.context() as patch:
            patch.setattr(NetCDF4ArrayWrapper, "__getitem__", unexpected)
            acc = ds.lmt_tel
            for name, variable in (
                ("tel_time", "TelTime"),
                ("az", "TelAzAct"),
                ("alt", "TelElAct"),
                ("az_des", "TelAzDes"),
                ("alt_des", "TelElDes"),
                ("az_map", "TelAzMap"),
                ("alt_map", "TelElMap"),
                ("ra", "SourceRaAct"),
                ("dec", "SourceDecAct"),
            ):
                array = getattr(acc, name)
                assert array.name == f"Data.TelescopeBackend.{variable}"
                assert not array.variable._in_memory
                assert array.shape == ds[array.name].shape
                assert array.dtype == ds[array.name].dtype
            assert "meta" not in acc.__dict__


@pytest.mark.parametrize("missing", ["Header.File.Name", "Header.Dcs.IntegrationTime"])
def test_required_header_missing(telescope, missing):
    with pytest.raises((KeyError, ValueError), match=missing):
        _ = telescope.drop_vars(missing).lmt_tel.meta


def test_empty_time(telescope):
    with pytest.raises(ValueError, match="nonempty"):
        _ = telescope.isel(sample=slice(0, 0)).lmt_tel.meta


@pytest.mark.parametrize("goal", ["AzScan", "ElScan", "M3Offset", "", "new-goal"])
def test_goal_values_are_not_reinterpreted(telescope, goal):
    telescope["Header.Dcs.ObsGoal"] = goal
    assert telescope.lmt_tel.meta.obs_goal == goal.lower()


@pytest.mark.parametrize("size", [0, 1, 3])
def test_main_and_reference_coordinates_required(telescope, size):
    ds = telescope.drop_dims("position")
    ds["Header.Source.Ra"] = ("position", np.zeros(size))
    ds["Header.Source.Dec"] = ("position", np.zeros(size))
    with pytest.raises(ValueError, match="two main/reference"):
        _ = ds.lmt_tel.meta


def test_reference_telescope_files(data_lmt_path, monkeypatch):
    paths = sorted((data_lmt_path / "tel").glob("tel*.nc"))
    assert paths, "Reference fixture must contain telescope files"
    original = NetCDF4ArrayWrapper.__getitem__
    reads = []

    def read_header_or_endpoint(array, key):
        if array.variable_name.startswith("Data."):
            assert array.variable_name == "Data.TelescopeBackend.TelTime"
            assert isinstance(key.tuple[0], int)
        reads.append((array.variable_name, key))
        return original(array, key)

    for path in paths:
        # Independent direct-file expectations, including the v2 coordinate layout.
        with xr.open_dataset(path) as ds:
            expected = {
                name: ds[f"Header.Dcs.{header}"].item()
                for name, header in (
                    ("obsnum", "ObsNum"),
                    ("subobsnum", "SubObsNum"),
                    ("scannum", "ScanNum"),
                    ("t_exp", "IntegrationTime"),
                )
            }
            ra = ds["Header.Source.Ra"].values
            dec = ds["Header.Source.Dec"].values
            goal = ds["Header.Dcs.ObsGoal"].item().decode().strip().lower()
            times = ds["Data.TelescopeBackend.TelTime"].isel(time=[0, -1]).values
            expected_az = (
                ds["Data.TelescopeBackend.TelAzAct"].isel(time=slice(0, 3)).values
            )
        reads.clear()
        with monkeypatch.context() as patch:
            patch.setattr(NetCDF4ArrayWrapper, "__getitem__", read_header_or_endpoint)
            with open_lmt_tel(path) as ds:
                acc = ds.lmt_tel
                assert not reads  # Mapper/accessor construction is structural only.
                meta = acc.meta
                assert sorted(name for name, _ in reads) == sorted(
                    [
                        "Data.TelescopeBackend.TelTime",
                        "Data.TelescopeBackend.TelTime",
                        "Header.Source.Ra",
                        "Header.Source.Dec",
                        "Header.Dcs.ObsPgm",
                        "Header.Dcs.ObsGoal",
                        "Header.Dcs.IntegrationTime",
                        "Header.Dcs.ObsNum",
                        "Header.Dcs.SubObsNum",
                        "Header.Dcs.ScanNum",
                        "Header.File.Name",
                        "Header.TelescopeBackend.Master",
                    ]
                )
                reads.clear()
                assert meta is acc.meta
                assert meta.t_elapsed == (meta.t1 - meta.t0).to(u.s)
                assert meta.t_elapsed is meta.t_elapsed
                assert not reads
                assert meta.master == 0
                assert meta.master_str == "tcs"
                assert meta.interface == "tel"
                assert meta.instru_component == "tcs"
                assert meta.obsnum == expected["obsnum"]
                assert meta.subobsnum == expected["subobsnum"]
                assert meta.scannum == expected["scannum"]
                assert meta.t_exp == expected["t_exp"] * u.s
                assert meta.obs_goal == goal
                assert meta.filepath_info_orig.obsnum == meta.obsnum
                assert meta.filepath_info_orig is meta.filepath_info_orig
                assert meta.target is not None and meta.target_ref is not None
                np.testing.assert_allclose(
                    [meta.t0.unix, meta.t1.unix], times, rtol=0, atol=1e-6
                )
                np.testing.assert_allclose(
                    [meta.target.ra.rad, meta.target_ref.ra.rad], ra
                )
                np.testing.assert_allclose(
                    [meta.target.dec.rad, meta.target_ref.dec.rad], dec
                )
                assert not acc.az.variable._in_memory
                assert not reads
        with open_lmt_tel(path) as ds:
            np.testing.assert_array_equal(
                ds.lmt_tel.az.isel(time=slice(0, 3)).values, expected_az
            )
            mapper = LmtTelIOMapper.from_data_source(ds)
            assert (
                mapper.get_metadata(ds, check_filepath_orig=True).obsnum
                == expected["obsnum"]
            )
