"""Telescope metadata and lazy pointing-array access."""

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from pydantic import TypeAdapter
from tollan.utils.fileloc import fileloc
from tollan.utils.log import logger
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt import (
    LmtTelAccessor,
    open_lmt_tel,
)
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt.metadata import LmtTelMetadata
from tolteca_datamodels.lmt.tel import LmtTelIOMapper
from tolteca_datamodels.lmt.types import (
    LmtObsGoal,
    LmtObsPgm,
    LmtRawObsMasterCode,
)
from tolteca_datamodels.toltec.file import ToltecFileWorkflow
from tolteca_dummy_instru.file import (
    DummyInstruTelescopeFileWorkflow,
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
            "Header.Dcs.Receiver": b"Toltec  \x00",
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
    assert meta.master_code is LmtRawObsMasterCode.TCS
    assert meta.master == "tcs"
    assert (meta.master_code, meta.master, meta.interface, meta.instru) == (
        0,
        "tcs",
        "tel",
        "lmt",
    )
    assert meta.interface_kind == "tel"
    assert meta.data_models == {"lmt", "toltec"}
    assert not hasattr(acc, "data_models")
    assert meta.obs_pgm == "lissajous"
    assert (meta.obsnum, meta.subobsnum, meta.scannum) == (10, 0, 1)
    assert meta.obs_goal == LmtObsGoal.pointing
    assert meta.t0.unix == 1761902400.0
    assert meta.t1.unix == 1761902401.0
    assert meta.t_int == 0 * u.s  # Not replaced by the elapsed span.
    assert meta.t_elapsed == (meta.t1 - meta.t0).to(u.s)
    assert meta.t_elapsed != meta.t_int
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
    assert restored.master_code == 0
    assert restored.master == "tcs"
    assert restored.filepath_orig == meta.filepath_orig
    assert restored.filepath_info_orig == meta.filepath_info_orig
    assert restored.target is not None
    assert restored.target.separation(meta.target).to_value(u.rad) == pytest.approx(0)


def test_metadata_debug_log(data_lmt_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    messages = []
    sink = logger.add(
        messages.append, level="DEBUG", filter="tolteca_datamodels.lmt.tel"
    )
    try:
        with open_lmt_tel(path) as ds:
            meta = ds.lmt_tel.meta
            assert ds.lmt_tel.meta is meta
    finally:
        logger.remove(sink)

    assert any(meta.uid_raw_obs_file in msg.record["message"] for msg in messages)


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
        "Header.TelescopeBackend.Master",
        "Header.Dcs.Receiver",
    ],
)
def test_established_headers_are_required(telescope, name):
    with pytest.raises((KeyError, ValueError), match=name):
        _ = telescope.drop_vars(name).lmt_tel.meta


def test_header_attributes(telescope):
    ds = telescope
    for name in (
        "Header.Dcs.ObsPgm",
        "Header.Dcs.ObsGoal",
        "Header.Dcs.Receiver",
    ):
        ds.attrs[name] = ds[name].item()
        ds = ds.drop_vars(name)
    assert ds.lmt_tel.meta.interface == "tel"
    assert ds.lmt_tel.meta.obsnum == 10
    assert ds.lmt_tel.meta.obs_goal == LmtObsGoal.pointing
    identified = ToltecFileWorkflow._source_reader_rules.identify(ds)
    assert identified is not None
    assert identified is LmtTelAccessor
    assert identified(ds).data_source is ds


def test_headers_are_authoritative(telescope):
    telescope["Header.Dcs.ObsNum"] = 999
    telescope.encoding["source"] = "/copied/tel_toltec_2025-10-30_000123_02_0003.nc"
    meta = telescope.lmt_tel.meta
    assert (meta.obsnum, meta.subobsnum, meta.scannum) == (999, 0, 1)
    assert meta.filepath_orig == "/data/tel_toltec2_2025-10-31_000010_00_0001.nc"
    assert meta.filepath_info_orig.obsnum == 10
    assert meta.filepath_info_orig.ut.isoformat() == "2025-10-31T00:00:00+00:00"


@pytest.mark.parametrize("master_code", [1, 2, 3, 99])
def test_invalid_master_is_not_missing(telescope, master_code):
    telescope["Header.TelescopeBackend.Master"] = master_code
    with pytest.raises(ValueError):
        _ = telescope.lmt_tel.meta


def test_metadata_does_not_parse_original_filename_by_default(telescope, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Reading telescope metadata parsed the original filename")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    telescope["Header.File.Name"] = "unrecognized.nc"
    acc = telescope.lmt_tel
    assert acc.meta.filepath_orig == "unrecognized.nc"


@pytest.mark.parametrize("as_tree", [False, True])
def test_instrument_identification_is_independent_of_filenames(
    telescope, as_tree, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("Receiver identification requested full metadata")

    monkeypatch.setattr(LmtTelIOMapper, "get_metadata", unexpected)
    telescope["Header.File.Name"] = "unrecognized.nc"
    telescope.encoding["source"] = "tel_dummy_instru_2025-10-31_000010_00_0001.nc"
    source = xr.DataTree(telescope) if as_tree else telescope
    identified = ToltecFileWorkflow._source_reader_rules.identify(source)
    assert identified is not None
    reader = identified(source)
    assert isinstance(reader, LmtTelAccessor)
    assert (
        DummyInstruTelescopeFileWorkflow._source_reader_rules.identify(source) is None
    )
    assert ToltecFileWorkflow._source_reader_rules.identify(source) is identified
    assert isinstance(source.lmt_tel, type(reader))


@pytest.mark.parametrize("as_tree", [False, True])
def test_dcs_receiver_overrides_backend_receiver(telescope, as_tree):
    telescope["Header.TelescopeBackend.Receiver"] = "Dummy_Instru"
    source = xr.DataTree(telescope) if as_tree else telescope
    identified = ToltecFileWorkflow._source_reader_rules.identify(source)
    assert identified is not None
    reader = identified(source)
    assert isinstance(reader, LmtTelAccessor)
    assert reader.meta.data_models == {"lmt", "toltec"}
    assert (
        DummyInstruTelescopeFileWorkflow._source_reader_rules.identify(source) is None
    )


def test_missing_dcs_receiver_has_no_fallback(telescope):
    telescope["Header.TelescopeBackend.Receiver"] = "Toltec"
    source = telescope.drop_vars("Header.Dcs.Receiver")
    # General structural identification remains usable before metadata validation.
    assert isinstance(source.lmt_tel, LmtTelAccessor)
    assert ToltecFileWorkflow._source_reader_rules.identify(source) is None
    assert (
        DummyInstruTelescopeFileWorkflow._source_reader_rules.identify(source) is None
    )
    with pytest.raises(KeyError, match="Receiver"):
        _ = source.lmt_tel.meta


def test_reference_receiver_identification_reads_only_receiver_header(
    data_lmt_path, monkeypatch
):
    getitem = NetCDF4ArrayWrapper.__getitem__
    reads = []

    def tracked(array, key):
        reads.append(array.variable_name)
        return getitem(array, key)

    paths = list((data_lmt_path / "tel").glob("*.nc"))
    assert paths
    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", tracked)
    for path in paths:
        with xr.open_dataset(path) as data:
            reads.clear()
            identified = ToltecFileWorkflow._source_reader_rules.identify(data)
            assert identified is not None
            reader = identified(data)
            assert isinstance(reader, LmtTelAccessor)
            assert reads == ["Header.Dcs.Receiver"]
            assert reader.meta.data_models == {"lmt", "toltec"}
            assert not data["Data.TelescopeBackend.TelAzAct"].variable._in_memory


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
    assert meta.obs_goal == LmtObsGoal.pointing


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
        assert path.samefile(ds.encoding["source"])
        assert ds.lmt_tel.meta.filepath_orig == f"/data_lmt/tel/{path.name}"


@pytest.mark.parametrize("as_tree", [False, True])
@pytest.mark.parametrize("location", ["missing", "attr", "dim"])
def test_accessor_requires_native_timestamps(telescope, as_tree, location):
    name = "Data.TelescopeBackend.TelTime"
    ds = telescope.drop_vars(name)
    if location == "attr":
        ds.attrs[name] = [0.0, 1.0]
    elif location == "dim":
        ds["unrelated"] = (name, [0.0, 1.0])
    obj = xr.DataTree(ds) if as_tree else ds
    with pytest.raises(ValueError, match="missing tel_time variable"):
        _ = obj.lmt_tel


def test_accessor_checks_timestamp_presence_in_current_source(telescope):
    mapper = LmtTelIOMapper.from_data_source(telescope)
    source = telescope.drop_vars("Data.TelescopeBackend.TelTime")
    with pytest.raises(ValueError, match="missing tel_time variable"):
        LmtTelAccessor(source, mapper=mapper)


def test_open_rejects_other_netcdf(data_lmt_path):
    path = next((data_lmt_path / "toltec").rglob("*.nc"))
    with pytest.raises(ValueError, match="Not an LMT telescope dataset"):
        open_lmt_tel(path)


def test_rejected_open_closes_dataset(tmp_path, monkeypatch):
    ds = xr.Dataset()
    closed = []
    ds.set_close(lambda: closed.append(True))
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: ds)
    with pytest.raises(ValueError, match="missing tel_time variable"):
        open_lmt_tel(tmp_path / "not_tel.nc")
    assert closed == [True]


@pytest.mark.parametrize("error", [RuntimeError("unexpected"), KeyboardInterrupt()])
def test_open_closes_on_initialization_failure(telescope, tmp_path, monkeypatch, error):
    closed = []
    telescope.set_close(lambda: closed.append(True))
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: telescope)

    def fail(self):
        raise error

    monkeypatch.setattr(LmtTelAccessor, "_validate", fail)
    with pytest.raises(type(error)) as exc:
        open_lmt_tel(tmp_path / "tel.nc")
    assert exc.value is error
    assert closed == [True]


def test_open_forwards_options_and_closes_owned_dataset(
    telescope, tmp_path, monkeypatch
):
    opened = []
    closed = []
    telescope.set_close(lambda: closed.append(True))

    def track_open(path, **kwargs):
        opened.append((path, kwargs))
        return telescope

    monkeypatch.setattr(xr, "open_dataset", track_open)
    path = tmp_path / "renamed.nc"
    with open_lmt_tel(path, chunks={}) as ds:
        assert ds is telescope
        assert isinstance(ds.lmt_tel, LmtTelAccessor)
        assert not closed
    assert opened == [(path, {"chunks": {}})]
    assert closed == [True]


@pytest.mark.parametrize("condition", ["missing_header", "empty_time", "scalar_time"])
def test_open_defers_metadata_validation(telescope, tmp_path, monkeypatch, condition):
    if condition == "missing_header":
        ds = telescope.drop_vars("Header.Dcs.ObsNum")
        message = "Header.Dcs.ObsNum"
    elif condition == "empty_time":
        ds = telescope.isel(sample=slice(0, 0))
        message = "nonempty one-dimensional"
    else:
        ds = telescope.assign({"Data.TelescopeBackend.TelTime": 1761902400.0})
        message = "nonempty one-dimensional"
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: ds)
    with (
        open_lmt_tel(tmp_path / "tel.nc") as opened,
        pytest.raises(ValueError, match=message),
    ):
        _ = opened.lmt_tel.meta


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
def test_unrecognized_goals_are_rejected(telescope, goal):
    telescope["Header.Dcs.ObsGoal"] = goal
    with pytest.raises(ValueError, match="LmtObsGoal"):
        _ = telescope.lmt_tel.meta


@pytest.mark.parametrize("goal", list(LmtObsGoal))
def test_reader_constructs_goal_enum(telescope, goal):
    telescope["Header.Dcs.ObsGoal"] = goal.value.capitalize()
    assert telescope.lmt_tel.meta.obs_goal is goal


@pytest.mark.parametrize("program", ["", "new-program"])
def test_unrecognized_programs_are_rejected(telescope, program):
    telescope["Header.Dcs.ObsPgm"] = program
    with pytest.raises(ValueError, match="LmtObsPgm"):
        _ = telescope.lmt_tel.meta


def test_reader_constructs_recorded_program_enums(telescope, lmtmc_records):
    # Map is recorded in tel_toltec_2026-01-10_148435_00_0002.nc on Unity.
    for program in {row["ObsPgm"] for row in lmtmc_records} | {"Map"}:
        ds = telescope.assign({"Header.Dcs.ObsPgm": program})
        assert ds.lmt_tel.meta.obs_pgm is LmtObsPgm(program.lower())


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
                    ("t_int", "IntegrationTime"),
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
                        "Header.Dcs.Receiver",
                    ]
                )
                reads.clear()
                assert meta is acc.meta
                assert meta.t_elapsed == (meta.t1 - meta.t0).to(u.s)
                assert meta.t_elapsed is meta.t_elapsed
                assert not reads
                assert (meta.master_code, meta.master) == (0, "tcs")
                assert meta.uid_raw_obs_file == (
                    f"tcs-{meta.obsnum}-{meta.subobsnum}-{meta.scannum}-tel"
                )
                assert meta.interface == "tel"
                assert meta.interface_kind == "tel"
                assert meta.obsnum == expected["obsnum"]
                assert meta.subobsnum == expected["subobsnum"]
                assert meta.scannum == expected["scannum"]
                assert meta.t_int == expected["t_int"] * u.s
                assert meta.obs_goal is LmtObsGoal(goal)
                assert isinstance(meta.filepath_info_orig, LmtFilepathInfo)
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
            assert ds.lmt_tel.meta.obsnum == expected["obsnum"]
