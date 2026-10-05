"""Dummy instrument discovery, metadata and resource ownership."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_dummy_instru import register
from tolteca_dummy_instru.data import (
    DummyInstruAccessor,
    identify_dummy_instru,
    open_dummy_instru,
)
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.file_store import DummyInstruFileStore
from tolteca_dummy_instru.metadata import DummyInstruMetadata
from tolteca_dummy_instru.testing import make_dataset


@pytest.fixture(autouse=True)
def enable_dummy(monkeypatch):
    # Keep the opt-in isolated from other test modules in the same process.
    import tolteca_dummy_instru
    from tolteca_datamodels import input_discovery

    input_models = dict(input_discovery._input_discovery_models)

    monkeypatch.setattr(tolteca_dummy_instru, "_registered", False)
    monkeypatch.setattr(
        LmtFileStore,
        "_substore_classes",
        dict(LmtFileStore._substore_classes),
    )
    register()
    yield
    input_discovery._input_discovery_models.clear()
    input_discovery._input_discovery_models.update(input_models)
    input_discovery.InputDiscoveryConfig.model_rebuild(force=True)


@pytest.fixture
def observation(tmp_path):
    path = tmp_path / "dummy_instru/chan01/chan01-tcs-1-0-2-20251031T120000.nc"
    meta = DummyInstruMetadata(
        interface="chan01",
        channel=1,
        master="tcs",
        obsnum=1,
        subobsnum=0,
        scannum=2,
        filepath_orig=str(path),
        t0=1761912000.0,
        t1=1761912003.0,
        t_int="3 s",
    )
    path.parent.mkdir(parents=True)
    make_dataset(meta, [1, 2, 4, 8]).to_netcdf(path)
    return path, meta


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
@pytest.mark.parametrize(
    "scope",
    [
        (LmtFileAccessor, {0}),
        (ToltecFileAccessor, {0, 1}),
        (DummyInstruFileAccessor, {0, 2}),
    ],
)
def test_accessor_scope_controls_guesses_and_metadata_attempts(
    scope, mode, observation, data_lmt_path, monkeypatch
):
    accessor_cls, identified = scope
    dummy_path, _ = observation
    sources = [
        data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        data_lmt_path
        / "toltec/tcs/toltec0/toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
        dummy_path,
        "unknown.nc",
    ]
    parsed, read_positions = [], []
    parse = FilepathInfo.from_source

    def trace_parse(source, **kwargs):
        assert kwargs["accepted_types"] == accessor_cls._filepath_info_types
        parsed.append(source)
        return parse(source, **kwargs)

    def read_metadata(self, rows, *, diagnostics):
        read_positions.extend(position for position, _ in rows)

    def unexpected(*args, **kwargs):
        pytest.fail("Filename projection opened a file")

    monkeypatch.setattr(FilepathInfo, "from_source", trace_parse)
    monkeypatch.setattr(accessor_cls, "_read_metadata", read_metadata)
    monkeypatch.setattr(xr, "open_dataset", unexpected)
    monkeypatch.setattr(accessor_cls, "_source_opener", xr.open_dataset)
    table = accessor_cls.from_sources(sources, mode=mode)
    assert parsed == table.source_loc.tolist()
    assert table.source.tolist() == sources
    assert set(table.index[table.obsnum.notna()]) == identified
    assert set(table.index[table.file_prefix.notna()]) == identified
    assert set(table.index[table.uid_raw_obs_file.notna()]) == identified
    expected_reads = {
        "guess": [],
        "guess_or_read": [i for i in range(len(sources)) if i not in identified],
        "read": list(range(len(sources))),
    }
    assert read_positions == expected_reads[mode]
    if accessor_cls is ToltecFileAccessor:
        assert table.data_kind.tolist() == [
            ToltecDataKind.lmt_tel.value,
            ToltecDataKind.targsweep.value,
            ToltecDataKind.unknown.value,
            ToltecDataKind.unknown.value,
        ]
    else:
        assert "data_kind" not in table


@pytest.mark.parametrize("accessor_cls", [LmtFileAccessor, ToltecFileAccessor])
def test_foreign_filename_does_not_prevent_header_identification(
    accessor_cls, observation, data_lmt_path
):
    dummy_path, _ = observation
    tel_path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(tel_path) as data:
        close = data._close
        table = accessor_cls.from_sources(
            [{"source": data, "source_loc": dummy_path}], mode="guess_or_read"
        )
        assert table.file_prefix.isna().all()
        assert table.obsnum.tolist() == [145609]
        assert table.interface.tolist() == ["tel"]
        assert table.uid_raw_obs_file.tolist() == ["tcs-145609-0-2-tel"]
        assert table.source.iloc[0] is data
        assert table.source_loc.iloc[0].path == dummy_path
        assert data._close is close


def test_foreign_header_stays_unidentified_in_toltec_table(observation):
    path, _ = observation
    table = ToltecFileAccessor.from_sources([path], mode="guess_or_read")
    assert table.source.tolist() == [path]
    assert table.file_prefix.isna().all()
    assert table.obsnum.isna().all()
    assert table.uid_raw_obs_file.isna().all()
    assert table.data_kind.tolist() == [ToltecDataKind.unknown.value]


@pytest.mark.parametrize("accessor_cls", [LmtFileAccessor, ToltecFileAccessor])
def test_open_rejects_other_instrument_after_permissive_read(accessor_cls, observation):
    path, _ = observation
    table = accessor_cls.from_sources([path], mode="read")
    assert table.obsnum.isna().all()
    with (
        pytest.raises(ValueError, match="No supported dataset recognized"),
        accessor_cls(table).open(on_error="raise"),
    ):
        pytest.fail("Dummy data was accepted by another instrument accessor")


def test_dummy_open_does_not_accept_kids(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc"))
    table = DummyInstruFileAccessor.from_sources([path], mode="read")
    assert table.obsnum.isna().all()
    with (
        pytest.raises(ValueError, match="No supported dataset recognized"),
        table.dummy_instru_file.open(on_error="raise"),
    ):
        pytest.fail("KIDs data was accepted by the dummy instrument accessor")


def test_explicit_registration_in_fresh_process():
    code = """
import sys
from tolteca_datamodels.input_discovery import InputDiscoveryConfig
from tolteca_datamodels.lmt.file_store import LmtFileStore
import tolteca_dummy_instru
assert 'dummy_instru' not in LmtFileStore('/data').substores
assert 'tolteca_dummy_instru.file' not in sys.modules
definitions = InputDiscoveryConfig.model_json_schema()['$defs']
assert 'DummyInstruInputDiscovery' not in definitions
tolteca_dummy_instru.register()
tolteca_dummy_instru.register()
store = LmtFileStore('/data').substores['dummy_instru']
assert store.path.as_posix() == '/data/dummy_instru'
config = InputDiscoveryConfig.model_validate({
    'inputs': [{'data_dummy_instru_path': '/data/dummy_instru'}]
})
entry = config.inputs[0]
assert type(entry).__name__ == 'DummyInstruInputDiscovery'
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test program.
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_guess_read_and_renamed_source(observation, tmp_path):
    path, meta = observation
    guessed = DummyInstruFileAccessor.from_sources([path])
    assert "t0_mjd" not in guessed
    assert guessed.interface.tolist() == ["chan01"]
    read = DummyInstruFileAccessor.from_sources([path], mode="read")
    assert read.uid_raw_obs_file.tolist() == [meta.uid_raw_obs_file]
    assert read.t_int_s.tolist() == [3.0]
    renamed = tmp_path / "renamed.nc"
    renamed.symlink_to(path)
    recovered = DummyInstruFileAccessor.from_sources([renamed], mode="guess_or_read")
    assert recovered.obsnum.tolist() == [1]
    assert recovered.file_prefix.isna().all()
    assert recovered.filepath_orig.tolist() == [str(path)]


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_registered_discovery_matches_typed_python(observation, tmp_path, mode):
    from tollan.config import RuntimeContext

    from tolteca_datamodels.input_discovery import InputDiscovery
    from tolteca_dummy_instru.input_discovery import DummyInstruInputDiscovery

    path, _ = observation
    runtime = RuntimeContext(
        {
            "input_discovery": {
                "inputs": [
                    {
                        "data_dummy_instru_path": "dummy_instru",
                        "obs_spec": "1/(1,3)",
                        "mode": mode,
                    }
                ]
            }
        }
    ).with_config(validation_context={"rootpath": tmp_path})
    handler = InputDiscovery(runtime)
    assert isinstance(handler.config.inputs[0], DummyInstruInputDiscovery)
    direct = DummyInstruInputDiscovery(
        data_dummy_instru_path=tmp_path / "dummy_instru",
        obs_spec=ObsSpec.parse("1/(1,3)"),
        mode=mode,
    ).discover()
    configured = handler.discover()
    pd.testing.assert_frame_equal(configured, direct)
    assert configured.source_loc.iloc[0].path == path
    assert "data_kind" not in configured
    assert "roach" not in configured


def test_lazy_samples_and_header_override(observation):
    path, meta = observation
    with open_dummy_instru(path) as data:
        assert identify_dummy_instru(data) is data.dummy_instru
        assert not data["counts"]._in_memory
        assert DummyInstruAccessor(data).meta == meta
        assert not data["counts"]._in_memory
        assert data.dummy_instru.counts.identical(data["counts"])
        data.attrs["obsnum"] = 7
        closed = []
        original_close = data._close
        assert original_close is not None

        def tracked_close():
            closed.append(True)
            original_close()

        data.set_close(tracked_close)
        table = DummyInstruFileAccessor.from_sources(
            [{"source": data, "source_loc": path}], mode="read"
        )
        assert table.obsnum.tolist() == [7]
        assert table.file_obsnum.tolist() == [1]
        assert table.uid_raw_obs.tolist() == ["tcs-7-0-2"]
        with table.dummy_instru_file.open() as opened:
            assert opened.io_obj.iloc[0] is data
        assert not closed
        assert "io_obj" not in opened


@pytest.mark.parametrize("spec", ["1/1", "1/chan01", "1/chan(1,3)", "1/(1,3)"])
def test_store_numeric_resolution_and_no_telescope_probe(observation, tmp_path, spec):
    path, _ = observation
    store = DummyInstruFileStore(tmp_path / "dummy_instru")
    assert store.find_candidates(spec) == [path]
    assert store.find_candidates("1/tel") == []
    table = DummyInstruFileAccessor.from_sources([path])
    assert len(table.dummy_instru_file.query_obs_spec(spec)) == 1


def test_root_delegates_and_selects_combined_table(observation, tmp_path, monkeypatch):
    path, _ = observation
    tel = tmp_path / "tel/tel_dummy_instru_2025-10-31_000001_00_0002.nc"
    tel.parent.mkdir()
    tel.touch()  # Filename-only discovery; no invented telescope header.
    store = LmtFileStore(tmp_path)
    candidates = store.find_candidates("1/{tel,1}", instru="dummy_instru")
    assert candidates == [tel, path]
    table = DummyInstruFileAccessor.from_sources(candidates)
    selected = table.dummy_instru_file.query_obs_spec("1/{tel,1}")
    assert selected.instru.tolist() == ["lmt", "dummy_instru"]
    assert selected.interface.tolist() == ["tel", "chan01"]
    assert "roach" not in selected and "data_kind" not in selected
    original = Path.glob
    probed = []

    def glob(directory, pattern):
        probed.append(directory)
        return original(directory, pattern)

    monkeypatch.setattr(Path, "glob", glob)
    assert store.find_candidates("1/{tel,1}", instru="toltec") == [tel]
    assert not any("dummy_instru" in directory.parts for directory in probed)


def test_open_closes_owned_resources_on_error(observation, monkeypatch):
    path, _ = observation
    table = DummyInstruFileAccessor.from_sources([path])
    closed = []
    open_dataset = xr.open_dataset

    def open_data(*args, **kwargs):
        data = open_dataset(*args, **kwargs)
        close = data._close
        assert close is not None

        def tracked_close():
            closed.append(True)
            close()

        data.set_close(tracked_close)
        return data

    monkeypatch.setattr(xr, "open_dataset", open_data)
    monkeypatch.setattr(DummyInstruFileAccessor, "_source_opener", xr.open_dataset)
    message = "consumer failed"
    with (  # noqa: PT012 -- exercise cleanup after a consumer raises inside the scope.
        pytest.raises(RuntimeError, match="consumer failed"),
        table.dummy_instru_file.open() as opened,
    ):
        assert len(opened.io_obj) == 1
        raise RuntimeError(message)
    assert closed == [True]
    assert "io_obj" not in table and "io_obj" not in opened


def test_open_reuses_duplicate_paths_and_nested_handles(observation, monkeypatch):
    path, meta = observation
    table = DummyInstruFileAccessor.from_sources([path, path])
    opened, closed = [], []
    open_dataset = xr.open_dataset

    def open_data(source):
        data = open_dataset(source)
        opened.append(data)
        close = data._close
        assert close is not None

        def tracked_close():
            closed.append(True)
            close()

        data.set_close(tracked_close)
        return data

    monkeypatch.setattr(xr, "open_dataset", open_data)
    monkeypatch.setattr(DummyInstruFileAccessor, "_source_opener", xr.open_dataset)
    with table.dummy_instru_file.open() as live:
        assert len(opened) == 1
        assert live.io_obj.iloc[0] is live.io_obj.iloc[1]
        with live.iloc[::-1].dummy_instru_file.open() as nested:
            assert nested.io_obj.iloc[0] is opened[0]
            assert nested.uid_raw_obs_file.tolist() == [meta.uid_raw_obs_file] * 2
        assert len(opened) == 1 and not closed
    assert closed == [True]
    assert "io_obj" not in live and "io_obj" not in nested


def test_remote_and_empty_tables():
    remote = "archive:/chan02-tcs-2-0-0-20251031T120000.nc"
    guessed = DummyInstruFileAccessor.from_sources([remote])
    read = DummyInstruFileAccessor.from_sources([remote], mode="read")
    pd.testing.assert_frame_equal(guessed, read)
    assert DummyInstruFileAccessor.from_sources([], mode="read").empty


def test_real_telescope_metadata_uses_shared_lmt_reader(observation, data_lmt_path):
    path, _ = observation
    telescope = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(telescope) as original:
        # This in-memory companion tests the declared dummy prefix, not a new
        # telescope layout. The reference input remains unchanged on disk.
        location = "tel_dummy_instru_2025-10-31_145609_00_0002.nc"
        companion = original.assign({"Header.File.Name": location})
        table = DummyInstruFileAccessor.from_sources(
            [path, {"source": companion, "source_loc": location}], mode="read"
        )
        assert table.instru.tolist() == ["dummy_instru", "lmt"]
        assert table.interface.tolist() == ["chan01", "tel"]
        assert table.channel.isna().tolist() == [False, True]
        assert table.obsnum.tolist() == [1, 145609]
        assert table.target_ra_deg.iloc[1] == companion.lmt_tel.meta.target.ra.deg
        assert companion.lmt_tel.meta.filepath_info_orig.prefix == "tel_dummy_instru"
        cached = companion.lmt_tel.meta
        companion["Header.Dcs.ObsNum"] = 145610
        refreshed = table.dummy_instru_file.read_metadata()
        assert refreshed.obsnum.tolist() == [1, 145610]
        assert companion.lmt_tel.meta is cached


def test_dummy_reader_rereads_headers_without_replacing_cached_metadata(observation):
    _, meta = observation
    data = make_dataset(meta, [1, 2])
    reader = identify_dummy_instru(data)
    cached = reader.meta
    data.attrs["obsnum"] = cached.obsnum + 1
    assert reader.read_metadata().obsnum == cached.obsnum + 1
    assert reader.meta is cached


def test_invalid_integer_header_is_not_truncated(observation):
    _, meta = observation
    data = make_dataset(meta, [1, 2])
    data.attrs["obsnum"] = 1.5
    with pytest.raises(TypeError):
        _ = DummyInstruAccessor(data).meta


def test_missing_required_header_propagates_and_closes(observation, monkeypatch):
    path, meta = observation
    data = make_dataset(meta, [1, 2])
    del data.attrs["master"]
    closed = []
    data.set_close(lambda: closed.append(True))
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: data)
    monkeypatch.setattr(DummyInstruFileAccessor, "_source_opener", xr.open_dataset)
    with pytest.raises(KeyError, match="master"):
        DummyInstruFileAccessor.from_sources([path], mode="read")
    assert closed == [True]


@pytest.mark.parametrize("on_error", ["raise", "record"])
def test_recognized_dummy_header_error_does_not_try_another_reader(
    observation, on_error
):
    path, meta = observation
    data = make_dataset(meta, [1, 2])
    del data.attrs["master"]

    def unexpected_reader(data):
        pytest.fail("Malformed dummy metadata fell through to another reader")

    class DummyFiles(DummyInstruFileAccessor):
        _source_readers = (identify_dummy_instru, unexpected_reader)

    sources = [{"source": data, "source_loc": path}]
    if on_error == "raise":
        with pytest.raises(KeyError, match="master"):
            DummyFiles.from_sources(sources, mode="read", on_error=on_error)
    else:
        table = DummyFiles.from_sources(sources, mode="read", on_error=on_error)
        [diagnostic] = table.diagnostics.iloc[0]
        assert diagnostic.stage == "metadata"
        assert diagnostic.exception_type == "KeyError"
        assert "master" in diagnostic.message
        assert table.obsnum.iloc[0] == meta.obsnum


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_reject_invalid_subtree_registration(name):
    with pytest.raises(ValueError, match="Invalid substore name"):
        LmtFileStore.register_substore(name, DummyInstruFileStore)


def test_subtree_registration_preserves_existing_class():
    LmtFileStore.register_substore("dummy_instru", DummyInstruFileStore)

    class ConflictingStore(DummyInstruFileStore):
        pass

    with pytest.raises(ValueError, match="already registered"):
        LmtFileStore.register_substore("dummy_instru", ConflictingStore)
    assert type(LmtFileStore("/data").substores["dummy_instru"]) is DummyInstruFileStore
