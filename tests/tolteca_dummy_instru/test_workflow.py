"""A separate instrument exercises mixed, scoped and configured file workflows."""

from pathlib import Path

import pandas as pd
import pytest
from astropy import units as u
from astropy.time import Time
from tollan.config import RuntimeContext
from tollan.utils.registry import Registry

from tolteca_datamodels.core.file import FileAccessorView
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt.input_discovery import LmtInputDiscovery
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_dummy_instru import register
from tolteca_dummy_instru.data import (
    DummyInstruAccessor,
)
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.file_store import DummyInstruFileStore
from tolteca_dummy_instru.metadata import DummyInstruMetadata
from tolteca_dummy_instru.testing import make_dataset
from tolteca_dummy_instru.types import DummyInstruDataKind


@pytest.fixture(autouse=True)
def enable_dummy(monkeypatch):
    # Keep the opt-in isolated from other test modules in the same process.
    import tolteca_dummy_instru
    from tolteca_datamodels import data_model as data_model_registry
    from tolteca_datamodels.core.input_discovery import config as input_discovery

    input_models = input_discovery._input_discovery_models
    substores = Registry()
    for store_cls, subpath in LmtFileStore._substore_paths.items():
        substores.register(store_cls, subpath)

    monkeypatch.setattr(tolteca_dummy_instru, "_registered", False)
    monkeypatch.setattr(
        data_model_registry._data_models,
        "_entries",
        dict(data_model_registry.data_models),
    )
    monkeypatch.setattr(
        LmtFileStore,
        "_substore_paths",
        substores,
    )
    register()
    yield
    input_discovery._input_discovery_models = input_models
    input_discovery.InputDiscoveryConfig.model_rebuild(force=True)


@pytest.fixture
def observation(tmp_path):
    path = tmp_path / "dummy_instru/chan01/chan01-tcs-1-0-2-20251031T120000.nc"
    meta = DummyInstruMetadata(
        data_models=frozenset({"dummy_instru"}),
        instru="dummy_instru",
        interface="chan01",
        interface_kind="counter",
        channel=1,
        master="tcs",
        obsnum=1,
        subobsnum=0,
        scannum=2,
        filepath_orig=str(path),
        t0=Time(1761912000.0, format="unix"),
        t1=Time(1761912003.0, format="unix"),
        t_int=3 * u.s,
    )
    path.parent.mkdir(parents=True)
    make_dataset(meta, [1, 2, 4, 8]).to_netcdf(path)
    return path, meta


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
@pytest.mark.parametrize("accessor_cls", [FileAccessor, DummyInstruFileAccessor])
def test_registered_discovery_matches_typed_python(
    observation, tmp_path, mode, accessor_cls
):
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
    ).with_config(validation_context={"root_path": tmp_path})
    handler = InputDiscovery(runtime)
    assert isinstance(handler.config.inputs[0], DummyInstruInputDiscovery)
    models = DataModelSet.from_select(
        ["dummy_instru"] if accessor_cls is DummyInstruFileAccessor else ".*"
    )
    direct = DummyInstruInputDiscovery(
        data_dummy_instru_path=tmp_path / "dummy_instru",
        obs_spec=ObsSpec.parse("1/(1,3)"),
        mode=mode,
    ).discover(data_models=models)
    configured = handler.discover(data_models=models)
    pd.testing.assert_frame_equal(configured, direct)
    assert configured.source_loc.iloc[0].path == path
    expected = DummyInstruDataKind.counts.value
    if accessor_cls is DummyInstruFileAccessor:
        assert "data_kind_toltec" not in configured
    assert configured.data_kind_dummy_instru.tolist() == [expected]


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_discovery_combines_entries_without_aliasing_flag_namespaces(
    observation, tmp_path, data_lmt_path, mode
):
    path, _ = observation
    toltec = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_vnasweep.nc"))
    discovery = InputDiscovery(
        RuntimeContext(
            {
                "input_discovery": {
                    "inputs": [
                        {"sources": [str(toltec)], "mode": mode},
                        {
                            "data_dummy_instru_path": str(tmp_path / "dummy_instru"),
                            "mode": mode,
                        },
                        {"sources": [str(path)], "mode": mode},
                    ]
                }
            }
        )
    )
    table = discovery.discover()
    assert table.interface.tolist() == ["toltec0", "chan01"]
    assert table.data_kind_toltec.tolist() == [
        ToltecDataKind.vnasweep.value,
        ToltecDataKind.unknown.value,
    ]
    assert table.data_kind_dummy_instru.tolist() == [
        DummyInstruDataKind.unknown.value,
        DummyInstruDataKind.counts.value,
    ]
    assert FileAccessor(table).query(
        "bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.counts)"
    ).interface.tolist() == ["chan01"]
    assert FileAccessor(table).query(
        "bits_any_set(data_kind_toltec, ToltecDataKind.raw_sweep)"
    ).interface.tolist() == ["toltec0"]
    assert "io_obj" not in table


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_root_discovery_populates_dummy_and_companion_tel(
    observation, dummy_tel_path, tmp_path, mode
):
    tel_dir = tmp_path / "tel"
    tel_dir.mkdir()
    (tel_dir / dummy_tel_path.name).symlink_to(dummy_tel_path)
    table = InputDiscovery(
        RuntimeContext(
            {
                "input_discovery": {
                    "inputs": [
                        {
                            "data_lmt_path": str(tmp_path),
                            "obs_spec": "*/{tel,1}",
                            "mode": mode,
                        },
                    ]
                }
            }
        )
    ).discover()
    assert table.interface.tolist() == ["tel", "chan01"]
    assert table.data_models.tolist() == [{"lmt", "dummy_instru"}, {"dummy_instru"}]
    assert table.data_kind_dummy_instru.tolist() == [
        DummyInstruDataKind.telescope.value,
        DummyInstruDataKind.counts.value,
    ]
    assert "io_obj" not in table


@pytest.mark.parametrize(
    ("selection", "interfaces"),
    [
        (".*", {"tel", "toltec0"}),
        (["toltec"], {"tel", "toltec0"}),
        (["dummy_instru"], set()),
    ],
)
@pytest.mark.usefixtures("observation")
def test_task_set_intersects_data_model_requirements(
    data_lmt_path, tmp_path, monkeypatch, selection, interfaces
):
    (tmp_path / "tel").symlink_to(data_lmt_path / "tel")
    (tmp_path / "toltec").symlink_to(data_lmt_path / "toltec")

    def unexpected(*args, **kwargs):
        pytest.fail("An excluded instrument store was searched")

    monkeypatch.setattr(DummyInstruFileStore, "_iter_local_glob_patterns", unexpected)
    if selection == ["dummy_instru"]:
        monkeypatch.setattr(ToltecFileStore, "_iter_local_glob_patterns", unexpected)
    models = DataModelSet.from_select(selection=["toltec"])
    discovery = InputDiscovery(
        RuntimeContext(
            {
                "input_discovery": {
                    "inputs": [
                        {
                            "data_lmt_path": str(tmp_path),
                            "data_model_select": selection,
                            "obs_spec": "145609/{tel,0}",
                        },
                        {"data_dummy_instru_path": str(tmp_path / "dummy_instru")},
                    ]
                }
            }
        )
    )
    before = discovery.config.model_dump()
    table = discovery.discover(data_models=models)
    assert (set() if table.empty else set(table.interface)) == interfaces
    if interfaces:
        assert table.data_kind_toltec.dtype == "Int64"
        assert all(isinstance(value, int) for value in table.data_kind_toltec.tolist())
    assert discovery.config.model_dump() == before


@pytest.mark.parametrize(
    ("selection", "selector", "expected"),
    [
        (".*", "tcs-[-1]", ["toltec0"]),
        ("toltec", "tcs-[-1]", ["toltec0"]),
        ("dummy_instru", "tcs-[-1]", ["chan01"]),
        (".*", "*/()[-1]", ["chan01"]),
        ("toltec", "*/()[-1]", ["toltec0"]),
        ("dummy_instru", "*/()[-1]", ["chan01"]),
    ],
)
def test_positions_apply_once_to_combined_store_candidates(  # noqa: PLR0913, PLR0917 -- reference inputs and selector matrix.
    observation, data_lmt_path, tmp_path, selection, selector, expected
):
    relative = "toltec/tcs/toltec0/toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    roach = tmp_path / relative
    roach.parent.mkdir(parents=True)
    roach.symlink_to(data_lmt_path / relative)
    store = LmtFileStore(tmp_path)
    models = DataModelSet.from_select(selection)
    search = store.partial(store_types=models.file_store_types)
    candidates = search.glob_obs_spec(selector)
    assert set(candidates) == {
        path
        for path, name in ((roach, "toltec"), (observation[0], "dummy_instru"))
        if selection in (".*", name)
    }

    entry = LmtInputDiscovery(
        data_lmt_path=tmp_path, obs_spec=selector, data_model_select=selection
    )
    table = entry.discover(data_models=models)
    assert table.interface.tolist() == expected
    populated = FileAccessorView.from_sources(
        candidates, accessor_types=models.file_accessor_types
    ).table
    pd.testing.assert_frame_equal(
        table,
        FileAccessor(populated)
        .partial(accessor_types=models.file_accessor_types)
        .query_obs_spec(selector),
    )


@pytest.mark.parametrize("spec", ["1/1", "1/chan01", "1/chan(1,3)", "1/(1,3)"])
def test_store_numeric_resolution_and_no_telescope_probe(observation, tmp_path, spec):
    path, _ = observation
    store = DummyInstruFileStore(tmp_path / "dummy_instru")
    assert store.glob_obs_spec(spec) == [path]
    assert store.glob_obs_spec("1/tel") == []
    table = DummyInstruFileAccessor.from_sources([path]).table
    assert len(table.dummy_instru_file.query_obs_spec(spec)) == 1


def test_root_delegates_and_selects_combined_table(observation, tmp_path, monkeypatch):
    path, _ = observation
    tel = tmp_path / "tel/tel_dummy_instru_2025-10-31_000001_00_0002.nc"
    tel.parent.mkdir()
    tel.touch()  # Filename-only discovery; no invented telescope header.
    store = LmtFileStore(tmp_path)
    candidates = store.partial("tel", "dummy_instru").glob_obs_spec("1/{tel,1}")
    assert candidates == [tel, path]
    table = DummyInstruFileAccessor.from_sources(candidates).table
    selected = table.dummy_instru_file.query_obs_spec("1/{tel,1}")
    assert selected.data_models.tolist() == [{"lmt", "dummy_instru"}, {"dummy_instru"}]
    assert selected.instru.tolist() == ["lmt", "dummy_instru"]
    assert selected.interface.tolist() == ["tel", "chan01"]
    kind = DummyInstruDataKind
    assert selected.data_kind_dummy_instru.tolist() == [
        kind.telescope.value,
        kind.counts.value,
    ]
    original = Path.glob
    probed = []

    def glob(directory, pattern):
        probed.append(directory)
        return original(directory, pattern)

    monkeypatch.setattr(Path, "glob", glob)
    assert store.partial("tel", "toltec").glob_obs_spec("1/{tel,1}") == [tel]
    assert not any("dummy_instru" in directory.parts for directory in probed)


def test_invalid_integer_header_is_not_truncated(observation):
    _, meta = observation
    data = make_dataset(meta, [1, 2])
    data.attrs["obsnum"] = 1.5
    with pytest.raises(TypeError):
        _ = DummyInstruAccessor(data).meta


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
@pytest.mark.parametrize(
    "accessor_type", [ToltecFileAccessor, DummyInstruFileAccessor, FileAccessor]
)
def test_modes_partition_known_foreign_and_renamed_sources(
    observation, data_lmt_path, tmp_path, mode, accessor_type
):
    dummy, meta = observation
    toltec = (
        data_lmt_path
        / "toltec/tcs/toltec0/toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
    )
    renamed = tmp_path / "renamed.nc"
    renamed.symlink_to(dummy)
    file_accessor = accessor_type.from_sources([toltec, dummy, renamed], mode=mode)
    can_read_dummy = accessor_type is not ToltecFileAccessor
    can_read_toltec = accessor_type is not DummyInstruFileAccessor
    original_paths = file_accessor.table.get(
        "filepath_orig", pd.Series(None, index=file_accessor.table.index, dtype=object)
    )
    assert original_paths.notna().tolist() == [
        mode == "read" and can_read_toltec,
        mode == "read" and can_read_dummy,
        mode != "guess" and can_read_dummy,
    ]
    assert file_accessor.table.file_data_models.tolist() == [
        {"toltec"},
        {"dummy_instru"},
        frozenset(),
    ]
    assert len(file_accessor.table) == 3
    if can_read_dummy and mode != "guess":
        assert file_accessor.table.obsnum.iloc[2] == meta.obsnum
        assert (
            file_accessor.table.data_kind_dummy_instru.iloc[2]
            == DummyInstruDataKind.counts.value
        )
    assert {"io_obj", "accessor_obj"}.isdisjoint(file_accessor.table)
    # Known foreign rows never attempt opening or reading.
    foreign = 0 if not can_read_toltec else (1 if not can_read_dummy else None)
    if foreign is not None:
        assert not file_accessor.source_trace.iloc[foreign].has_error


def test_partial_and_concrete_reads_agree_and_reuse_live_readers(
    observation, dummy_tel_path
):
    paths = [observation[0], dummy_tel_path]
    direct = DummyInstruFileAccessor.from_sources(paths, mode="read")
    composite = FileAccessor.from_sources(paths)
    selected = composite.partial(accessor_types=(DummyInstruFileAccessor,))
    with selected.open() as table:
        readers = list(table.accessor_obj)
        selected.read_metadata()
        assert list(table.accessor_obj) == readers
    columns = [name for name in direct.table if name != "source_trace"]
    pd.testing.assert_frame_equal(composite.table[columns], direct.table[columns])
    assert {"io_obj", "accessor_obj"}.isdisjoint(composite.table)


def test_current_membership_can_replace_a_wrong_filename_guess(
    observation, data_lmt_path
):
    # A cached TolTEC resource is deliberately accompanied by a wrong location.
    toltec = (
        data_lmt_path
        / "toltec/tcs/toltec0/toltec0_145609_000_0002_2025_10_31_09_27_57.nc"
    )
    file_accessor = ToltecFileAccessor.from_sources(
        [{"source": toltec, "source_loc": observation[0]}], mode="read"
    )
    assert "filepath_orig" not in file_accessor.table
    file_accessor.table["data_models"] = [["toltec"]]
    file_accessor.table["interface_kind"] = "roach"
    file_accessor.read_metadata(on_error="raise")
    assert file_accessor.table.data_models.tolist() == [{"toltec"}]
    assert file_accessor.table.file_data_models.tolist() == [{"dummy_instru"}]
    assert file_accessor.table.obsnum.tolist() == [145609]


def test_corrupt_metadata_preserves_guesses_and_records_the_failure(observation):
    path, meta = observation
    dataset = make_dataset(meta, [1, 2])
    del dataset.attrs["obsnum"]
    file_accessor = DummyInstruFileAccessor.from_sources(
        [{"source": dataset, "source_loc": path}]
    )
    before = file_accessor.table.obsnum.copy()
    file_accessor.read_metadata()
    pd.testing.assert_series_equal(file_accessor.table.obsnum, before)
    assert [
        entry.kind.value
        for entry in file_accessor.source_trace.iloc[0].iter_error_entries()
    ] == ["metadata_read_failed"]
