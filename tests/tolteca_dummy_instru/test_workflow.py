"""Dummy instrument discovery, metadata and resource ownership."""

import subprocess
import sys
from itertools import permutations
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from astropy import units as u
from astropy.time import Time
from tollan.config import RuntimeContext
from tollan.utils.registry import Registry

from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.input_discovery import InputDiscovery
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt.input_discovery import LmtInputDiscovery
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_dummy_instru import register
from tolteca_dummy_instru.data import (
    DummyInstruAccessor,
    open_dummy_instru,
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
    from tolteca_datamodels.core import data_model as data_model_registry
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
        data_models=["dummy_instru"],
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
@pytest.mark.parametrize(
    "scope",
    [
        (LmtFileAccessor, {0, 3, 4}),
        (ToltecFileAccessor, {0, 1}),
        (DummyInstruFileAccessor, {2, 3}),
    ],
)
def test_universal_guesses_and_scoped_derived_columns(  # noqa: PLR0913, PLR0917 -- fixtures plus the scope/mode matrix.
    scope, mode, observation, dummy_tel_path, data_lmt_path, monkeypatch
):
    accessor_cls, identified = scope
    dummy_path, _ = observation
    sources = [
        data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        data_lmt_path
        / "toltec/tcs/toltec0/toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
        dummy_path,
        dummy_tel_path,
        "tel_unregistered_2025-10-31_145609_00_0002.nc",
        "unknown.nc",
    ]
    parsed, read_positions = [], []
    parse = FilepathInfo.from_source

    def trace_parse(source, *, raise_on_error=True):
        parsed.append(source)
        return parse(source, raise_on_error=raise_on_error)

    def read_metadata(self, rows, *, trace):
        read_positions.extend(position for position, _ in rows)

    def unexpected(*args, **kwargs):
        pytest.fail("Filename projection opened a file")

    monkeypatch.setattr(FilepathInfo, "from_source", trace_parse)
    monkeypatch.setattr(accessor_cls, "_read_metadata", read_metadata)
    monkeypatch.setattr(xr, "open_dataset", unexpected)
    table = accessor_cls.from_sources(sources, mode=mode)
    assert parsed == table.source_loc.tolist()
    assert table.source.tolist() == sources
    assert set(table.index[table.obsnum.notna()]) == {0, 1, 2, 3, 4}
    assert set(table.index[table.file_prefix.notna()]) == {0, 1, 2, 3, 4}
    assert table.data_models.iloc[4] == ["lmt"]
    assert table.source_trace.map(source_trace_issues).iloc[4] == []
    assert set(table.index[table.uid_raw_obs_file.notna()]) == identified
    expected_reads = {
        "guess": [],
        "guess_or_read": [5],
        "read": sorted(identified | {5}),
    }
    assert read_positions == expected_reads[mode]
    if accessor_cls is ToltecFileAccessor:
        assert table.data_kind_toltec.tolist() == [
            ToltecDataKind.lmt_tel.value,
            ToltecDataKind.targsweep.value,
            ToltecDataKind.unknown.value,
            ToltecDataKind.unknown.value,
            ToltecDataKind.unknown.value,
            ToltecDataKind.unknown.value,
        ]
    elif accessor_cls is DummyInstruFileAccessor:
        kind = DummyInstruDataKind
        assert table.data_kind_dummy_instru.tolist() == [
            kind.unknown.value,
            kind.unknown.value,
            kind.counts.value,
            kind.telescope.value,
            kind.unknown.value,
            kind.unknown.value,
        ]
    else:
        assert "data_kind_dummy_instru" not in table


@pytest.mark.parametrize("accessor_cls", [ToltecFileAccessor, DummyInstruFileAccessor])
def test_current_membership_controls_foreign_companion_reads(
    accessor_cls, data_lmt_path, dummy_tel_path
):
    toltec_tel_path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    path, location, receiver = (
        (toltec_tel_path, dummy_tel_path, "toltec")
        if accessor_cls is ToltecFileAccessor
        else (dummy_tel_path, toltec_tel_path, "dummy_instru")
    )
    # Both paths remain valid telescope filenames in the context-free parser.
    info = FilepathInfo.from_source(location)
    assert isinstance(info, LmtFilepathInfo)
    assert info.interface == "tel"
    with xr.open_dataset(path) as data:
        sources = [{"source": data, "source_loc": location}]
        guessed = accessor_cls.from_sources(sources)
        assert guessed.file_prefix.tolist() == [info.prefix]
        assert guessed.obsnum.tolist() == [info.obsnum]
        assert guessed.data_models.tolist() == [info.data_models]
        assert guessed.source_trace.map(source_trace_issues).iloc[0] == []
        # A recognized foreign membership is skipped until corrected explicitly.
        for result in (
            accessor_cls.from_sources(sources, mode="guess_or_read"),
            accessor_cls.from_sources(sources, mode="read"),
            accessor_cls(guessed).update_from_source_metadata(),
        ):
            pd.testing.assert_frame_equal(
                result.drop(columns="source_trace"),
                guessed.drop(columns="source_trace"),
            )
            assert any(
                event.outcome == "out_of_scope" for event in result.source_trace.iloc[0]
            )
        corrected = LmtFileAccessor(guessed).update_from_source_metadata()
        recovered = accessor_cls(corrected).update_from_source_metadata(force=True)
        assert recovered.data_models.tolist() == [["lmt", receiver]]
        assert recovered.obsnum.tolist() == [145609]
        assert recovered.uid_raw_obs_file.tolist() == ["tcs-145609-0-2-tel"]
        assert recovered.file_prefix.tolist() == [info.prefix]
        assert recovered.source.iloc[0] is data
        assert recovered.source_loc.iloc[0].path == location
        assert recovered.source_trace.map(source_trace_issues).iloc[0] == []
        assert not data["Data.TelescopeBackend.TelAzAct"].variable._in_memory


@pytest.mark.parametrize("accessor_cls", [LmtFileAccessor, ToltecFileAccessor])
def test_corrected_current_identity_can_read_data_despite_foreign_filename(
    accessor_cls, observation, data_lmt_path
):
    dummy_path, _ = observation
    tel_path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(tel_path) as data:
        close = data._close
        table = accessor_cls.from_sources(
            [{"source": data, "source_loc": dummy_path}], mode="read"
        )
        assert table.interface.tolist() == ["chan01"]
        table["data_models"] = [["lmt", "toltec"]]
        table["instru"] = "lmt"
        table["interface"] = "tel"
        table = accessor_cls(table).update_from_source_metadata()
        assert table.file_prefix.tolist() == ["chan01"]
        assert table.obsnum.tolist() == [145609]
        assert table.interface.tolist() == ["tel"]
        assert table.uid_raw_obs_file.tolist() == ["tcs-145609-0-2-tel"]
        assert table.source.iloc[0] is data
        assert table.source_loc.iloc[0].path == dummy_path
        assert data._close is close


def test_foreign_filename_retained_without_toltec_classification(observation):
    path, _ = observation
    table = ToltecFileAccessor.from_sources([path], mode="guess_or_read")
    assert table.source.tolist() == [path]
    assert table.file_prefix.tolist() == ["chan01"]
    assert table.obsnum.tolist() == [1]
    assert table.uid_raw_obs_file.isna().all()
    assert table.data_kind_toltec.tolist() == [ToltecDataKind.unknown.value]


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_classification_contexts_use_distinct_telescope_companions(
    observation, dummy_tel_path, data_lmt_path, mode, monkeypatch
):
    from tolteca_datamodels.lmt.tel import LmtTelIOMapper

    dummy_path, _ = observation
    tel_path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    kids_path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_targsweep.nc"))
    sources = [
        tel_path,
        kids_path,
        dummy_path,
        dummy_tel_path,
        "archive:/unidentified.nc",
    ]
    telescope_reads = []
    get_metadata = LmtTelIOMapper.get_metadata

    def tracked(self, data_source):
        telescope_reads.append(data_source)
        return get_metadata(self, data_source)

    monkeypatch.setattr(LmtTelIOMapper, "get_metadata", tracked)
    lmt = LmtFileAccessor.from_sources(sources, mode=mode)
    assert len(telescope_reads) == (2 if mode == "read" else 0)
    assert "data_kind_dummy_instru" not in lmt
    assert "counts" not in LmtFileAccessor._query_namespace
    assert "vnasweep" not in LmtFileAccessor._query_namespace

    toltec = ToltecFileAccessor.from_sources(sources, mode=mode)
    assert len(telescope_reads) == (3 if mode == "read" else 0)
    before = toltec.copy(deep=True)
    dummy = DummyInstruFileAccessor.from_sources(sources, mode=mode)
    assert len(telescope_reads) == (4 if mode == "read" else 0)
    pd.testing.assert_frame_equal(toltec, before)
    kind = DummyInstruDataKind
    assert dummy.data_kind_dummy_instru.tolist() == [
        kind.unknown.value,
        kind.unknown.value,
        kind.counts.value,
        kind.telescope.value,
        kind.unknown.value,
    ]
    assert toltec.data_kind_toltec.tolist() == [
        ToltecDataKind.lmt_tel.value,
        ToltecDataKind.targsweep.value,
        ToltecDataKind.unknown.value,
        ToltecDataKind.unknown.value,
        ToltecDataKind.unknown.value,
    ]
    assert kind.counts.value == ToltecDataKind.vnasweep.value
    assert kind.counts != ToltecDataKind.vnasweep
    assert dummy.dummy_instru_file.query(
        "bits_any_set(data_kind_dummy_instru, counts | telescope)"
    ).index.tolist() == [2, 3]
    assert toltec.toltec_file.query(
        "bits_any_set(data_kind_toltec, raw_sweep | lmt_tel)"
    ).index.tolist() == [0, 1]
    assert "counts" not in ToltecFileAccessor._query_namespace
    assert "raw_sweep" not in DummyInstruFileAccessor._query_namespace
    for table in (lmt, toltec, dummy):
        assert table.source.tolist() == sources


def test_dummy_classification_follows_corrected_current_routing(
    observation, dummy_tel_path
):
    path, _ = observation
    tel_path = dummy_tel_path
    kind = DummyInstruDataKind
    with xr.open_dataset(path) as counts, xr.open_dataset(tel_path) as telescope:
        sources = [
            {"source": counts, "source_loc": tel_path},
            {"source": telescope, "source_loc": path},
        ]
        table = DummyInstruFileAccessor.from_sources(sources)
        table.index = [8, 8]
        table["instru"] = ["dummy_instru", "lmt"]
        table["interface"] = ["chan01", "tel"]
        table["interface_kind"] = ["counter", "tel"]
        before = table.copy(deep=True)
        assert table.data_kind_dummy_instru.tolist() == [
            kind.telescope.value,
            kind.counts.value,
        ]
        refreshed = table.dummy_instru_file.update_from_source_metadata()
        assert refreshed.data_kind_dummy_instru.tolist() == [
            kind.counts.value,
            kind.telescope.value,
        ]
        assert refreshed.data_models.tolist() == [
            ["dummy_instru"],
            ["lmt", "dummy_instru"],
        ]
        assert refreshed.file_data_models.tolist() == [
            ["lmt", "dummy_instru"],
            ["dummy_instru"],
        ]
        assert refreshed.instru.tolist() == ["dummy_instru", "lmt"]
        assert refreshed.file_instru.tolist() == ["lmt", "dummy_instru"]
        pd.testing.assert_index_equal(refreshed.index, table.index)
        pd.testing.assert_frame_equal(table, before)
        assert not counts["counts"].variable._in_memory
        assert not telescope["Data.TelescopeBackend.TelAzAct"].variable._in_memory
        assert counts._close is not None and telescope._close is not None


def test_dummy_classification_failure_and_retry_preserve_row_information(observation):
    path, meta = observation
    data = make_dataset(meta, [1, 2])
    del data.attrs["master"]
    kind = DummyInstruDataKind
    # A misleading location first selects an incompatible reader.
    location = "tel_dummy_instru_2025-10-31_000001_00_0002.nc"
    table = DummyInstruFileAccessor.from_sources(
        [{"source": data, "source_loc": location}, f"archive:{path}"],
        mode="read",
        on_error="record",
    )
    assert table.data_kind_dummy_instru.tolist() == [
        kind.telescope.value,
        kind.counts.value,
    ]
    assert table.source_trace.map(source_trace_issues).iloc[0][-1].stage == "identify"
    assert table.file_prefix.iloc[0] == "tel_dummy_instru"
    table["source"] = pd.Series(
        [make_dataset(meta, [1, 2]), table.source.iloc[1]], dtype=object
    ).array
    table["instru"] = "dummy_instru"
    table["interface"] = "chan01"
    table.dummy_instru_file.update_from_source_metadata(inplace=True, on_error="record")
    assert table.data_kind_dummy_instru.tolist() == [
        kind.counts.value,
        kind.counts.value,
    ]
    issues = table.source_trace.map(source_trace_issues)
    assert issues.iloc[0] == []
    assert [event.stage for event in issues.iloc[1]] == ["open"]
    assert table.file_prefix.iloc[0] == "tel_dummy_instru"


@pytest.mark.parametrize(
    "order",
    tuple(permutations(("lmt", "toltec", "dummy"))),
)
def test_classification_contexts_do_not_depend_on_registration_order(order):
    code = """
import importlib
import sys

for name in sys.argv[1:]:
    if name == "dummy":
        from tolteca_dummy_instru import register
        register()
    else:
        importlib.import_module(f"tolteca_datamodels.{name}.file")

from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.types import DummyInstruDataKind

sources = [
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
    "chan01-tcs-1-0-2-20251031T120000.nc",
]
lmt = LmtFileAccessor.from_sources(sources)
toltec = ToltecFileAccessor.from_sources(sources)
dummy = DummyInstruFileAccessor.from_sources(sources)
assert "data_kind_dummy_instru" not in lmt
assert toltec.data_kind_toltec.tolist() == [
    ToltecDataKind.lmt_tel.value,
    ToltecDataKind.targsweep.value,
    ToltecDataKind.unknown.value,
]
assert dummy.data_kind_dummy_instru.tolist() == [
    DummyInstruDataKind.unknown.value,
    DummyInstruDataKind.unknown.value,
    DummyInstruDataKind.counts.value,
]
assert "counts" not in ToltecFileAccessor._query_namespace
assert "vnasweep" not in DummyInstruFileAccessor._query_namespace
generic = FileAccessor.from_sources(sources)
assert generic.data_kind_toltec.tolist() == toltec.data_kind_toltec.tolist()
assert generic.data_kind_dummy_instru.tolist() == dummy.data_kind_dummy_instru.tolist()
assert "data_kind" not in generic
assert FileAccessor(generic).query(
    "bits_any_set(data_kind_dummy_instru, DummyInstruDataKind.counts)"
).index.tolist() == [2]
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test program.
        [sys.executable, "-c", code, *order],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("accessor_cls", [LmtFileAccessor, ToltecFileAccessor])
def test_open_skips_other_instrument_as_during_construction(accessor_cls, observation):
    path, _ = observation
    table = accessor_cls.from_sources([path], mode="read")
    assert table.obsnum.tolist() == [1]
    assert table.source_trace.map(source_trace_issues).iloc[0] == []
    with accessor_cls(table).open(on_error="raise") as opened:
        assert opened.io_obj.iloc[0] is None
        assert opened.accessor_obj.iloc[0] is None


def test_dummy_open_does_not_accept_kids(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc"))
    table = DummyInstruFileAccessor.from_sources([path], mode="read")
    assert table.obsnum.notna().all()
    assert table.source_trace.map(source_trace_issues).iloc[0] == []
    with table.dummy_instru_file.open(on_error="raise") as opened:
        assert opened.io_obj.iloc[0] is None
        assert opened.accessor_obj.iloc[0] is None


def test_dummy_open_accepts_only_its_telescope_receiver(data_lmt_path, dummy_tel_path):
    toltec_tel = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    table = DummyInstruFileAccessor.from_sources(
        [toltec_tel, dummy_tel_path], mode="read"
    )
    assert table.data_kind_dummy_instru.tolist() == [
        DummyInstruDataKind.unknown.value,
        DummyInstruDataKind.telescope.value,
    ]
    assert table.data_models.iloc[0] == ["lmt", "toltec"]
    assert table.data_models.iloc[1] == ["lmt", "dummy_instru"]
    with table.dummy_instru_file.open() as opened:
        assert opened.io_obj.iloc[0] is None
        assert opened.io_obj.iloc[1].lmt_tel.meta.data_models == ["lmt", "dummy_instru"]


def test_explicit_registration_in_fresh_process():
    code = """
import sys
from tolteca_datamodels.core import data_model as data_model_module
from tolteca_datamodels.input_discovery import InputDiscoveryConfig
from tolteca_datamodels.lmt.file_store import LmtFileStore
import tolteca_dummy_instru
assert not any(cls.__module__.startswith('tolteca_dummy_instru.')
           for cls in LmtFileStore('/data').substores)
definitions = InputDiscoveryConfig.model_json_schema()['$defs']
assert 'DummyInstruInputDiscovery' not in definitions
assert 'tolteca_dummy_instru.data_model' not in sys.modules
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.types import DummyInstruDataModel
assert DummyInstruFileAccessor.data_model_cls is DummyInstruDataModel
assert 'tolteca_dummy_instru.data_model' not in sys.modules
assert 'dummy_instru' not in data_model_module._data_models
tolteca_dummy_instru.register()
from tolteca_dummy_instru.data_model import data_model
definition = data_model_module._data_models['dummy_instru']
assert definition is data_model
assert definition.file_accessor_cls is DummyInstruFileAccessor
before = dict(data_model_module._data_models)
tolteca_dummy_instru.register()
assert data_model_module._data_models == before
from tolteca_dummy_instru.file_store import DummyInstruFileStore
store = LmtFileStore('/data').substores[DummyInstruFileStore]
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
    assert table.data_models.tolist() == [["lmt", "dummy_instru"], ["dummy_instru"]]
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
    assert set(table.interface) == interfaces
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
    table = entry.discover()
    assert table.interface.tolist() == expected
    files = FileAccessor.partial(accessor_types=models.file_accessor_types)
    populated = files.from_sources(candidates)
    pd.testing.assert_frame_equal(table, files(populated).query_obs_spec(selector))


def test_kids_cli_sets_scope_before_discovery(
    observation, data_lmt_path, tmp_path, monkeypatch
):
    from typer.testing import CliRunner

    from tolteca_kids.cli import app

    (tmp_path / "tel").symlink_to(data_lmt_path / "tel")
    (tmp_path / "toltec").symlink_to(data_lmt_path / "toltec")

    def unexpected(*args, **kwargs):
        pytest.fail("KIDs discovery searched a foreign instrument store")

    monkeypatch.setattr(DummyInstruFileStore, "_iter_local_glob_patterns", unexpected)
    result = CliRunner().invoke(
        app, ["--data_lmt_path", str(tmp_path), "--show_inputs", "--", "145609/{tel,0}"]
    )
    assert result.exit_code == 0, result.exception
    import json

    rows = json.loads(result.stdout)
    assert len(rows) == 4
    assert {row["interface"] for row in rows} == {"tel", "toltec0"}


@pytest.mark.parametrize("accessor_cls", [FileAccessor, ToltecFileAccessor])
def test_explicit_locations_are_not_filtered_by_instrument_store_scope(
    observation, accessor_cls
):
    from tolteca_datamodels.lmt_base.input_discovery import SourcesInputDiscovery

    path, _ = observation
    entry = SourcesInputDiscovery(sources=[path], mode="guess")
    table = entry.discover(
        data_models=DataModelSet.from_select(
            selection=["toltec"] if accessor_cls is ToltecFileAccessor else ".*",
        )
    )
    assert len(table) == 1
    assert table.source_loc.iloc[0].path == path
    if accessor_cls is FileAccessor:
        assert table.data_kind_dummy_instru.iloc[0] == DummyInstruDataKind.counts.value
        assert table.interface.iloc[0] == "chan01"
    else:
        assert table.data_kind_toltec.iloc[0] == ToltecDataKind.unknown.value
        assert table.file_prefix.iloc[0] == "chan01"
        assert table.interface.iloc[0] == "chan01"


def test_lazy_samples_and_header_override(observation):
    path, meta = observation
    with open_dummy_instru(path) as data:
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
    assert store.glob_obs_spec(spec) == [path]
    assert store.glob_obs_spec("1/tel") == []
    table = DummyInstruFileAccessor.from_sources([path])
    assert len(table.dummy_instru_file.query_obs_spec(spec)) == 1


def test_root_delegates_and_selects_combined_table(observation, tmp_path, monkeypatch):
    path, _ = observation
    tel = tmp_path / "tel/tel_dummy_instru_2025-10-31_000001_00_0002.nc"
    tel.parent.mkdir()
    tel.touch()  # Filename-only discovery; no invented telescope header.
    store = LmtFileStore(tmp_path)
    candidates = store.partial("tel", "dummy_instru").glob_obs_spec("1/{tel,1}")
    assert candidates == [tel, path]
    table = DummyInstruFileAccessor.from_sources(candidates)
    selected = table.dummy_instru_file.query_obs_spec("1/{tel,1}")
    assert selected.data_models.tolist() == [["lmt", "dummy_instru"], ["dummy_instru"]]
    assert selected.instru.tolist() == ["lmt", "dummy_instru"]
    assert selected.interface.tolist() == ["tel", "chan01"]
    assert selected.roach.isna().all()
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
    pd.testing.assert_frame_equal(
        guessed.drop(columns="source_trace"), read.drop(columns="source_trace")
    )
    assert read.source_trace.iloc[0][-1].stage == "open"
    assert read.source_trace.iloc[0][-1].outcome == "error"
    assert DummyInstruFileAccessor.from_sources([], mode="read").empty


@pytest.mark.parametrize("accessor_cls", [FileAccessor, DummyInstruFileAccessor])
def test_live_reader_metadata_populates_table(observation, monkeypatch, accessor_cls):
    path, meta = observation
    data = make_dataset(meta, [1, 2])
    table = accessor_cls.from_sources(
        [{"source": data, "source_loc": path}], mode="guess"
    )
    table["obsnum"] = 0
    with accessor_cls(table).open() as opened:
        assert opened.io_obj.iloc[0] is data
        reader = opened.accessor_obj.iloc[0]
        metadata = reader.meta

        def unexpected(*args, **kwargs):
            pytest.fail("Table enrichment bypassed the live reader's metadata")

        monkeypatch.setattr(type(reader.mapper), "get_metadata", unexpected)
        accessor_cls(opened).update_from_source_metadata(inplace=True, force=True)
        assert opened.obsnum.tolist() == [metadata.obsnum]
        assert opened.uid_raw_obs_file.tolist() == [metadata.uid_raw_obs_file]
    assert reader.meta is metadata


def test_dummy_telescope_metadata_uses_shared_lmt_reader(observation, dummy_tel_path):
    path, _ = observation
    with xr.open_dataset(dummy_tel_path) as companion:
        table = DummyInstruFileAccessor.from_sources(
            [path, {"source": companion, "source_loc": dummy_tel_path}],
            mode="read",
        )
        assert table.data_models.tolist() == [["dummy_instru"], ["lmt", "dummy_instru"]]
        assert table.instru.tolist() == ["dummy_instru", "lmt"]
        assert table.interface.tolist() == ["chan01", "tel"]
        assert table.channel.isna().tolist() == [False, True]
        assert table.obsnum.tolist() == [1, 145609]
        assert table.target_ra_deg.iloc[1] == companion.lmt_tel.meta.target.ra.deg
        assert companion.lmt_tel.meta.filepath_info_orig.prefix == "tel_dummy_instru"


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
    with pytest.raises(KeyError, match="master"):
        DummyInstruFileAccessor.from_sources([path], mode="read", on_error="raise")
    assert closed == [True]


def test_corrupt_dummy_metadata_records_failure_and_continues(observation):
    path, meta = observation
    data = make_dataset(meta, [1, 2])
    del data.attrs["master"]

    sources = [
        {"source": data, "source_loc": path},
        {"source": make_dataset(meta, [3, 4]), "source_loc": "renamed.nc"},
    ]
    table = DummyInstruFileAccessor.from_sources(sources, mode="read")
    [diagnostic] = table.source_trace.map(source_trace_issues).iloc[0]
    assert diagnostic.stage == "metadata"
    assert diagnostic.exception_type == "KeyError"
    assert "master" in diagnostic.message
    assert table.obsnum.tolist() == [meta.obsnum, meta.obsnum]
    assert table.filepath_orig.isna().tolist() == [True, False]
    assert [
        item.stage for item in table.source_trace.map(source_trace_issues).iloc[1]
    ] == ["filepath"]


@pytest.mark.parametrize("subpath", ["", ".", "..", "a/../b", "/outside"])
def test_reject_invalid_subtree_registration(subpath):
    with pytest.raises(ValueError, match="relative descendant path"):
        LmtFileStore.register_substore(DummyInstruFileStore, subpath)


def test_subtree_registration_preserves_existing_path():
    with pytest.raises(ValueError, match="already registered"):
        LmtFileStore.register_substore(DummyInstruFileStore, "elsewhere")
    assert LmtFileStore("/data").substores[DummyInstruFileStore].path == Path(
        "/data/dummy_instru"
    )
