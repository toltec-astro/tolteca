"""Configuration of input discovery."""

import glob
import pickle
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from pydantic import TypeAdapter, ValidationError
from tollan.config import RuntimeContext
from tollan.config.types import AnyPath
from tollan.utils.fileloc import FileLoc, fileloc
from tollan.utils.log import logger

from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.file_store import FileStoreBase
from tolteca_datamodels.input_discovery import (
    GlobSource,
    InputDiscovery,
    InputDiscoveryConfig,
    LmtInputDiscovery,
    SourcesInputDiscovery,
    ToltecInputDiscovery,
)
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.obs_spec import ObsSpecContext
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.file_store import ToltecFileStore


def discover(entries):
    return InputDiscovery(
        RuntimeContext({"input_discovery": {"inputs": entries}})
    ).discover()


@pytest.mark.parametrize("selector", ["145609/2/tel", "tcs-145609-0-1/0"])
@pytest.mark.parametrize("subtree", [False, True])
def test_stores_only_discover_candidates(data_lmt_path, monkeypatch, selector, subtree):
    def unexpected(*args, **kwargs):
        pytest.fail("Store discovery parsed filenames, built a table, or opened data")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    monkeypatch.setattr(ToltecFileAccessor, "from_sources", unexpected)
    monkeypatch.setattr(xr, "open_dataset", unexpected)
    store = (
        ToltecFileStore(data_lmt_path / "toltec")
        if subtree
        else LmtFileStore(data_lmt_path)
    )
    paths = store.find_candidates(selector)
    assert len(paths) == (0 if subtree and selector.endswith("/tel") else 2)


@pytest.mark.parametrize("subtree", [False, True])
@pytest.mark.parametrize("selector", ["*", "tcs-[-1]"])
def test_store_rows_keep_glob_order_after_selection(
    data_lmt_path, monkeypatch, subtree, selector
):
    telescope = (data_lmt_path / "tel").resolve()
    instrument = (data_lmt_path / "toltec").resolve()
    candidates = {
        telescope: sorted(telescope.glob("*.nc"), reverse=True),
        instrument: sorted(instrument.glob("*/*/*.nc"), reverse=True),
    }
    assert len(candidates[telescope]) == 7
    assert len(candidates[instrument]) == 4
    expected = ([] if subtree else candidates[telescope]) + candidates[instrument]
    if selector == "tcs-[-1]":
        latest = "145609" if subtree else "145611"
        expected = [path for path in expected if latest in path.name]

    def glob(path, pattern):
        return iter(candidates[path])

    monkeypatch.setattr(Path, "glob", glob)
    entry = (
        {"data_toltec_path": instrument}
        if subtree
        else {"data_lmt_path": data_lmt_path}
    )
    table = discover([entry | {"obs_spec": selector}])
    assert [loc.path for loc in table.source_loc] == [
        path.resolve() for path in expected
    ]


def test_defaults_and_implicit_entry_types(data_lmt_path):
    handler = InputDiscovery(
        RuntimeContext(
            {
                "input_discovery": {
                    "inputs": [
                        {"data_lmt_path": data_lmt_path},
                        {"data_toltec_path": data_lmt_path / "toltec"},
                        {"sources": ["https://example.org/data.nc"]},
                    ]
                }
            }
        )
    )
    tree, subtree, sources = handler.config.inputs
    assert isinstance(tree, LmtInputDiscovery) and tree.mode == "guess"
    assert tree.obs_spec == ObsSpec()
    assert tree.model_dump()["obs_spec"] == "*"
    assert tree.instru == ".*"
    assert isinstance(subtree, ToltecInputDiscovery) and subtree.mode == "guess"
    assert subtree.obs_spec == ObsSpec()
    assert subtree.model_dump()["obs_spec"] == "*"
    assert (
        isinstance(sources, SourcesInputDiscovery) and sources.mode == "guess_or_read"
    )
    assert InputDiscovery(RuntimeContext({})).discover().empty


def test_lmt_discovery_reuses_private_store_not_candidates_or_context(
    tmp_path, monkeypatch
):
    entry = LmtInputDiscovery.model_validate(
        {"data_lmt_path": "data", "instru": "toltec", "obs_spec": "1"},
        context={"rootpath": tmp_path},
    )
    constructed, searched, contextualized = [], [], []
    init = LmtFileStore.__init__

    def track_init(self, path):
        constructed.append(self)
        init(self, path)

    def find_candidates(self, spec, *, instru):
        searched.append(self)
        assert spec is entry.obs_spec
        assert instru == entry.instru
        return []

    def get_context(self, *, instru):
        contextualized.append(self)
        assert instru == entry.instru
        return ObsSpecContext()

    monkeypatch.setattr(LmtFileStore, "__init__", track_init)
    monkeypatch.setattr(LmtFileStore, "find_candidates", find_candidates)
    monkeypatch.setattr(LmtFileStore, "get_obs_spec_context", get_context)
    dumped = entry.model_dump()
    for _ in range(2):
        assert entry._find_candidates() == []
        assert entry._obs_spec_context == ObsSpecContext()
    assert constructed == [entry._store]
    assert searched == contextualized == [entry._store, entry._store]
    assert entry._store.path == tmp_path / "data"
    assert entry.model_dump() == dumped
    assert "_store" not in entry.model_json_schema()["properties"]


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
@pytest.mark.parametrize(
    "store_case",
    [
        (LmtInputDiscovery, "data_lmt_path", LmtFileStore, {"instru": "toltec"}),
        (ToltecInputDiscovery, "data_toltec_path", ToltecFileStore, {}),
    ],
)
def test_store_discovery_populates_before_selection(
    tmp_path, monkeypatch, mode, store_case
):
    discovery_cls, path_key, store_cls, options = store_case
    entry = discovery_cls.model_validate(
        {path_key: "data", "obs_spec": "tcs-145609/tel", "mode": mode} | options,
        context={"rootpath": tmp_path},
    )
    events = []
    paths = [tmp_path / "first.nc", tmp_path / "second.nc"]
    populated = pd.DataFrame(
        {
            "master": ["tcs", "tcs"],
            "obsnum": [145608, 145609],
            "interface": ["tel", "tel"],
        }
    )

    def find_candidates(store, spec, **kwargs):
        assert store.path == tmp_path / "data"
        assert spec is entry.obs_spec
        assert kwargs == options
        events.append("find")
        return paths

    def from_sources(cls, sources, *, mode, on_error):
        assert sources is paths
        assert mode == entry.mode
        assert on_error == "raise"
        events.append("populate")
        return populated

    monkeypatch.setattr(store_cls, "find_candidates", find_candidates)
    monkeypatch.setattr(ToltecFileAccessor, "from_sources", classmethod(from_sources))
    selected = entry.discover()
    assert events == ["find", "populate"]
    pd.testing.assert_frame_equal(selected, populated.iloc[[1]])


@pytest.mark.parametrize(
    ("instru", "expected"),
    [(".*", ["toltec3"]), ("foo", ["bar2"]), ("toltec", ["toltec3"])],
)
def test_lmt_selection_uses_participating_store_contexts(
    tmp_path, monkeypatch, instru, expected
):
    # Synthetic declarations exercise delegation, not real acquisition filenames.
    class Store(FileStoreBase[ObsSpec]):
        _obs_spec_context = ObsSpecContext(labels={"interface": {"bar1": 1, "bar2": 2}})

        def _iter_candidate_patterns(self, spec):
            return []

    original = LmtFileStore.substores
    monkeypatch.setattr(
        LmtFileStore,
        "substores",
        property(
            lambda self: dict(original.fget(self)) | {"foo": Store(self.path / "foo")}
        ),
    )
    monkeypatch.setattr(
        ToltecFileAccessor,
        "from_sources",
        classmethod(
            lambda cls, paths, *, mode, on_error: pd.DataFrame(
                {
                    "obsnum": [1, 1, 1, 1],
                    "interface": ["toltec1", "toltec3", "bar1", "bar2"],
                }
            )
        ),
    )
    entry = LmtInputDiscovery.model_validate(
        {"data_lmt_path": tmp_path, "instru": instru, "obs_spec": "1/()[-1]"}
    )
    # Apply positions once to the combined entry, not separately per instrument.
    assert entry.discover().interface.tolist() == expected


@pytest.mark.parametrize("selector", ["145609/{tel,0}", "145609/0", "tcs-[-1]/0"])
def test_file_table_accessor_supplies_toltec_resolution(data_lmt_path, selector):
    paths = LmtFileStore(data_lmt_path).find_candidates("*")
    table = ToltecFileAccessor.from_sources(paths, mode="guess")
    spec = ObsSpec.parse(selector)
    expected = spec.select(
        table, context=LmtFileStore(data_lmt_path).get_obs_spec_context()
    )
    pd.testing.assert_frame_equal(table.toltec_file.query_obs_spec(spec), expected)
    pd.testing.assert_frame_equal(table.toltec_file.query_obs_spec(selector), expected)


@pytest.mark.parametrize(
    "entry",
    [
        {"sources": []},
        {"sources": ["a.nc"], "data_lmt_path": "."},
        {"sources": ["a.nc"], "data_toltec_path": "."},
        {"data_lmt_path": ".", "data_toltec_path": "toltec"},
        {"data_toltec_path": ".", "mode": "always"},
        {"data_lmt_path": ".", "instru": "["},
        {"data_lmt_path": ".", "instru": None},
        {"data_lmt_path": ".", "instru": ["toltec"]},
        {"data_toltec_path": ".", "instru": ".*"},
        {"sources": ["a.nc"], "instru": ".*"},
        {"data_lmt_path": ".", "match_instru": "toltec"},
        {"sources": [{"unknown": "a.nc"}]},
        {"sources": [{"glob": "*.nc", "unknown": True}]},
        {"sources": [{"glob": "https://example.org/*.nc"}]},
        {"sources": ["a.nc"], "mode": "always"},
        {"sources": ["a.nc"], "select": "obsnum == 1"},
        {"sources": ["a.nc"], "obs_spec": "not-a-selector"},
    ],
)
def test_invalid_discovery_fails_before_opening(entry, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid configuration performed I/O")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    with pytest.raises(ValidationError):
        discover([entry])


@pytest.mark.parametrize("pattern", [".*", "toltec|rsr", "(?!rsr)"])
def test_instrument_regex_config_serialization(tmp_path, pattern):
    config = InputDiscoveryConfig.model_validate(
        {"inputs": [{"data_lmt_path": "data_lmt", "instru": pattern}]},
        context={"rootpath": tmp_path},
    )
    [entry] = config.inputs
    assert isinstance(entry, LmtInputDiscovery)
    assert entry.data_lmt_path == tmp_path / "data_lmt"
    assert entry.instru == pattern
    assert config.model_dump(mode="json")["inputs"][0]["instru"] == pattern
    assert InputDiscoveryConfig.model_validate_json(config.model_dump_json()) == config
    assert InputDiscoveryConfig.model_validate_yaml(config.model_dump_yaml()) == config
    assert pickle.loads(pickle.dumps(config)) == config  # noqa: S301 -- own model dump


@pytest.mark.parametrize(
    ("pattern", "interfaces"),
    [
        ("toltec", {"toltec0", "tel"}),
        ("(?!toltec)", {"tel"}),
        ("toltec0", {"tel"}),
    ],
)
@pytest.mark.parametrize("selector", ["145609", "145609/*", "145609/{tel,0}"])
def test_instrument_filter_keeps_telescope_selection(
    data_lmt_path, monkeypatch, pattern, interfaces, selector
):
    def unexpected(*args, **kwargs):
        pytest.fail("Guess-mode discovery opened a file")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    table = discover(
        [
            {
                "data_lmt_path": data_lmt_path,
                "instru": pattern,
                "obs_spec": selector,
            }
        ]
    )
    assert set(table.interface) == interfaces
    assert len(table) == 2 * len(interfaces)


def test_instrument_filter_logs_skipped_store(data_lmt_path):
    messages = []
    sink = logger.add(messages.append, level="DEBUG")
    try:
        table = discover([{"data_lmt_path": data_lmt_path, "instru": "(?!toltec)"}])
    finally:
        logger.remove(sink)
    assert not table.empty
    assert set(table.interface) == {"tel"}
    assert any(
        "Skipping instrument toltec" in m and "filter '(?!toltec)'" in m
        for m in messages
    )


@pytest.mark.parametrize(
    "selector", ["*", "tcs-(145605,145610)[-1]/0", "145609/{tel,0}", "145609/{tel,1}"]
)
def test_toltec_subtree_matches_lmt_toltec_candidates(
    data_lmt_path, monkeypatch, selector
):
    def unexpected(*args, **kwargs):
        pytest.fail("Filename-only discovery opened a file")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    whole = discover([{"data_lmt_path": data_lmt_path, "obs_spec": selector}])
    subtree = discover(
        [{"data_toltec_path": data_lmt_path / "toltec", "obs_spec": selector}]
    )
    expected = whole.loc[whole["interface"] != "tel"]
    assert subtree.equals(expected.reset_index(drop=True))
    root = (data_lmt_path / "toltec").resolve()
    assert all(loc.path.is_relative_to(root) for loc in subtree.source_loc)


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_toltec_subtree_population_modes(data_lmt_path, mode):
    table = discover(
        [
            {
                "data_toltec_path": data_lmt_path / "toltec",
                "obs_spec": "145606/0",
                "mode": mode,
            }
        ]
    )
    assert len(table) == 1
    assert table.iloc[0].file_suffix == "targsweep"
    assert ("master_code" in table) == (mode == "read")
    if mode == "read":
        assert table.iloc[0].master_code == 0
        assert table.iloc[0].atten_drive == 7.25
    assert "io_obj" not in table


def test_toltec_subtree_context_and_serialization(tmp_path):
    config = InputDiscoveryConfig.model_validate(
        {"inputs": [{"data_toltec_path": "toltec", "obs_spec": "145609/{tel,0}"}]},
        context={"rootpath": tmp_path},
    )
    [entry] = config.inputs
    assert isinstance(entry, ToltecInputDiscovery)
    assert entry.data_toltec_path == tmp_path / "toltec"
    assert entry.discover().empty
    assert InputDiscoveryConfig.model_validate_json(config.model_dump_json()) == config
    assert pickle.loads(pickle.dumps(config)) == config  # noqa: S301 -- own model dump


def test_overlapping_lmt_and_toltec_inputs_keep_first_entry(data_lmt_path):
    entry = {"data_toltec_path": data_lmt_path / "toltec", "obs_spec": "145609/0"}
    subtree = discover([entry])
    combined = discover(
        [entry, {"data_lmt_path": data_lmt_path, "obs_spec": "145609/{tel,0}"}]
    )
    assert len(subtree) == 2
    assert len(combined) == 4
    assert combined.iloc[:2][subtree.columns].equals(subtree)
    assert combined.iloc[2:]["interface"].tolist() == ["tel", "tel"]


def test_exact_locations_do_not_expand_wildcards(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An exact location was globbed or opened")

    monkeypatch.setattr(GlobSource, "resolve", unexpected)
    monkeypatch.setattr(xr, "open_dataset", unexpected)
    urls = ["https://example.org/a[1]*.nc?token=x*y", tmp_path / "literal[1]*.nc"]
    table = discover([{"sources": urls, "mode": "guess"}])
    assert [str(loc.url) for loc in table.source_loc] == [
        str(fileloc(v).url) for v in urls
    ]


def test_source_union_uses_native_validation_and_serialization(tmp_path):
    location = fileloc(tmp_path / "prevalidated.nc")
    pattern = GlobSource(glob=tmp_path / "*.nc")
    config = SourcesInputDiscovery(
        sources=[
            {"path": tmp_path / "exact[1].nc"},
            "https://example.org/a.nc?token=*",
            {"glob": str(tmp_path / "*.nc")},
            location,
            pattern,
        ]
    )
    assert isinstance(config.sources[0], FileLoc)
    assert isinstance(config.sources[2], GlobSource)
    assert config.sources[3] is location
    assert config.sources[4] is pattern
    dumped = config.model_dump(mode="json")
    assert dumped["sources"][0] == config.sources[0].model_dump(mode="json")
    assert dumped["sources"][1] == config.sources[1].model_dump(mode="json")
    restored = SourcesInputDiscovery.model_validate(dumped)
    assert restored.model_dump(mode="json") == dumped
    assert SourcesInputDiscovery.model_validate_json(config.model_dump_json()) == config


def test_discovery_config_has_an_explicit_inputs_field(tmp_path):
    data = [{"sources": [{"path": "relative.nc"}], "mode": "guess"}]
    config = InputDiscoveryConfig.model_validate(
        {"inputs": data}, context={"rootpath": tmp_path}
    )
    assert config.inputs[0].sources[0].path == tmp_path / "relative.nc"
    assert config.model_dump() == {"inputs": [config.inputs[0].model_dump()]}
    assert InputDiscoveryConfig().inputs == []
    assert InputDiscoveryConfig.model_validate_json(config.model_dump_json()) == config
    assert (
        InputDiscoveryConfig.model_validate_yaml(config.model_dump_yaml(mode="json"))
        == config
    )
    schema = InputDiscoveryConfig.model_json_schema()
    assert schema["type"] == "object"
    assert schema["properties"]["inputs"]["type"] == "array"
    for obsolete in (data, {"entries": data}, {"root": data}):
        with pytest.raises(ValidationError):
            InputDiscoveryConfig.model_validate(obsolete)
    with pytest.raises(ValidationError):
        InputDiscoveryConfig.model_validate({"inputs": tuple(data)})
    with pytest.raises(ValidationError, match="frozen"):
        config.inputs = []


def test_discovery_uses_standard_subconfig_transformer_and_patches(tmp_path):
    runtime = RuntimeContext(
        {"input_discovery": {"inputs": [{"sources": ["a.nc"], "mode": "guess"}]}}
    ).with_config(validation_context={"rootpath": tmp_path})
    discovery = InputDiscovery(runtime)
    assert (
        discovery.prepare_config_data(runtime.config) == runtime.config.input_discovery
    )
    assert discovery.prepare_runtime_config_data({"inputs": []}) == {
        "input_discovery": {"inputs": []}
    }
    with discovery.temporary_config(
        overrides={"inputs": {0: {"mode": "read"}}}
    ) as other:
        assert other.config.inputs[0].mode == "read"
        assert other.config.inputs[0].sources[0].path == tmp_path / "a.nc"
    assert discovery.config.inputs[0].mode == "guess"
    assert discovery.recreate().config == discovery.config


def test_globs_preserve_enumeration_order_and_context(tmp_path, monkeypatch):
    nested = tmp_path / "nested"
    nested.mkdir()
    for name in ("b.nc", "a.nc"):
        (nested / name).touch()
    (nested / "not_a_file.nc").mkdir()
    paths = [nested / "b.nc", nested / "a.nc"]

    def matches(pattern, *, recursive):
        assert pattern == str(tmp_path / "**/*.nc")
        assert recursive is True
        return iter([*paths, nested / "not_a_file.nc", paths[0]])

    monkeypatch.setattr(glob, "iglob", matches)
    runtime = RuntimeContext(
        {
            "input_discovery": {
                "inputs": [
                    {
                        "sources": [{"glob": "**/*.nc"}, "nested/a.nc"],
                        "mode": "guess",
                    }
                ]
            }
        }
    ).with_config(validation_context={"rootpath": tmp_path})
    table = InputDiscovery(runtime).discover()
    assert [loc.path for loc in table.source_loc] == paths


def test_only_double_star_searches_recursively(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    deeper = nested / "deeper"
    deeper.mkdir()
    paths = [root / "data.nc" for root in (tmp_path, nested, deeper)]
    for path in paths:
        path.touch()
    (tmp_path / "directory.nc").mkdir()
    (tmp_path / ".hidden.nc").touch()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden/data.nc").touch()

    def resolve(pattern):
        return [loc.path for loc in GlobSource(glob=tmp_path / pattern).resolve()]

    assert resolve("*.nc") == paths[:1]
    assert resolve("*/*.nc") == paths[1:2]
    assert set(resolve("**/*.nc")) == set(paths)


def test_empty_discovery_logs_and_returns_standard_empty_table(tmp_path):
    messages = []
    sink = logger.add(messages.append, level="DEBUG")
    try:
        pattern = str(tmp_path / "missing*.nc")
        assert GlobSource(glob=pattern).resolve() == []
        result = discover(
            [
                {"sources": [{"glob": pattern}], "obs_spec": "tcs-[-1]/tel"},
                {"data_lmt_path": tmp_path / "absent", "obs_spec": "tcs-[-1]/0"},
            ]
        )
    finally:
        logger.remove(sink)
    assert result.equals(ToltecFileAccessor.from_sources([]))
    assert any(pattern in message and "0 files" in message for message in messages)
    assert any("no selected candidates" in message for message in messages)


def test_entry_logs_include_settings_selection_and_deduplication(
    data_lmt_path, tmp_path
):
    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    other = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0001.nc"
    alias = tmp_path / source.name
    alias.symlink_to(source)
    messages = []
    sink = logger.add(messages.append, level="DEBUG")
    try:
        table = discover(
            [
                {
                    "sources": [source, source, other],
                    "obs_spec": "145609/2/tel",
                    "mode": "guess",
                },
                {"sources": [alias], "mode": "guess"},
                {"data_lmt_path": tmp_path / "missing", "obs_spec": "1/tel"},
            ]
        )
    finally:
        logger.remove(sink)
    assert len(table) == 1
    text = "\n".join(messages)
    assert "Input discovery: 3 configured entries" in text
    assert "inputs[0] (SourcesInputDiscovery)" in text
    assert source.name in text
    assert "inputs[1] (SourcesInputDiscovery)" in text
    assert "inputs[2] (LmtInputDiscovery)" in text
    assert str(tmp_path / "missing") in text
    assert (
        "3 sources -> 3 resolved locations -> 2 distinct candidates -> 1 selected; "
        "mode=guess, obs_spec=145609/2/tel"
    ) in text
    assert "mode=guess, obs_spec=1/tel; 0 candidates -> 0 selected" in text
    assert "inputs[2]: 0 selected candidates" in text
    assert "2 nonempty entries, 2 selected rows -> 1 distinct files" in text
    assert "1 duplicate locations or symlink targets removed" in text


def test_entry_settings_are_logged_before_a_read_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        msg = "header read failed"
        raise OSError(msg)

    monkeypatch.setattr(ToltecFileAccessor, "from_sources", fail)
    messages = []
    sink = logger.add(messages.append, level="DEBUG")
    try:
        with pytest.raises(OSError, match="header read failed"):
            discover([{"sources": [tmp_path / "unreadable.nc"], "mode": "read"}])
    finally:
        logger.remove(sink)
    text = "\n".join(messages)
    assert "inputs[0] (SourcesInputDiscovery)" in text
    assert "unreadable.nc" in text
    assert "mode='read'" in text
    assert "selected candidates" not in text


@pytest.mark.parametrize("kind", ["sources", "data_lmt_path", "data_toltec_path"])
def test_discovery_can_record_population_failures(tmp_path, kind):
    filename = "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    candidate = tmp_path / "toltec/tcs/toltec0" / filename
    candidate.parent.mkdir(parents=True)
    # A corrupt candidate deliberately tests failure handling, not acquisition data.
    candidate.write_bytes(b"not a netCDF file")
    entry = (
        {"sources": [candidate]}
        if kind == "sources"
        else {kind: tmp_path if kind == "data_lmt_path" else tmp_path / "toltec"}
    )
    handler = InputDiscovery(
        RuntimeContext(
            {
                "input_discovery": {"inputs": [entry | {"mode": "read"}]},
            }
        )
    )
    with pytest.raises(ValueError):
        handler.discover()
    table = handler.discover(on_error="record")
    assert len(table) == 1
    (diagnostic,) = table.diagnostics.iloc[0]
    assert diagnostic.stage == "open" and diagnostic.exception_type == "ValueError"


@pytest.mark.parametrize("rootpath", [None, "relative-root", "~"])
@pytest.mark.parametrize("pattern", ["*.nc", "**/*.nc", "*/../*.nc", "~/data/*.nc"])
def test_glob_uses_shared_path_context_policy(rootpath, pattern):
    context = {"rootpath": rootpath}
    expected = TypeAdapter(AnyPath).validate_python(pattern, context=context)
    config = GlobSource.model_validate({"glob": pattern}, context=context)
    assert config.glob == expected


def test_glob_context_survives_a_different_working_directory(tmp_path, monkeypatch):
    configured_root = tmp_path / "configured"
    configured_root.mkdir()
    match = configured_root / "data.nc"
    match.touch()
    unrelated = tmp_path / "cwd"
    unrelated.mkdir()
    config = GlobSource.model_validate(
        {"glob": "*.nc"}, context={"rootpath": configured_root}
    )
    monkeypatch.chdir(unrelated)
    assert [loc.path for loc in config.resolve()] == [match]


def test_glob_without_context_does_not_capture_working_directory(tmp_path, monkeypatch):
    config = GlobSource(glob="*.nc")
    assert config.glob == Path("*.nc")
    match = tmp_path / "data.nc"
    match.touch()
    monkeypatch.chdir(tmp_path)
    assert [loc.path for loc in config.resolve()] == [match]


def test_glob_parent_component_is_not_resolved_before_matching(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    match = tmp_path / "data.nc"
    match.touch()
    config = GlobSource.model_validate(
        {"glob": "nest*/../*.nc"}, context={"rootpath": tmp_path}
    )
    assert config.glob == tmp_path / "nest*/../*.nc"
    assert [loc.path for loc in config.resolve()] == [match]
    unmatched = GlobSource.model_validate(
        {"glob": "missing*/../*.nc"}, context={"rootpath": tmp_path}
    )
    assert unmatched.resolve() == []


def test_discovery_config_pickle_roundtrip_retains_context(tmp_path, monkeypatch):
    nested = tmp_path / "nested"
    nested.mkdir()
    match = nested / "data.nc"
    match.touch()
    runtime = RuntimeContext(
        {
            "input_discovery": {
                "inputs": [
                    {
                        "sources": ["nested/data.nc", {"glob": "nested/*.nc"}],
                        "obs_spec": "tcs-1",
                        "mode": "guess",
                    }
                ]
            }
        }
    ).with_config(validation_context={"rootpath": tmp_path})
    config = InputDiscovery(runtime).config
    # Only this test's own pickle data is deserialized.
    restored = pickle.loads(pickle.dumps(config))  # noqa: S301
    assert isinstance(restored, InputDiscoveryConfig)
    assert restored.model_dump(mode="json") == config.model_dump(mode="json")
    monkeypatch.chdir(nested)
    entry = restored.inputs[0]
    assert entry.sources[0].path == match
    assert [loc.path for loc in entry.sources[1].resolve()] == [match]


def test_selectors_keep_their_own_matches_and_order(data_lmt_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Guess discovery opened data")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    table = discover(
        [
            {"data_lmt_path": data_lmt_path, "obs_spec": spec}
            for spec in ("145609/2/tel", "145609/1/tel", "145609/tel")
        ]
    )
    assert table.scannum.tolist() == [2, 1]
    assert table.index.tolist() == [0, 1]


def test_positional_selection_is_per_entry(data_lmt_path):
    table = discover(
        [
            {"data_lmt_path": data_lmt_path, "obs_spec": "tcs-[-1]/0"},
            {"data_lmt_path": data_lmt_path, "obs_spec": "tcs-[-1]/tel"},
        ]
    )
    assert set(table.interface) == {"toltec0", "tel"}
    assert set(table.loc[table.interface == "toltec0", "obsnum"]) == {145609}
    assert set(table.loc[table.interface == "tel", "obsnum"]) == {145611}


@pytest.mark.parametrize("missing", ["145609/99/tel", "999999/tel"])
def test_empty_discovery_entry_does_not_discard_other_matches(data_lmt_path, missing):
    table = discover(
        [
            {"data_lmt_path": data_lmt_path, "obs_spec": spec}
            for spec in (missing, "145609/tel", missing)
        ]
    )
    expected = discover([{"data_lmt_path": data_lmt_path, "obs_spec": "145609/tel"}])
    assert table.equals(expected)


def test_symlink_deduplication_follows_exact_selection(data_lmt_path, tmp_path):
    original = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    (tmp_path / "tel").mkdir()
    alias = tmp_path / "tel/tel_toltec_2025-10-31_145609_00_0001.nc"
    alias.symlink_to(original)
    matched = alias.with_name(original.name)
    matched.symlink_to(original)
    table = discover([{"data_lmt_path": tmp_path, "obs_spec": "145609/2/tel"}])
    assert [loc.path for loc in table.source_loc] == [matched]
    assert table.file_scannum.tolist() == [2]


@pytest.mark.parametrize("mode", ["guess_or_read", "read"])
def test_renamed_inputs_use_sources_with_metadata(data_lmt_path, tmp_path, mode):
    original = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    renamed = tmp_path / "downloaded.nc"
    renamed.symlink_to(original)
    table = discover(
        [
            {
                "sources": [{"glob": str(tmp_path / "*.nc")}],
                "obs_spec": "tcs-145609-0-2/tel",
                "mode": mode,
            }
        ]
    )
    assert [loc.path for loc in table.source_loc] == [renamed]
    assert table.uid_raw_obs_file.tolist() == ["tcs-145609-0-2-tel"]
    assert table.file_obsnum.isna().all()


def test_read_mode_precedes_obs_spec_for_conflicting_filename(data_lmt_path, tmp_path):
    original = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    alias = tmp_path / "tel_toltec_2025-10-31_145609_00_0001.nc"
    alias.symlink_to(original)
    entry = {"sources": [alias], "obs_spec": "145609/2/tel"}
    # A recognizable filename does not trigger guess_or_read.
    assert discover([entry]).empty
    table = discover([entry | {"mode": "read"}])
    assert table.scannum.tolist() == [2]
    assert table.file_scannum.tolist() == [1]
