"""Registered discovery inputs use ordinary nested Pydantic validation."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar

import pytest
from pydantic import (
    BaseModel,
    Field,
    PlainSerializer,
    PlainValidator,
    TypeAdapter,
    ValidationError,
    create_model,
    model_validator,
)
from tollan.config import RuntimeContext
from tollan.config.types import FileLocField
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.core.file import FileAccessor
from tolteca_datamodels.core.file_store import FileStoreBase
from tolteca_datamodels.core.input_discovery import (
    FileStoreInputDiscovery,
    InputDiscoveryEntry,
)
from tolteca_datamodels.core.input_discovery import config as input_discovery
from tolteca_datamodels.core.obs_spec import FieldConstraint, ObsSpecBase, Selector
from tolteca_datamodels.input_discovery import InputDiscovery, InputDiscoveryConfig
from tolteca_datamodels.lmt.input_discovery import LmtInputDiscovery
from tolteca_datamodels.lmt_base.input_discovery import SourcesInputDiscovery
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec.input_discovery import ToltecInputDiscovery
from tolteca_dummy_instru.input_discovery import DummyInstruInputDiscovery


class ArchiveInput(InputDiscoveryEntry):
    archive: FileLocField

    def discover(self, *, data_models=None, on_error="raise"):
        data_models = self._select_data_models(data_models)
        files = FileAccessor.partial(accessor_types=data_models.file_accessor_types)
        return files.from_sources([self.archive], on_error=on_error)


@pytest.fixture(autouse=True)
def restore_registry():
    registered = input_discovery._input_discovery_models
    yield
    input_discovery._input_discovery_models = registered
    InputDiscoveryConfig.model_rebuild(force=True)


def test_builtin_typed_entries_and_schema(tmp_path):
    entries: list[InputDiscoveryEntry] = [
        LmtInputDiscovery(data_lmt_path=tmp_path),
        ToltecInputDiscovery(data_toltec_path=tmp_path),
        SourcesInputDiscovery(sources=[fileloc("https://example.org/data.nc")]),
    ]
    config = InputDiscoveryConfig(inputs=entries)
    assert all(a is b for a, b in zip(config.inputs, entries, strict=True))
    assert config.model_dump()["inputs"] == [entry.model_dump() for entry in entries]
    assert config.data_model_select.model_dump(mode="json") == {"pattern": ".*"}
    assert all("type" not in entry for entry in config.model_dump()["inputs"])
    assert InputDiscoveryConfig.model_validate_json(config.model_dump_json()) == config
    variants = InputDiscoveryConfig.model_json_schema()["properties"]["inputs"][
        "items"
    ]["oneOf"]
    assert {item["$ref"].split("/")[-1] for item in variants} == {
        "LmtInputDiscovery",
        "ToltecInputDiscovery",
        "SourcesInputDiscovery",
    }


@pytest.mark.parametrize(
    ("entry_cls", "path_key"),
    [
        (LmtInputDiscovery, "data_lmt_path"),
        (ToltecInputDiscovery, "data_toltec_path"),
        (DummyInstruInputDiscovery, "data_dummy_instru_path"),
    ],
)
@pytest.mark.parametrize(
    "selector_data",
    [{}, {"obs_spec": "tcs-001"}, {"obs_spec": 1}, {"obs_spec": ObsSpec(obsnum=1)}],
)
def test_inherited_store_selector_field(entry_cls, path_key, selector_data, tmp_path):
    entry = entry_cls.model_validate({path_key: tmp_path, **selector_data})
    expected = ObsSpec.parse(selector_data.get("obs_spec", "*"))
    assert type(entry.obs_spec) is ObsSpec
    assert entry.obs_spec == expected
    assert entry.model_dump()["obs_spec"] == str(expected)
    assert entry_cls.model_validate_json(entry.model_dump_json()) == entry
    assert entry_cls.model_json_schema()["properties"]["obs_spec"]["default"] == "*"


def test_registered_store_uses_its_bound_selector_field(tmp_path):
    @dataclass(frozen=True)
    class SequenceSpec(ObsSpecBase):
        sequence: Selector = None
        _field_constraints: ClassVar = {"sequence": FieldConstraint(kind="integer")}
        _primary_field: ClassVar = "sequence"

    sequence_spec_field = Annotated[
        SequenceSpec,
        PlainValidator(SequenceSpec.parse, json_schema_input_type=str | int),
        PlainSerializer(str, return_type=str),
    ]

    class SequenceStore(FileStoreBase[SequenceSpec]):
        def _iter_local_glob_patterns(self, spec, subpath):
            return []

    class SequenceInput(FileStoreInputDiscovery[sequence_spec_field]):
        sequence_path: Path

        @property
        def _store(self) -> SequenceStore:
            return SequenceStore(self.sequence_path)

    InputDiscoveryConfig.register_input("sequence_path", SequenceInput)
    for selector_data in ({}, {"obs_spec": "001"}, {"obs_spec": 1}):
        config = InputDiscoveryConfig.model_validate(
            {"inputs": [{"sequence_path": tmp_path, **selector_data}]}
        )
        entry = config.inputs[0]
        assert isinstance(entry, SequenceInput)
        expected = SequenceSpec.parse(selector_data.get("obs_spec", "*"))
        assert type(entry.obs_spec) is SequenceSpec
        assert entry.obs_spec == expected
        assert entry.model_dump()["obs_spec"] == str(expected)
        assert (
            InputDiscoveryConfig.model_validate_json(config.model_dump_json()) == config
        )
        assert entry._store.glob_obs_spec(entry.obs_spec) == []

    with pytest.raises(ValidationError):
        SequenceInput.model_validate({"sequence_path": tmp_path, "obs_spec": "tcs-1"})


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"unknown_path": "/data"},
        {"sources": [], "data_lmt_path": "/data"},
        {"data_lmt_path": "/data", "data_toltec_path": "/data/toltec"},
    ],
)
def test_missing_or_ambiguous_discovery_key(entry):
    with pytest.raises(ValidationError) as exc:
        InputDiscoveryConfig.model_validate({"inputs": [entry]})
    assert exc.value.errors()[0]["type"] == "input_discovery_key"
    assert exc.value.errors()[0]["loc"] == ("inputs", 0)
    assert exc.value.errors()[0]["ctx"] == {
        "keys": ", ".join(input_discovery._input_discovery_models)
    }


def test_only_selected_model_is_validated():
    calls = []

    class ObservedArchive(ArchiveInput):
        @model_validator(mode="before")
        @classmethod
        def track_validation(cls, data):
            calls.append(data)
            return data

    InputDiscoveryConfig.register_input("archive", ObservedArchive)
    config = InputDiscoveryConfig.model_validate({"inputs": [{"sources": ["data.nc"]}]})
    assert isinstance(config.inputs[0], SourcesInputDiscovery)
    with pytest.raises(ValidationError) as exc:
        InputDiscoveryConfig.model_validate({"inputs": [{"sources": []}]})
    errors = exc.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("inputs", 0, "sources", "sources")
    assert calls == []
    data = {"archive": "data.nc"}
    config = InputDiscoveryConfig.model_validate({"inputs": [data]})
    assert isinstance(config.inputs[0], ObservedArchive)
    assert calls == [data]


def test_registration_context_serialization_and_discovery(tmp_path, monkeypatch):
    InputDiscoveryConfig.register_input("archive", ArchiveInput)
    validator = InputDiscoveryConfig.__pydantic_validator__
    with pytest.raises(ValueError, match="already registered"):
        InputDiscoveryConfig.register_input("archive", ArchiveInput)
    assert InputDiscoveryConfig.__pydantic_validator__ is validator

    def unexpected(*args, **kwargs):
        pytest.fail("Schema was rebuilt during config validation or serialization")

    monkeypatch.setattr(InputDiscoveryConfig, "model_rebuild", unexpected)
    runtime = RuntimeContext(
        {"input_discovery": {"inputs": [{"archive": "data.nc"}]}}
    ).with_config(validation_context={"root_path": tmp_path})
    discovery = InputDiscovery(runtime)
    config = discovery.config
    entry = config.inputs[0]
    assert isinstance(entry, ArchiveInput)
    assert entry.archive.path == tmp_path / "data.nc"
    assert discovery.discover().source_loc.iloc[0] == entry.archive
    assert config.model_dump()["inputs"] == [entry.model_dump()]
    assert InputDiscoveryConfig.model_validate_json(config.model_dump_json()) == config
    assert (
        InputDiscoveryConfig.model_validate_yaml(config.model_dump_yaml(mode="json"))
        == config
    )
    assert pickle.loads(pickle.dumps(config)) == config  # noqa: S301 -- own test data.
    assert "ArchiveInput" in InputDiscoveryConfig.model_json_schema()["$defs"]
    assert InputDiscoveryConfig.__pydantic_validator__ is validator


def test_precompiled_schemas_are_snapshots():
    earlier = TypeAdapter(InputDiscoveryConfig)
    before = earlier.validate_python({"inputs": []})
    InputDiscoveryConfig.register_input("archive", ArchiveInput)
    data = {"inputs": [{"archive": "https://example.org/data.nc"}]}
    assert isinstance(InputDiscoveryConfig.model_validate(data).inputs[0], ArchiveInput)
    with pytest.raises(ValidationError, match="discovery key"):
        earlier.validate_python(data)
    assert before.inputs == []

    class Outer(BaseModel):
        discovery: InputDiscoveryConfig

    assert isinstance(
        Outer.model_validate({"discovery": data}).discovery.inputs[0], ArchiveInput
    )


def test_registration_rejects_conflicts_without_changing_schema():
    class OtherSources(InputDiscoveryEntry):
        sources: list[str]

        def discover(self, *, data_models=None, on_error="raise"):
            data_models = self._select_data_models(data_models)
            files = FileAccessor.partial(accessor_types=data_models.file_accessor_types)
            return files.from_sources(self.sources, on_error=on_error)

    validator = InputDiscoveryConfig.__pydantic_validator__
    with pytest.raises(ValueError, match="already registered"):
        InputDiscoveryConfig.register_input("sources", OtherSources)
    assert InputDiscoveryConfig.__pydantic_validator__ is validator
    assert isinstance(
        InputDiscoveryConfig.model_validate(
            {"inputs": [{"sources": ["data.nc"]}]}
        ).inputs[0],
        SourcesInputDiscovery,
    )


def test_registration_rejects_model_under_another_key():
    class AlternateArchive(ArchiveInput):
        alternate: str

    InputDiscoveryConfig.register_input("archive", AlternateArchive)
    validator = InputDiscoveryConfig.__pydantic_validator__
    with pytest.raises(ValueError, match="already registered"):
        InputDiscoveryConfig.register_input("alternate", AlternateArchive)
    assert InputDiscoveryConfig.__pydantic_validator__ is validator
    assert "alternate" not in input_discovery._input_discovery_models
    entry = InputDiscoveryConfig.model_validate(
        {"inputs": [{"archive": "data.nc", "alternate": "other"}]}
    ).inputs[0]
    assert isinstance(entry, AlternateArchive)


@pytest.mark.parametrize(
    ("key", "model_cls", "error_loc"),
    [
        (1, ArchiveInput, (1, "[key]")),
        (b"archive", ArchiveInput, ("b'archive'", "[key]")),
        ("archive", None, ("archive",)),
    ],
)
def test_registration_field_errors_leave_schema_and_registry_unchanged(
    key, model_cls, error_loc
):
    previous = dict(input_discovery._input_discovery_models)
    validator = InputDiscoveryConfig.__pydantic_validator__
    with pytest.raises(ValidationError) as exc:
        InputDiscoveryConfig.register_input(key, model_cls)
    assert exc.value.errors()[0]["loc"] == error_loc
    assert input_discovery._input_discovery_models == previous
    assert InputDiscoveryConfig.__pydantic_validator__ is validator


def test_registration_requires_concrete_model_and_identifying_key():
    with pytest.raises(ValueError, match="concrete"):
        InputDiscoveryConfig.register_input("archive", InputDiscoveryEntry)
    with pytest.raises(ValidationError, match="subclass of InputDiscoveryEntry"):
        InputDiscoveryConfig.register_input("archive", BaseModel)  # ty: ignore[invalid-argument-type] -- invalid registration.
    with pytest.raises(ValueError, match="required"):
        InputDiscoveryConfig.register_input("missing", ArchiveInput)

    class OptionalArchive(ArchiveInput):
        archive: FileLocField = fileloc("https://example.org/data.nc")

    class AliasedArchive(ArchiveInput):
        archive: FileLocField = Field(alias="location")

    class SerializedAliasArchive(ArchiveInput):
        archive: FileLocField = Field(serialization_alias="location")

    for model in (OptionalArchive, AliasedArchive, SerializedAliasArchive):
        with pytest.raises(ValueError, match="required unaliased"):
            InputDiscoveryConfig.register_input("archive", model)


def test_unregistered_typed_entry_is_rejected():
    entry = ArchiveInput(archive=fileloc("https://example.org/data.nc"))
    with pytest.raises(ValidationError, match="discovery key"):
        InputDiscoveryConfig(inputs=[entry])
    InputDiscoveryConfig.register_input("archive", ArchiveInput)
    assert InputDiscoveryConfig(inputs=[entry]).inputs[0] is entry


def test_failed_schema_build_preserves_registered_inputs():
    from pydantic import PydanticUndefinedAnnotation

    UnresolvedInput = create_model(
        "UnresolvedInput",
        __base__=ArchiveInput,
        unresolved=("UndefinedArchiveSetting", ...),
    )

    before = dict(input_discovery._input_discovery_models)
    with pytest.raises(PydanticUndefinedAnnotation):
        InputDiscoveryConfig.register_input("unresolved", UnresolvedInput)
    assert input_discovery._input_discovery_models == before
    assert isinstance(
        InputDiscoveryConfig.model_validate(
            {"inputs": [{"sources": ["https://example.org/data.nc"]}]}
        ).inputs[0],
        SourcesInputDiscovery,
    )
