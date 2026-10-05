"""Registered discovery inputs use ordinary nested Pydantic validation."""

import pickle

import pytest
from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
    model_validator,
)
from tollan.config import RuntimeContext
from tollan.config.types import FileLocField
from tollan.utils.fileloc import fileloc

from tolteca_datamodels import input_discovery
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.input_discovery import (
    InputDiscovery,
    InputDiscoveryConfig,
    InputDiscoveryEntry,
    LmtInputDiscovery,
    SourcesInputDiscovery,
    ToltecInputDiscovery,
)


class ArchiveInput(InputDiscoveryEntry):
    archive: FileLocField

    def discover(self, *, on_error="raise"):
        return FileAccessor.from_sources([self.archive], on_error=on_error)


@pytest.fixture(autouse=True)
def restore_registry():
    registered = dict(input_discovery._input_discovery_models)
    yield
    input_discovery._input_discovery_models.clear()
    input_discovery._input_discovery_models.update(registered)
    InputDiscoveryConfig.model_rebuild(force=True)


def test_builtin_typed_entries_and_schema(tmp_path):
    entries: list[InputDiscoveryEntry] = [
        LmtInputDiscovery(data_lmt_path=tmp_path),
        ToltecInputDiscovery(data_toltec_path=tmp_path),
        SourcesInputDiscovery(sources=[fileloc("https://example.org/data.nc")]),
    ]
    config = InputDiscoveryConfig(inputs=entries)
    assert all(a is b for a, b in zip(config.inputs, entries, strict=True))
    assert config.model_dump() == {"inputs": [entry.model_dump() for entry in entries]}
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
    InputDiscoveryConfig.register_input("archive", ArchiveInput)
    assert InputDiscoveryConfig.__pydantic_validator__ is validator

    def unexpected(*args, **kwargs):
        pytest.fail("Schema was rebuilt during config validation or serialization")

    monkeypatch.setattr(InputDiscoveryConfig, "model_rebuild", unexpected)
    runtime = RuntimeContext(
        {"input_discovery": {"inputs": [{"archive": "data.nc"}]}}
    ).with_config(validation_context={"rootpath": tmp_path})
    discovery = InputDiscovery(runtime)
    config = discovery.config
    entry = config.inputs[0]
    assert isinstance(entry, ArchiveInput)
    assert entry.archive.path == tmp_path / "data.nc"
    assert discovery.discover().source_loc.iloc[0] == entry.archive
    assert config.model_dump() == {"inputs": [entry.model_dump()]}
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

        def discover(self, *, on_error="raise"):
            return FileAccessor.from_sources(self.sources)

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


def test_registration_requires_concrete_model_and_identifying_key():
    with pytest.raises(TypeError, match="concrete"):
        InputDiscoveryConfig.register_input("archive", InputDiscoveryEntry)
    with pytest.raises(TypeError, match="concrete"):
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
