"""Selected data models supply stores and handlers, not filename registrations."""

import inspect

import pytest
from pydantic import Field, ValidationError
from tollan.config import RuntimeContext

from tolteca_datamodels.core import data_model as data_model_module
from tolteca_datamodels.core.data_model import DataModelSpec
from tolteca_datamodels.core.file import source_trace_issues
from tolteca_datamodels.core.file_store import (
    FileStoreBase,  # noqa: TC001 -- Pydantic needs the local model's annotation at runtime.
)
from tolteca_datamodels.core.source import SourceReaderRules
from tolteca_datamodels.core.types import FrozenDataModel
from tolteca_datamodels.data_model import DataModelSet
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.input_discovery import InputDiscovery, InputDiscoveryConfig
from tolteca_datamodels.lmt.data_model import data_model as lmt_data_model
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.file_store import LmtFileStore
from tolteca_datamodels.lmt.input_discovery import LmtInputDiscovery
from tolteca_datamodels.lmt.types import LmtDataModel
from tolteca_datamodels.lmt_base.input_discovery import SourcesInputDiscovery
from tolteca_datamodels.toltec.data_model import data_model as toltec_data_model
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.file_store import ToltecFileStore
from tolteca_datamodels.toltec.types import ToltecDataModel
from tolteca_dummy_instru.data_model import data_model as dummy_data_model
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.file_store import DummyInstruFileStore
from tolteca_dummy_instru.types import DummyInstruDataModel


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setattr(data_model_module._data_models, "_entries", {})
    for model in (lmt_data_model, toltec_data_model, dummy_data_model):
        model.register()


@pytest.mark.parametrize(
    ("info", "name", "accessor"),
    [
        (LmtDataModel, "lmt", LmtFileAccessor),
        (ToltecDataModel, "toltec", ToltecFileAccessor),
        (DummyInstruDataModel, "dummy_instru", DummyInstruFileAccessor),
    ],
)
def test_declared_names_select_handlers(info, name, accessor):
    assert info.name == name
    assert (
        DataModelSet.from_select([info.name]).data_models[name].file_accessor_cls
        is accessor
    )


def test_set_unions_store_requirements_without_enabling_companion_handlers():
    data_models = DataModelSet.from_select(["toltec", "dummy_instru"])
    assert data_models.file_store_types == (
        *lmt_data_model.file_store_types,
        ToltecFileStore,
        DummyInstruFileStore,
    )
    assert set(data_models.data_models) == {"toltec", "dummy_instru"}
    assert DataModelSet.from_select([]).file_store_types == ()
    lmt_models = DataModelSet.from_select(["lmt"])
    assert lmt_models.file_store_types == lmt_data_model.file_store_types
    assert issubclass(LmtFileStore, lmt_models.file_store_types)


def test_store_requirements_validate_class_type():
    with pytest.raises(ValidationError, match="subclass of FileStoreBase"):
        DataModelSpec.model_validate(
            {"file_accessor_cls": ToltecFileAccessor, "file_store_types": (str,)}
        )


def test_spec_is_frozen_and_requires_declared_container_types():
    spec = DataModelSpec(file_accessor_cls=ToltecFileAccessor)
    assert isinstance(spec, FrozenDataModel)
    assert spec.file_store_types == ()
    with pytest.raises(ValidationError, match="frozen_instance"):
        setattr(spec, "file_store_types", (ToltecFileStore,))  # noqa: B010 -- Test frozen assignment.
    with pytest.raises(ValidationError, match="tuple_type"):
        DataModelSpec.model_validate(
            {
                "file_accessor_cls": ToltecFileAccessor,
                "file_store_types": [ToltecFileStore],
            }
        )


def test_spec_validates_default_store_requirements():
    class InvalidDefault(DataModelSpec):
        file_store_types: tuple[type[FileStoreBase], ...] = Field(default=(str,))

    with pytest.raises(ValidationError, match="subclass of FileStoreBase"):
        InvalidDefault(file_accessor_cls=ToltecFileAccessor)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (["toltec"], (ToltecFileAccessor,)),
        ({"excludes": ["lmt"]}, (ToltecFileAccessor, DummyInstruFileAccessor)),
        ("lmt|toltec", (LmtFileAccessor, ToltecFileAccessor)),
        (["not_registered"], ()),
        ([], ()),
    ],
)
def test_set_selects_registered_names(selection, expected):
    assert (
        tuple(
            model.file_accessor_cls
            for model in DataModelSet.from_select(selection).data_models.values()
        )
        == expected
    )


def test_set_keeps_selected_handlers_when_new_support_is_registered(monkeypatch):
    monkeypatch.setattr(data_model_module._data_models, "_entries", {})
    for model in (lmt_data_model, toltec_data_model):
        model.register()
    selected = DataModelSet.from_select()
    dummy_data_model.register()
    assert set(selected.data_models) == {"lmt", "toltec"}
    assert set(DataModelSet.from_select().data_models) == {
        "lmt",
        "toltec",
        "dummy_instru",
    }


def test_workflow_set_is_not_a_file_accessor_argument():
    assert "data_models" not in inspect.signature(FileAccessor).parameters
    assert "data_models" not in inspect.signature(FileAccessor.from_sources).parameters


def test_explicit_set_can_use_an_unregistered_definition():
    class LocalFiles(ToltecFileAccessor):
        pass

    definition = DataModelSpec(file_accessor_cls=LocalFiles)
    data_models = DataModelSet(data_models={"toltec": definition})
    files = FileAccessor.partial(accessor_types=data_models.file_accessor_types)
    table = files.from_sources(
        ["toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"]
    )
    assert table.data_kind_toltec.notna().all()
    assert "data_kind_dummy_instru" not in table
    assert (
        DataModelSet.from_select(["toltec"]).data_models["toltec"].file_accessor_cls
        is ToltecFileAccessor
    )


@pytest.mark.parametrize("level", ["global", "entry"])
@pytest.mark.parametrize("invalid", [None, "[", {"includes": [], "excludes": []}])
def test_invalid_capability_selection_rejected_in_config(level, invalid):
    data = {"inputs": [{"sources": ["unidentified.nc"]}]}
    target = data if level == "global" else data["inputs"][0]
    target["data_model_select"] = invalid
    with pytest.raises(ValidationError):
        InputDiscoveryConfig.model_validate(data)


@pytest.mark.parametrize(
    ("global_select", "entry_select", "columns"),
    [
        (".*", ["toltec"], {"data_kind_toltec"}),
        (["toltec"], ".*", {"data_kind_toltec"}),
        ({"excludes": ["lmt"]}, {"excludes": ["toltec"]}, {"data_kind_dummy_instru"}),
        (["toltec"], ["dummy_instru"], set()),
    ],
)
def test_global_and_entry_capabilities_intersect(global_select, entry_select, columns):
    runtime = RuntimeContext(
        {
            "input_discovery": {
                "data_model_select": global_select,
                "inputs": [
                    {
                        "sources": ["chan01-tcs-1-0-2-20251031T120000.nc"],
                        "mode": "guess",
                        "data_model_select": entry_select,
                    }
                ],
            }
        }
    )
    table = InputDiscovery(runtime).discover()
    assert {
        name
        for name in table
        if isinstance(name, str) and name.startswith("data_kind_")
    } == columns
    assert table.interface.tolist() == ["chan01"]


def test_config_cannot_widen_task_set_or_invoke_excluded_readers(
    data_lmt_path, dummy_tel_path, monkeypatch
):
    from tolteca_datamodels.core.file import accessor as file_module

    opened = []
    open_source = file_module.FileAccessor._open_source

    def traced(source, **kwargs):
        opened.append(source)
        return open_source(source, **kwargs)

    def unexpected(data):
        pytest.fail("Discovery invoked an excluded identifier")

    monkeypatch.setattr(file_module.FileAccessor, "_open_source", staticmethod(traced))
    monkeypatch.setattr(
        DummyInstruFileAccessor,
        "_reader_rules",
        SourceReaderRules(
            rules=DummyInstruFileAccessor._reader_rules.rules,
            prepare_record=unexpected,
        ),
    )
    runtime = RuntimeContext(
        {
            "input_discovery": {
                "data_model_select": ".*",
                "inputs": [
                    {
                        "sources": [
                            data_lmt_path
                            / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
                            dummy_tel_path,
                        ],
                        "data_model_select": ".*",
                        "mode": "read",
                    }
                ],
            }
        }
    )
    data_models = DataModelSet.from_select(["toltec"])
    table = InputDiscovery(runtime).discover(data_models=data_models)
    assert "data_kind_dummy_instru" not in table
    assert table.t0_mjd.notna().tolist() == [True, False]
    assert table.file_instru.tolist() == ["lmt", "lmt"]
    assert table.file_data_models.tolist() == [
        ["lmt", "toltec"],
        ["lmt", "dummy_instru"],
    ]
    assert opened == [table.source.iloc[0]]
    assert table.source_trace.map(source_trace_issues).iloc[1] == []


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_empty_selection_searches_no_stores_but_retains_explicit_sources(
    data_lmt_path, monkeypatch, mode
):
    def unexpected(*args, **kwargs):
        pytest.fail("Filename-only discovery opened a resource")

    monkeypatch.setattr(FileAccessor, "_iter_metadata_sources", unexpected)
    store = LmtInputDiscovery.model_validate(
        {
            "data_lmt_path": data_lmt_path,
            "obs_spec": "145609/tel",
            "data_model_select": [],
            "mode": mode,
        }
    )
    sources = SourcesInputDiscovery.model_validate(
        {
            "sources": [{"glob": str(data_lmt_path / "tel" / "*.nc")}],
            "obs_spec": "145609/tel",
            "data_model_select": [],
            "mode": mode,
        }
    )
    assert store.discover().empty
    table = sources.discover()
    assert table.obsnum.tolist() == [145609, 145609]
    assert table.interface.tolist() == ["tel", "tel"]
    assert not any(
        isinstance(name, str) and name.startswith("data_kind_") for name in table
    )
    assert "uid_raw_obs" not in table
    assert "t0_mjd" not in table


def test_filename_only_obs_spec_requires_support_for_numeric_labels():
    from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

    data_models = DataModelSet.from_select([])
    sources = ["toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"]
    files = FileAccessor.partial(accessor_types=data_models.file_accessor_types)
    table = files.from_sources(sources)
    assert len(files(table).query_obs_spec(ObsSpec.parse("*/toltec8"))) == 1
    assert files(table).query_obs_spec(ObsSpec.parse("*/8")).empty


def test_entries_keep_separate_classifier_choices_after_combining():
    config = {
        "input_discovery": {
            "inputs": [
                {
                    "sources": ["one.nc"],
                    "mode": "guess",
                    "data_model_select": ["toltec"],
                },
                {
                    "sources": ["two.nc"],
                    "mode": "guess",
                    "data_model_select": ["dummy_instru"],
                },
            ]
        }
    }
    table = InputDiscovery(RuntimeContext(config)).discover()
    assert table.data_kind_toltec.notna().tolist() == [True, False]
    assert table.data_kind_dummy_instru.notna().tolist() == [False, True]
    assert "data_kind" not in table
