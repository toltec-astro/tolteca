"""Declared attribute columns and file-table projection."""

import re
from dataclasses import MISSING
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from types import SimpleNamespace
from typing import Annotated, Any, ClassVar, Literal, cast

import astropy.units as u
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    computed_field,
    create_model,
    field_serializer,
    field_validator,
)
from tollan.config.types import (
    FrequencyQuantityField,
    QuantityField,
    SkyCoordField,
    TimeQuantityField,
    UnixTimeField,
    quantity_field,
)

from tolteca_datamodels.file import FilepathInfo
from tolteca_datamodels.lmt import LmtFilepathInfo
from tolteca_datamodels.lmt.tel import LmtTelAccessor
from tolteca_datamodels.lmt_base.types import LabelT, SeqNumT
from tolteca_datamodels.model_table import (
    ModelColumn,
    ModelColumnMapper,
    ModelTableAccessor,
    QuantityColumnRule,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)
from tolteca_datamodels.toltec.file import (
    ToltecFileAccessor,
    ToltecFilepathInfo,
    get_info_from_source,
)
from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor


class Reading(BaseModel):
    value: int | None


class Measurement(BaseModel):
    label: str
    reading: Reading | None


type CountT = Annotated[int, Field(ge=0)]
type ChoiceT = Literal["one", "two"]


class StateCode(IntEnum):
    ready = 1


@pytest.mark.parametrize(
    ("annotation", "dtype"),
    [
        (bool, "boolean"),
        (int, "Int64"),
        (float, "Float64"),
        (str, "string"),
        (bytes, None),
        (type(None), None),
        (int | None, "Int64"),
        (int | str, None),
        (Annotated[int, Field(ge=0)], "Int64"),
        (LabelT, "string"),
        (SeqNumT, "Int64"),
        (CountT | None, "Int64"),
        (Literal["one", "two"], "string"),
        (Literal[1, 2] | None, "Int64"),
        (ChoiceT, "string"),
        (Literal[ChoiceT, "three"], "string"),
    ],
)
def test_column_mapper_selects_scalar_annotations(annotation, dtype):
    model = create_model("Scalar", value=(annotation, ...))
    assert ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(model) == {"value": ModelColumn("value", dtype=dtype)}


@pytest.mark.parametrize(
    "annotation",
    [Any, list[int], dict[str, int], Reading, Reading | None, int | list[int]],
)
def test_column_mapper_leaves_other_types_explicit(annotation):
    model = create_model("NonScalar", value=(annotation, ...))
    assert (
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")],
            type_rules=[ScalarColumnRule()],
        )(model)
        == {}
    )
    assert ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(model) | {"value": ModelColumn("value")} == {"value": ModelColumn("value")}


def test_column_mapper_uses_input_field_names_without_evaluation():
    def unexpected(*args, **kwargs):
        pytest.fail("Column declaration evaluated a model value")

    class Parent(BaseModel):
        label: str

    class Model(Parent):
        value: int = Field(default_factory=unexpected, alias="other", exclude=True)
        constant: ClassVar[int] = 3

        @computed_field
        @property
        def computed(self) -> int:
            return unexpected()

        @property
        def ordinary(self):
            return unexpected()

    expected = {
        "label": ModelColumn("label", dtype="string"),
        "value": ModelColumn("value", dtype="Int64"),
    }
    assert (
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")],
            type_rules=[ScalarColumnRule()],
        )(Model)
        == expected
    )
    assert list(
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")],
            type_rules=[ScalarColumnRule()],
        )(Model)
    ) == list(expected)
    assert ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
        include_computed=True,
    )(Model) == expected | {"computed": ModelColumn("computed", dtype="Int64")}


def test_name_rules_run_only_during_declaration(monkeypatch):
    class Sample(Reading):
        label: str

    columns = ModelColumnMapper(
        type_rules=[ScalarColumnRule()],
        name_rules=[(r"value", r"sample_\g<0>")],
        model_path="sample.reading",
        default=None,
    )(Sample)

    def unexpected(*args, **kwargs):
        pytest.fail("Row projection invoked the column mapper")

    monkeypatch.setattr(ModelColumnMapper, "__call__", unexpected)
    monkeypatch.setattr(ScalarColumnRule, "__call__", unexpected)
    columns |= {"sample_object": ModelColumn("sample.reading", default=None)}
    model = Measurement(label="a", reading=Reading(value=3))
    table = ModelTableAccessor.from_models(
        [SimpleNamespace(sample=model), SimpleNamespace()], columns=columns
    )
    assert list(table) == ["sample_value", "sample_object"]
    assert table["sample_value"].tolist()[0] == 3
    assert table["sample_object"].iloc[0] is model.reading
    assert pd.isna(table["sample_value"].iloc[1])


@pytest.mark.parametrize(
    ("pattern", "exclude", "expected"),
    [
        (r".+", None, ["value", "label", "quality"]),
        (r"", None, []),
        (r"value|label", r"label", ["value"]),
        (r"value|quality", r"quality", ["value"]),
        (re.compile(r"VALUE", re.IGNORECASE), None, ["value"]),
        (r"al", None, []),
        (r".*al.*", None, ["value", "quality"]),
        (r".+", r"al", ["value", "label", "quality"]),
        (r".+", r".*al.*", ["label"]),
        (r".+", re.compile(r"LABEL", re.IGNORECASE), ["value", "quality"]),
        (r".+", r".+", []),
        (r".+", r"x_value", ["value", "label", "quality"]),
    ],
)
def test_name_rule_and_exclude_fullmatch_original_field_names(
    pattern, exclude, expected
):
    class Sample(BaseModel):
        value: int
        label: str
        quality: float

    mapper = ModelColumnMapper(
        type_rules=[ScalarColumnRule()],
        name_rules=[(pattern, r"x_\g<0>")],
        exclude=exclude,
    )
    assert list(mapper(Sample)) == [f"x_{name}" for name in expected]


def test_name_rule_changes_base_names_before_conversion_not_attribute_paths():
    class Sample(BaseModel):
        t_int: TimeQuantityField
        label: str

    mapper = ModelColumnMapper(
        type_rules=[
            QuantityColumnRule(physical_type="time", unit="s"),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
        model_path="metadata",
        name_rules=[(r"t_(?P<kind>.+)", r"header_\g<kind>")],
    )
    columns = mapper(Sample)
    assert list(columns) == ["header_int_s"]
    column = columns["header_int_s"]
    assert column.attr == "metadata.t_int"
    assert (
        column.read(SimpleNamespace(metadata=Sample(t_int="2 min", label="x"))) == 120
    )


def test_name_rule_is_reusable_under_different_model_paths():
    mapper = ModelColumnMapper(
        type_rules=[ScalarColumnRule()], name_rules=[(r"(.+)", r"file_\1")]
    )
    source = SimpleNamespace(
        a=SimpleNamespace(filepath_info=Reading(value=3)),
        b=SimpleNamespace(filepath_info_orig=Reading(value=4)),
    )
    columns_a = mapper(Reading, model_path="a.filepath_info")
    columns_b = mapper(Reading, model_path="b.filepath_info_orig")
    assert columns_a["file_value"].attr == "a.filepath_info.value"
    assert columns_b["file_value"].attr == "b.filepath_info_orig.value"
    assert columns_a["file_value"].read(source) == 3
    assert columns_b["file_value"].read(source) == 4
    assert mapper.model_path is None
    assert mapper(Reading)["file_value"].read(Reading(value=5)) == 5


def test_name_rule_precedes_time_and_coordinate_suffixes():
    class Sample(BaseModel):
        t0: UnixTimeField
        target: SkyCoordField

    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"header_\g<0>")],
        type_rules=[
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
        ],
    )
    columns = mapper(Sample)
    assert list(columns) == [
        "header_t0_mjd",
        "header_target_ra_deg",
        "header_target_dec_deg",
        "header_target_frame",
    ]
    assert [column.attr for column in columns.values()] == [
        "t0",
        "target",
        "target",
        "target",
    ]
    assert list(mapper(Sample, exclude=r"target")) == ["header_t0_mjd"]
    assert list(mapper(Sample, exclude=r"target_ra_deg")) == list(columns)


def test_name_rule_rejects_collisions_instead_of_losing_columns():
    class Sample(BaseModel):
        value: int
        label: str

    with pytest.raises(ValueError, match="multiple projections"):
        ModelColumnMapper(
            type_rules=[ScalarColumnRule()], name_rules=[(r".+", "same")]
        )(Sample)


def test_name_rule_rejects_collisions_after_representation_suffixes():
    class Sample(BaseModel):
        t0: UnixTimeField
        t0_mjd: float

    with pytest.raises(ValueError, match="multiple projections"):
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")],
            type_rules=[TimeColumnRule(format="mjd"), ScalarColumnRule()],
        )(Sample)


def test_mapper_validates_only_at_construction_or_replacement():
    calls = []

    class CountingMapper(ModelColumnMapper):
        @field_validator("include_computed")
        @classmethod
        def count_validation(cls, value):
            calls.append(value)
            return value

    mapper = CountingMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
        include_computed=True,
    )
    assert calls == [True]
    for _ in range(3):
        assert list(mapper(Reading)) == ["value"]
    assert calls == [True]
    renamed = mapper(Reading, name_rules=[(r".+", r"\g<0>_copy")])
    assert list(renamed) == ["value_copy"]
    assert renamed["value_copy"].default is MISSING
    assert calls == [True, True]
    assert mapper.name_rules == ((re.compile(r".+"), r"\g<0>"),)
    assert list(mapper(Reading)) == ["value"]
    assert calls == [True, True]
    with pytest.raises(ValidationError, match="frozen"):
        cast("Any", mapper).include_computed = False


def test_mapper_subclass_defaults_are_validated():
    class InvalidNamesMapper(ModelColumnMapper):
        name_rules: tuple[tuple[re.Pattern[str], str], ...] = (
            (re.compile(".+"), r"\2"),
        )

    with pytest.raises(ValidationError, match="group reference"):
        InvalidNamesMapper(
            type_rules=[ScalarColumnRule()],
        )


def test_mapper_replacement_keeps_default_objects_without_serializing_them():
    class Fallback(BaseModel):
        value: int

        @field_serializer("value")
        def unexpected(self, value):
            pytest.fail("Replacing mapper settings serialized the fallback")

    fallback = Fallback(value=42)
    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
        model_path="missing",
        default=fallback,
    )
    column = mapper(Reading, name_rules=[(r".+", r"\g<0>_copy")])["value_copy"]
    assert column.default is fallback
    assert column.read(SimpleNamespace()) == 42


def test_generated_columns_keep_missing_and_null_distinct():
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(Reading)
    table = ModelTableAccessor.from_models([Reading(value=None)], columns=columns)
    assert pd.isna(table["value"].iloc[0])
    with pytest.raises(AttributeError, match="value"):
        ModelTableAccessor.from_models([SimpleNamespace()], columns=columns)


def test_generated_columns_can_be_overridden_without_shared_state():
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(Reading)
    overridden = columns | {"value": ModelColumn("value", default=None, dtype="Int64")}
    table = ModelTableAccessor.from_models(
        [Reading(value=2), SimpleNamespace()], columns=overridden
    )
    assert str(table["value"].dtype) == "Int64"
    assert table["value"].isna().tolist() == [False, True]
    columns.clear()
    assert ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(Reading) == {"value": ModelColumn("value", dtype="Int64")}
    assert list(ModelTableAccessor.from_models([], columns=overridden)) == ["value"]


@pytest.mark.parametrize(
    "annotation", [StateCode, Literal[StateCode.ready], StateCode | int | None]
)
def test_scalar_enum_column_uses_values(annotation):
    model = create_model("StateModel", state=(annotation, ...))
    table = ModelTableAccessor.from_models(
        [model.model_validate({"state": StateCode.ready})],
        columns=ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")],
            type_rules=[ScalarColumnRule()],
        )(model),
    )
    assert table["state"].tolist() == [1]
    assert str(table["state"].dtype) == "Int64"


def test_string_enum_column_uses_value_not_member_name():
    class State(StrEnum):
        Ready = "ready"

    class Model(BaseModel):
        state: State | str | None

    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(Model)
    table = ModelTableAccessor.from_models(
        [Model(state=State.Ready), Model(state="other"), Model(state=None)],
        columns=columns,
    )
    assert table["state"].iloc[0] == "ready"
    assert table["state"].iloc[1] == "other"
    assert pd.isna(table["state"].iloc[2])


def test_scientific_columns_use_explicit_units_and_preserve_metadata():
    class Metadata(BaseModel):
        t0: UnixTimeField
        t_int: TimeQuantityField
        f_smp: FrequencyQuantityField
        target: SkyCoordField

    metadata = Metadata(
        t0=Time("2025-10-31", scale="utc").tai,
        t_int=2 * u.min,
        f_smp=2 * u.kHz,
        target=SkyCoord(ra=1 * u.rad, dec=0.5 * u.rad),
    )
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            QuantityColumnRule(physical_type="time", unit="s"),
            QuantityColumnRule(physical_type="frequency", unit="Hz"),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
    )(Metadata, default=None)
    table = ModelTableAccessor.from_models([metadata, None], columns=columns)
    assert list(table) == [
        "t0_mjd",
        "t_int_s",
        "f_smp_Hz",
        "target_ra_deg",
        "target_dec_deg",
        "target_frame",
    ]
    row = table.iloc[0]
    assert row["t0_mjd"] == metadata.t0.mjd
    assert row["t_int_s"] == 120
    assert row["f_smp_Hz"] == 2000
    assert row["target_ra_deg"] == metadata.target.ra.deg
    assert row["target_dec_deg"] == metadata.target.dec.deg
    assert row["target_frame"] == metadata.target.frame.name
    assert table.iloc[1].isna().all()
    assert metadata.t0.scale == "tai"
    assert metadata.t_int.unit == u.min
    assert metadata.f_smp.unit == u.kHz
    assert "t_int_s" not in ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(Metadata)


def test_quantity_rule_rejects_incompatible_values_and_arrays():
    model = create_model("Duration", value=(TimeQuantityField, ...))
    column = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            QuantityColumnRule(physical_type="time", unit="s"),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
    )(model)
    with pytest.raises(u.UnitConversionError):
        ModelTableAccessor.from_models(
            [SimpleNamespace(value=2 * u.Hz)], columns=column
        )
    with pytest.raises(ValueError, match="scalar"):
        ModelTableAccessor.from_models(
            [SimpleNamespace(value=[1, 2] * u.s)], columns=column
        )


@pytest.mark.parametrize(
    "annotation",
    [
        TimeQuantityField,
        TimeQuantityField | None,
        quantity_field("time"),
        cast("Any", quantity_field("time")) | None,
        Annotated[QuantityField, u.s],
    ],
)
def test_quantity_physical_types_survive_aliases_and_field_metadata(annotation):
    model = create_model("Duration", duration=(annotation, ...))
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            QuantityColumnRule(physical_type="time", unit=u.s),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
    )(model)
    assert list(columns) == ["duration_s"]
    table = ModelTableAccessor.from_models(
        [model.model_validate({"duration": "2 min"})], columns=columns
    )
    assert table["duration_s"].tolist() == [120]


@pytest.mark.parametrize(
    "annotation",
    [QuantityField, quantity_field(["time", "frequency"]), FrequencyQuantityField],
)
def test_quantity_without_matching_column_declaration_is_not_projected(annotation):
    model = create_model("Value", value=(annotation, ...))
    assert (
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")],
            type_rules=[
                QuantityColumnRule(physical_type="time", unit="s"),
                TimeColumnRule(format="mjd"),
                SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
                ScalarColumnRule(),
            ],
        )(model)
        == {}
    )
    quantity = 2 * u.kHz
    value = model.model_validate({"value": quantity})
    table = ModelTableAccessor.from_models(
        [value], columns={"value": ModelColumn("value")}
    )
    assert table["value"].iloc[0] is quantity


def test_quantity_rules_and_field_selection_compose():
    class Sample(BaseModel):
        duration: TimeQuantityField
        frequency: FrequencyQuantityField
        label: str

    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            QuantityColumnRule(physical_type="time", unit="s"),
            QuantityColumnRule(physical_type="frequency", unit="Hz"),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
    )
    assert list(
        mapper(
            Sample, name_rules=[(r"duration|frequency", r"\g<0>")], exclude=r"frequency"
        )
    ) == ["duration_s"]
    assert list(
        mapper(
            Sample,
            type_rules=[
                QuantityColumnRule(physical_type="frequency", unit="kHz"),
                TimeColumnRule(format="mjd"),
                SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
                ScalarColumnRule(),
            ],
        )
    ) == [
        "frequency_kHz",
        "label",
    ]
    assert (
        list(
            mapper(
                Sample,
                type_rules=[
                    TimeColumnRule(format="mjd"),
                    SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
                    ScalarColumnRule(),
                ],
                name_rules=[(r"duration", "duration")],
            )
        )
        == []
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"time_format": "mjd"}, "Extra inputs"),
        ({"time_scale": "utc"}, "Extra inputs"),
        ({"include_quantity_types": {"time": "s"}}, "Extra inputs"),
        ({"coordinate_unit": "deg"}, "Extra inputs"),
        ({"name_rule": (r".+", r"\g<0>")}, "Extra inputs"),
        ({"type_rules": [object()]}, "callable"),
        ({"type_rules": None}, "tuple"),
        ({"include": "value"}, "Extra inputs"),
        ({"rename": (r".+", r"\g<0>")}, "Extra inputs"),
        ({"attr_prefix": "metadata."}, "Extra inputs"),
        ({"name_rules": [(r"[", "prefix_")]}, "pattern"),
        ({"exclude": "["}, "pattern"),
        ({"exclude": {"value"}}, "pattern"),
        ({"name_rules": [(r".+", r"\2")]}, "group reference"),
        ({"name_rules": [(r".+", r"\g<missing>")]}, "unknown group"),
        ({"model_path": ""}, "pattern"),
        ({"model_path": "metadata."}, "pattern"),
        ({"model_path": ".metadata"}, "pattern"),
        ({"model_path": "a..metadata"}, "pattern"),
    ],
)
def test_invalid_mapper_settings_fail_at_construction_and_replacement(options, message):
    settings = {
        "name_rules": [(r".+", r"\g<0>")],
        "type_rules": [ScalarColumnRule()],
    } | options
    with pytest.raises(ValidationError, match=message):
        ModelColumnMapper(**settings)
    with pytest.raises(ValidationError, match=message):
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")],
            type_rules=[ScalarColumnRule()],
        )(Reading, **options)


@pytest.mark.parametrize("time_format", ["mjd", "jd", "unix", "isot", "datetime"])
@pytest.mark.parametrize("scale", ["utc", "tai"])
def test_time_format_preserves_native_scale_without_mutating_metadata(
    time_format, scale
):
    class Timestamp(BaseModel):
        t0: UnixTimeField

    value = Time("2025-10-31", scale=scale)
    model = Timestamp(t0=value)
    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            TimeColumnRule(format=time_format),
            ScalarColumnRule(),
        ],
    )
    columns = mapper(Timestamp)
    table = ModelTableAccessor.from_models([model], columns=columns)
    assert list(table) == [f"t0_{time_format}"]
    assert table.iloc[0, 0] == value.to_value(time_format)
    assert model.t0 is value
    assert value.scale == scale
    with pytest.raises(ValueError, match="scalar"):
        ModelTableAccessor.from_models(
            [SimpleNamespace(t0=Time([1, 2], format="mjd"))], columns=columns
        )


def test_coordinate_unit_is_configurable_without_changing_frame_or_units():
    class Position(BaseModel):
        target: SkyCoordField

    value = SkyCoord(ra=30 * u.deg, dec=20 * u.deg, frame="fk5")
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            SkyCoordColumnRule(attrs=["ra", "dec"], unit=u.rad),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
    )(Position)
    table = ModelTableAccessor.from_models([Position(target=value)], columns=columns)
    assert list(table) == ["target_ra_rad", "target_dec_rad", "target_frame"]
    assert table["target_ra_rad"].iloc[0] == value.ra.rad
    assert table["target_dec_rad"].iloc[0] == value.dec.rad
    assert table["target_frame"].iloc[0] == "fk5"
    assert value.frame.name == "fk5"
    assert value.ra.unit == u.deg


def test_mapper_overrides_are_isolated_from_caller_settings_and_previous_columns():
    class Duration(BaseModel):
        value: TimeQuantityField

    rule = QuantityColumnRule(physical_type="time", unit="s")
    rules = [rule]
    mapper = ModelColumnMapper(name_rules=[(r".+", r"\g<0>")], type_rules=rules)
    columns_seconds = mapper(Duration)
    rules[0] = QuantityColumnRule(physical_type="time", unit="min")
    assert mapper.type_rules == (rule,)
    assert rule.unit == u.s
    columns_minutes = mapper(Duration, type_rules=rules)
    value = Duration(value="2 min")
    assert columns_seconds["value_s"].read(value) == 120
    assert columns_minutes["value_min"].read(value) == 2
    assert mapper(Duration)["value_s"].read(value) == 120
    empty = ModelTableAccessor.from_models([], columns=columns_seconds)
    assert str(empty["value_s"].dtype) == "Float64"


def test_name_rules_select_computed_fields_and_preserve_annotation_metadata():
    class Duration(BaseModel):
        value: Annotated[QuantityField, u.get_physical_type("time")]

        @computed_field
        @property
        def computed(self) -> TimeQuantityField:
            pytest.fail("Computed value evaluated during column declaration")

    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        include_computed=True,
        type_rules=[
            QuantityColumnRule(physical_type="time", unit="s"),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
    )
    assert list(mapper(Duration)) == ["value_s", "computed_s"]
    assert list(mapper(Duration, exclude=r"computed")) == ["value_s"]
    assert list(mapper(Duration, name_rules=[(r"computed", "elapsed")])) == [
        "elapsed_s"
    ]


def test_skycoord_columns_do_not_silently_replace_an_unsupported_frame_with_nulls():
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg")],
    )(create_model("Position", target=(SkyCoordField, ...)), default=None)
    galactic = SkyCoord(l=10 * u.deg, b=20 * u.deg, frame="galactic")
    with pytest.raises(AttributeError, match="ra"):
        ModelTableAccessor.from_models(
            [SimpleNamespace(target=galactic)], columns=columns
        )


def test_column_converter_skips_none_and_preserves_lookup_errors():
    calls = []

    def convert(value):
        calls.append(value)
        return value * 2

    columns = {"value": ModelColumn("value", default=None, converter=convert)}
    table = ModelTableAccessor.from_models(
        [Reading(value=3), Reading(value=None), SimpleNamespace()], columns=columns
    )
    assert table["value"].tolist() == [6, None, None]
    assert calls == [3]


def test_filepath_columns_include_concrete_and_inherited_fields():
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(
        ToltecFilepathInfo,
        name_rules=[(r".+", r"file_\g<0>")],
        default=None,
    )
    assert set(columns) == {f"file_{name}" for name in ToltecFilepathInfo.model_fields}
    assert {
        name: ToltecFileAccessor._filepath_info_columns[name] for name in columns
    } == columns
    assert all(
        name == f"file_{column.attr.split('.')[-1]}" for name, column in columns.items()
    )
    assert all(column.default is None for column in columns.values())
    assert "file_obsnum" in columns
    assert "file_roach" in columns
    assert "file_proc_suffix" in columns


def test_metadata_columns_are_declared_once_without_reading_counts(
    monkeypatch,
):
    columns = ToltecFileAccessor._metadata_columns
    assert columns["n_blocks"].default is None
    assert columns["block"].default is None
    assert "filepath_info_orig" not in columns
    assert "adc_snap" not in columns

    def unexpected(*args, **kwargs):
        pytest.fail("Repeated field inspection")

    monkeypatch.setattr(ModelColumnMapper, "__call__", unexpected)
    table = ModelTableAccessor.from_models(
        [SimpleNamespace(obsnum=145609)], columns=columns
    )
    assert table["obsnum"].tolist() == [145609]
    assert pd.isna(table["n_blocks"].iloc[0])
    assert ToltecFileAccessor._metadata_columns is columns


def test_filepath_columns_are_declared_once(monkeypatch):
    columns = ToltecFileAccessor._filepath_info_columns
    assert columns["file_obsnum"].attr == "obsnum"
    assert columns["obsnum"].attr == "obsnum"

    def unexpected(*args, **kwargs):
        pytest.fail("Repeated filename field inspection")

    monkeypatch.setattr(ModelColumnMapper, "__call__", unexpected)
    table = ToltecFileAccessor.from_sources(["tel_toltec_2025-10-31_145609_00_0002.nc"])
    assert table["file_obsnum"].tolist() == [145609]
    assert table["obsnum"].tolist() == [145609]
    assert ToltecFileAccessor._filepath_info_columns is columns


def test_file_and_current_identifiers_remain_separate():
    table = ToltecFileAccessor.from_sources(["tel_toltec_2025-10-31_145609_00_0002.nc"])
    table["obsnum"] = 1
    assert table["obsnum"].tolist() == [1]
    assert table["file_obsnum"].tolist() == [145609]
    assert "filepath_info" not in table


def test_model_column_is_a_small_declaration():
    column = ModelColumn("reading.value")
    assert column.attr == "reading.value"
    assert column.default is MISSING
    assert column.dtype is None


def test_flat_projection_of_mixed_models_preserves_order_and_nulls():
    models = [
        Measurement(label="a", reading=Reading(value=2**60 + 1)),
        Measurement(label="b", reading=Reading(value=None)),
        Measurement(label="c", reading=None),
        Reading(value=4),
    ]
    columns = {
        "sample": ModelColumn("label", default="unknown"),
        "value": ModelColumn("reading.value", default=None, dtype="Int64"),
    }
    table = ModelTableAccessor.from_models(iter(models), columns=columns)
    assert list(table) == ["sample", "value"]
    assert table["sample"].tolist() == ["a", "b", "c", "unknown"]
    assert table["sample"].dtype == object
    assert int(table["value"].iloc[0]) == 2**60 + 1
    assert table["value"].isna().tolist() == [False, True, True, True]
    assert str(table["value"].dtype) == "Int64"
    assert "reading" not in table
    assert "_record" not in table
    first = models[0]
    assert isinstance(first, Measurement)
    assert first.reading is not None
    assert first.reading.value == 2**60 + 1


def test_projection_reads_attributes_without_model_dump_or_serializers(monkeypatch):
    calls = []

    class Model(BaseModel):
        value: int = Field(exclude=True)

        @field_serializer("value")
        def serialize_value(self, value):
            pytest.fail("Attribute projection called a field serializer")

        @computed_field
        @property
        def selected(self) -> int:
            calls.append("selected")
            return self.value * 2

        @computed_field
        @property
        def unrelated(self) -> int:
            pytest.fail("Unselected property was read")

    def unexpected(*args, **kwargs):
        pytest.fail("Projection dumped the whole model")

    monkeypatch.setattr(Model, "model_dump", unexpected)
    table = ModelTableAccessor.from_models(
        [Model(value=3)],
        columns={"raw": ModelColumn("value"), "double": ModelColumn("selected")},
    )
    assert table.to_dict("records") == [{"raw": 3, "double": 6}]
    assert calls == ["selected"]
    assert ModelTableAccessor(table).query("double > 0").equals(table)
    assert calls == ["selected"]


def test_default_is_only_for_missing_attributes_and_follows_rgetattr():
    columns = {"x": ModelColumn("reading.value", default=9)}
    models = [Measurement(label="a", reading=Reading(value=None)), Reading(value=2)]
    table = ModelTableAccessor.from_models(models, columns=columns)
    assert table["x"].tolist() == [None, 9]
    fallback = SimpleNamespace(value=42)
    table = ModelTableAccessor.from_models(
        [Reading(value=2)],
        columns={"x": ModelColumn("reading.value", default=fallback)},
    )
    assert table["x"].tolist() == [42]


def test_required_missing_attribute_raises():
    with pytest.raises(AttributeError, match="absent"):
        ModelTableAccessor.from_models(
            [Reading(value=1)], columns={"required": ModelColumn("absent")}
        )


def test_present_property_errors_follow_rgetattr():
    class Model(BaseModel):
        @property
        def broken(self):
            msg = "read failed"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="read failed"):
        ModelTableAccessor.from_models(
            [Model()], columns={"x": ModelColumn("broken", default=None)}
        )


@pytest.mark.parametrize("models", [[], [Reading(value=1), Reading(value=2)]])
def test_empty_declaration_preserves_row_count(models):
    table = ModelTableAccessor.from_models(iter(models), columns={})
    assert table.shape == (len(models), 0)


def test_empty_input_retains_declared_columns_and_types():
    table = ModelTableAccessor.from_models(
        [],
        columns={
            "value": ModelColumn("reading.value", dtype="Int64"),
            "label": ModelColumn("label"),
        },
    )
    assert table.shape == (0, 2)
    assert table["value"].dtype == pd.Int64Dtype()
    assert table["label"].dtype == object


def test_arrow_dtype_is_an_explicit_column_choice():
    class Model(BaseModel):
        value: int | None
        when: datetime

    when = datetime(2026, 9, 26, tzinfo=UTC)
    table = ModelTableAccessor.from_models(
        [Model(value=2**60 + 1, when=when), Model(value=None, when=when)],
        columns={
            "value": ModelColumn("value", dtype=pd.ArrowDtype(pa.int64())),
            "when": ModelColumn("when", dtype=pd.ArrowDtype(pa.timestamp("us", "UTC"))),
        },
    )
    assert table["value"].dtype == pd.ArrowDtype(pa.int64())
    assert table["value"].array.__arrow_array__().to_pylist() == [2**60 + 1, None]
    assert table["when"].iloc[0] == when


def test_object_columns_preserve_native_enum_identity():
    class State(IntEnum):
        ready = 1

    class Model(BaseModel):
        state: State

    table = ModelTableAccessor.from_models(
        [Model(state=State.ready)], columns={"state": ModelColumn("state")}
    )
    assert table["state"].iloc[0] is State.ready


@pytest.mark.parametrize("options", [{"mode": "json"}, {"round_trip": True}])
@pytest.mark.parametrize("factory", [ModelTableAccessor, ToltecFileAccessor])
def test_projection_rejects_dump_options_before_consuming_models(factory, options):
    def models():
        pytest.fail("Invalid options consumed the input")
        yield

    with pytest.raises(TypeError, match="unexpected keyword"):
        factory.from_models(models(), columns={}, **options)


def test_file_accessor_delegates_projection_without_schema_inference(monkeypatch):
    # Recorded acquisition name; no file is opened for this projection.
    info = FilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert info is not None

    def unexpected(*args, **kwargs):
        pytest.fail("Projection inspected a filepath schema or dumped a model")

    assert not hasattr(type(info), "_get_filepath_info_schema")
    monkeypatch.setattr(type(info), "model_dump", unexpected)
    columns = {
        "file_suffix": ModelColumn("suffix", default=None, dtype="string"),
        "obsnum": ModelColumn("obsnum", dtype="Int64"),
    }
    table = ToltecFileAccessor.from_models([info], columns=columns)
    assert list(table) == list(columns)
    assert table["obsnum"].tolist() == [145609]
    assert ModelTableAccessor(table).query("obsnum == 145609").equals(table)


def test_subclass_can_keep_projection_as_a_plain_class_declaration():
    class Readings(ModelTableAccessor):
        columns: ClassVar = {"value": ModelColumn("value", dtype="Int64")}

    table = Readings.from_models([Reading(value=2)], columns=Readings.columns)
    assert table["value"].tolist() == [2]


def test_toltec_projection_does_not_convert_undeclared_values():
    info = get_info_from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert info is not None
    table = ToltecFileAccessor.from_models(
        [info], columns={"data_kind": ModelColumn("data_kind")}
    )
    assert table["data_kind"].dtype == object
    assert table["data_kind"].iloc[0] is info.data_kind


def test_toltec_projection_rejects_implicit_header_columns():
    with pytest.raises(TypeError, match="metadata"):
        ToltecFileAccessor.from_models([], columns={}, metadata=[])  # ty: ignore[unknown-argument]


def test_reference_filename_projection_header_enrichment_and_open(data_lmt_path):
    paths = [
        path
        for directory in ("tel", "toltec")
        for path in sorted((data_lmt_path / directory).rglob("*.nc"))
    ]
    assert len(paths) == 11
    filepath_models = []
    for path in paths:
        info = FilepathInfo.from_source(path.name)
        assert isinstance(info, LmtFilepathInfo | ToltecFilepathInfo)
        filepath_models.append(info)
    table = ModelTableAccessor.from_models(
        filepath_models,
        columns={
            "master": ModelColumn("master", default=None, dtype="string"),
            "obsnum": ModelColumn("obsnum", dtype="Int64"),
            "file_suffix": ModelColumn("suffix", default=None, dtype="string"),
            "roach": ModelColumn("roach", default=None, dtype="Int64"),
        },
    )
    table["source"] = paths
    assert table["master"].isna().any()
    metadata = []
    for path, filepath_info in zip(paths, filepath_models, strict=True):
        with xr.open_dataset(path) as data:
            metadata.append(
                LmtTelAccessor(data).meta
                if filepath_info.instru == "lmt"
                else ToltecKidsAccessor(data).meta
            )
    header = ModelTableAccessor.from_models(
        metadata,
        columns={"master": ModelColumn("master", dtype="string")},
    )
    table["master"] = header["master"]
    selected = ModelTableAccessor(table).query(
        'master == "tcs" & file_suffix in ["vnasweep", "targsweep", "tune"]'
    )
    assert len(selected) == 3
    for (source,) in selected[["source"]].itertuples(index=False, name=None):
        with xr.open_dataset(source) as data:
            assert ToltecKidsAccessor(data).meta.master == "tcs"
