"""Ordered column rules select once and leave row reads independent."""

from types import SimpleNamespace
from typing import Any, cast

import astropy.units as u
import pytest
from astropy.coordinates import ICRS, SkyCoord
from astropy.time import Time
from pydantic import BaseModel, ValidationError, field_validator
from tollan.config.types import SkyCoordField, TimeQuantityField, UnixTimeField

from tolteca_datamodels.model_table import (
    ModelColumn,
    ModelColumnMapper,
    ModelTableAccessor,
    QuantityColumnRule,
    ScalarColumnRule,
    SkyCoordColumnRule,
    TimeColumnRule,
)


class Reading(BaseModel):
    value: int
    label: str


@pytest.mark.parametrize(
    ("options", "missing"),
    [
        ({}, {"name_rules", "type_rules"}),
        ({"name_rules": [(r".+", r"\g<0>")]}, {"type_rules"}),
        ({"type_rules": [ScalarColumnRule()]}, {"name_rules"}),
    ],
)
def test_both_rule_lists_are_required(options, missing):
    with pytest.raises(ValidationError) as exc:
        ModelColumnMapper(**options)
    assert {(error["loc"][0], error["type"]) for error in exc.value.errors()} == {
        (name, "missing") for name in missing
    }


def test_name_rules_collect_all_matches_in_order_without_chaining():
    rules = [(r"value", "special"), (r"special", "again"), (r".+", r"all_\g<0>")]
    mapper = ModelColumnMapper(type_rules=[ScalarColumnRule()], name_rules=rules)
    rules.clear()
    columns = mapper(Reading)
    assert list(columns) == ["special", "all_value", "all_label"]
    assert [column.attr for column in columns.values()] == ["value", "value", "label"]
    assert list(mapper(Reading, name_rules=list(reversed(mapper.name_rules)))) == [
        "all_value",
        "special",
        "all_label",
    ]


def test_name_aliases_are_deduplicated_before_type_rules_run():
    class Stamp(BaseModel):
        ut: UnixTimeField

    calls = []
    rule = TimeColumnRule(format="mjd")

    def trace(name, annotation, column_name):
        calls.append(column_name)
        return rule(name, annotation, column_name)

    columns = ModelColumnMapper(
        name_rules=[
            (r"ut", r"file_\g<0>"),
            (r".+", r"\g<0>"),
            (r"u(t)", r"file_u\1"),
            (r".+", r"\g<0>"),
        ],
        type_rules=[trace],
    )(Stamp)
    assert calls == ["file_ut", "ut"]
    assert list(columns) == ["file_ut_mjd", "ut_mjd"]
    model = Stamp(ut=Time(60000, format="mjd"))
    assert all(column.attr == "ut" for column in columns.values())
    assert [column.read(model) for column in columns.values()] == [60000, 60000]


def test_each_name_alias_selects_its_type_rule_independently():
    def file_only(name, annotation, column_name):
        if column_name.startswith("file_"):
            return {column_name: ModelColumn(name, dtype=object)}
        return None

    columns = ModelColumnMapper(
        name_rules=[(r"value", r"file_\g<0>"), (r"value", r"\g<0>")],
        type_rules=[file_only, ScalarColumnRule()],
    )(Reading)
    assert list(columns) == ["file_value", "value"]
    assert columns["file_value"].dtype is object
    assert columns["value"].dtype == "Int64"


def test_type_expansion_cannot_merge_distinct_name_aliases_silently():
    def collapse(name, annotation, column_name):
        return {"same": ModelColumn(name)}

    with pytest.raises(ValueError, match="multiple projections"):
        ModelColumnMapper(
            name_rules=[(r"value", "first"), (r"value", "second")],
            type_rules=[collapse],
        )(Reading)


def test_exclusion_and_name_selection_precede_type_rules():
    calls = []

    def trace(name, annotation, column_name):
        calls.append((name, annotation, column_name))
        return {column_name: ModelColumn(name)}

    mapper = ModelColumnMapper(name_rules=[(r"value", "renamed")], type_rules=[trace])
    assert list(mapper(Reading, exclude="value")) == []
    assert calls == []
    assert list(mapper(Reading, exclude="renamed")) == ["renamed"]
    assert calls == [("value", int, "renamed")]


def test_rule_expansion_uses_renamed_base_but_reads_original_attribute():
    class Position(BaseModel):
        target: SkyCoordField

    mapper = ModelColumnMapper(
        name_rules=[(r"target", "myobj_target"), (r"myobj_.+", "wrong")],
        type_rules=[
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
        model_path="metadata",
    )
    columns = mapper(Position)
    assert list(columns) == [
        "myobj_target_ra_deg",
        "myobj_target_dec_deg",
        "myobj_target_frame",
    ]
    assert [column.attr for column in columns.values()] == [
        "metadata.target",
        "metadata.target",
        "metadata.target",
    ]

    class Object(BaseModel):
        metadata: Position

    model = Object(metadata=Position(target=SkyCoord(ra=30 * u.deg, dec=20 * u.deg)))
    table = ModelTableAccessor.from_models([model], columns=columns)
    assert table.iloc[0].tolist() == [30, 20, "icrs"]


def test_first_type_match_wins_and_unmatched_types_are_skipped():
    calls = []

    def integers(name, annotation, column_name):
        calls.append(name)
        if annotation is not int:
            return None
        return {f"{column_name}_int": ModelColumn(name, dtype="Int64")}

    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")], type_rules=[integers, ScalarColumnRule()]
    )(Reading)
    assert list(columns) == ["value_int", "label"]
    assert calls == ["value", "label"]
    assert list(
        ModelColumnMapper(name_rules=[(r".+", r"\g<0>")], type_rules=[integers])(
            Reading
        )
    ) == ["value_int"]


@pytest.mark.parametrize("option", ["name_rules", "type_rules"])
def test_empty_rule_lists_emit_no_columns(option):
    options: dict[str, Any] = {
        "name_rules": [(r".+", r"\g<0>")],
        "type_rules": [ScalarColumnRule()],
        option: [],
    }
    assert ModelColumnMapper(**options)(Reading) == {}


def test_empty_mapping_is_a_match_not_a_fallback():
    def suppress(name, annotation, column_name):
        return {}

    def unexpected(*args):
        pytest.fail("Another type rule ran after a matching rule")

    assert (
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")], type_rules=[suppress, unexpected]
        )(Reading)
        == {}
    )


def test_rule_and_converter_errors_do_not_fall_through():
    def fail(*args):
        msg = "deliberate error"
        raise ValueError(msg)

    def unexpected(*args):
        pytest.fail("A later rule ran after an error or match")

    with pytest.raises(ValueError, match="deliberate error"):
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")], type_rules=[fail, unexpected]
        )(Reading)

    def rule(name, annotation, column_name):
        return {column_name: ModelColumn(name, converter=fail)}

    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")], type_rules=[rule, unexpected]
    )(Reading)
    with pytest.raises(ValueError, match="deliberate error"):
        ModelTableAccessor.from_models([Reading(value=1, label="x")], columns=columns)


def test_custom_rule_can_expand_and_rename_without_being_a_model(monkeypatch):
    class Pair(BaseModel):
        bounds: tuple[int, int]

    def expand(name, annotation, column_name):
        assert annotation == tuple[int, int]
        return {
            f"{column_name}_lo": ModelColumn(name, converter=lambda value: value[0]),
            f"{column_name}_hi": ModelColumn(name, converter=lambda value: value[1]),
        }

    columns = ModelColumnMapper(
        name_rules=[(r".+", r"range_\g<0>")], type_rules=[expand], default=None
    )(Pair)

    def unexpected(*args, **kwargs):
        pytest.fail("Row projection tried to regenerate columns")

    monkeypatch.setattr(ModelColumnMapper, "__call__", unexpected)
    table = ModelTableAccessor.from_models([Pair(bounds=(1, 5)), None], columns=columns)
    assert list(table) == ["range_bounds_lo", "range_bounds_hi"]
    assert table.iloc[0].tolist() == [1, 5]
    assert table.iloc[1].isna().all()


@pytest.mark.parametrize(
    ("rule_cls", "options", "message"),
    [
        (TimeColumnRule, {"format": "invalid"}, "time format"),
        (TimeColumnRule, {"format": "mjd", "scale": "utc"}, "Extra inputs"),
        (
            SkyCoordColumnRule,
            {"attrs": ["ra", "dec"], "unit": "s"},
            "not an angle",
        ),
        (SkyCoordColumnRule, {"attrs": [], "unit": "deg"}, "at least 1"),
        (SkyCoordColumnRule, {"attrs": [""], "unit": "deg"}, "at least 1"),
        (SkyCoordColumnRule, {"attrs": ["ra", "ra"], "unit": "deg"}, "unique"),
        (QuantityColumnRule, {"physical_type": "time", "unit": "Hz"}, "physical type"),
        (
            QuantityColumnRule,
            {"physical_type": "unknown type", "unit": "s"},
            "physical type",
        ),
        (QuantityColumnRule, {"physical_type": "time", "unit": "invalid unit"}, "unit"),
    ],
)
def test_rule_settings_are_validated_when_constructed(rule_cls, options, message):
    with pytest.raises(ValidationError, match=message):
        rule_cls(**options)


@pytest.mark.parametrize(
    ("rule_cls", "options"),
    [
        (TimeColumnRule, {}),
        (SkyCoordColumnRule, {"attrs": ["ra", "dec"]}),
        (SkyCoordColumnRule, {"unit": "deg"}),
    ],
)
def test_rule_representation_settings_are_required(rule_cls, options):
    with pytest.raises(ValidationError, match="Field required"):
        rule_cls(**options)


@pytest.mark.parametrize(
    "rules",
    [
        [ScalarColumnRule(), ScalarColumnRule()],
        [TimeColumnRule(format="mjd"), TimeColumnRule(format="mjd")],
        [
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            SkyCoordColumnRule(attrs=("ra", "dec"), unit=u.deg),
        ],
        [
            QuantityColumnRule(physical_type="time", unit="s"),
            QuantityColumnRule(physical_type=u.s.physical_type, unit=u.s),
        ],
    ],
)
def test_equal_type_rules_fail_at_construction_and_replacement(rules):
    with pytest.raises(
        ValidationError, match="Duplicate type rules at indices 0 and 1"
    ):
        ModelColumnMapper(name_rules=[(r".+", r"\g<0>")], type_rules=rules)
    with pytest.raises(ValidationError, match="Duplicate type rules"):
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")],
            type_rules=[ScalarColumnRule()],
        )(Reading, type_rules=rules)


def test_duplicate_callable_identity_is_rejected():
    def scalar(name, annotation, column_name):
        return {column_name: ModelColumn(name)}

    for rule in (scalar, ScalarColumnRule()):
        with pytest.raises(ValidationError, match="Duplicate type rules"):
            ModelColumnMapper(name_rules=[(r".+", r"\g<0>")], type_rules=[rule, rule])


def test_duplicate_rules_in_subclass_defaults_are_rejected():
    class DuplicateMapper(ModelColumnMapper):
        type_rules: tuple[Any, ...] = (ScalarColumnRule(), ScalarColumnRule())

    with pytest.raises(ValidationError, match="Duplicate type rules"):
        DuplicateMapper(
            name_rules=[(r".+", r"\g<0>")],
        )


@pytest.mark.parametrize(
    ("frame", "attrs"),
    [
        ("icrs", ["ra", "dec"]),
        ("fk5", ["ra", "dec"]),
        ("altaz", ["az", "alt"]),
        ("galactic", ["l", "b"]),
    ],
)
def test_coordinate_attributes_follow_explicit_selection_without_frame_conversion(
    frame, attrs, monkeypatch
):
    class Position(BaseModel):
        target: SkyCoordField

    value = SkyCoord(10 * u.deg, 20 * u.deg, frame=frame)
    original_frame = value.frame
    metadata = Position(target=value)
    rule = SkyCoordColumnRule(attrs=attrs, unit="rad")
    columns = ModelColumnMapper(
        name_rules=[(r"target", "selected")], type_rules=[rule]
    )(Position)
    expected = [getattr(value, attr).to_value(u.rad) for attr in attrs]

    def unexpected(*args, **kwargs):
        pytest.fail("Projection must not transform the coordinate frame")

    monkeypatch.setattr(SkyCoord, "transform_to", unexpected)
    table = ModelTableAccessor.from_models([metadata], columns=columns)
    assert list(table) == [f"selected_{attr}_rad" for attr in attrs] + [
        "selected_frame"
    ]
    assert table.iloc[0].tolist() == [*expected, frame]
    assert table.selected_frame.dtype == "string"
    assert all(column.attr == "target" for column in columns.values())
    assert value.frame is original_frame
    assert metadata.target is value


def test_coordinate_attribute_order_and_subset_are_retained():
    class Position(BaseModel):
        target: SkyCoordField

    attrs = ["b", "l"]
    rule = SkyCoordColumnRule(attrs=attrs, unit="deg")
    attrs.clear()
    assert rule.attrs == ("b", "l")
    assert list(
        ModelColumnMapper(name_rules=[(r".+", r"\g<0>")], type_rules=[rule])(Position)
    ) == [
        "target_b_deg",
        "target_l_deg",
        "target_frame",
    ]
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[SkyCoordColumnRule(attrs=["alt"], unit="deg")],
    )(Position)
    value = SkyCoord(az=10 * u.deg, alt=20 * u.deg, frame="altaz")
    assert list(columns) == ["target_alt_deg", "target_frame"]
    assert columns["target_alt_deg"].read(Position(target=value)) == 20
    with pytest.raises(ValueError, match="scalar"):
        columns["target_alt_deg"].read(SimpleNamespace(target=SkyCoord([value, value])))


def test_coordinate_frames_remain_row_values_including_null_and_custom_frames():
    class LocalFrame(ICRS):
        name = "local_instrument"

    class Position(BaseModel):
        target: SkyCoordField | None

    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg")],
    )
    models = [
        Position(target=SkyCoord(10 * u.deg, 20 * u.deg, frame=frame))
        for frame in ("icrs", "fk5", LocalFrame)
    ] + [Position(target=None)]
    table = ModelTableAccessor.from_models(models, columns=mapper(Position))
    assert table.target_frame.iloc[:3].tolist() == ["icrs", "fk5", "local_instrument"]
    assert table.iloc[3].isna().all()
    assert ModelTableAccessor(table).query('target_frame == "icrs"').index.tolist() == [
        0
    ]


def test_coordinate_frame_column_collision_is_rejected():
    class Position(BaseModel):
        target: SkyCoordField
        target_frame: str

    with pytest.raises(ValueError, match="target_frame.*multiple projections"):
        ModelColumnMapper(
            name_rules=[(r".+", r"\g<0>")],
            type_rules=[
                SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
                ScalarColumnRule(),
            ],
        )(Position)


def test_rule_subclass_defaults_are_validated():
    class InvalidTimeRule(TimeColumnRule):
        format: str = "invalid"

    with pytest.raises(ValidationError, match="time format"):
        InvalidTimeRule()


def test_rule_validation_does_not_repeat_during_mapping_or_row_reads():
    calls = []

    class CountedRule(TimeColumnRule):
        @field_validator("format")
        @classmethod
        def count(cls, value):
            calls.append(value)
            return value

    class Timestamp(BaseModel):
        time: UnixTimeField

    rule = CountedRule(format="mjd")
    mapper = ModelColumnMapper(name_rules=[(r".+", r"\g<0>")], type_rules=[rule])
    for _ in range(3):
        columns = mapper(Timestamp)
        ModelTableAccessor.from_models(
            [Timestamp(time=Time(50000, format="mjd"))], columns=columns
        )
    mapper(Timestamp, name_rules=[(r".+", "other")])
    assert calls == ["mjd"]
    with pytest.raises(ValidationError, match="frozen"):
        cast("Any", rule).format = "jd"


def test_first_configured_type_rule_wins_without_duplicate_outputs():
    class Sample(BaseModel):
        time: UnixTimeField
        duration: TimeQuantityField

    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            TimeColumnRule(format="jd"),
            QuantityColumnRule(physical_type="time", unit="min"),
            QuantityColumnRule(physical_type="time", unit="s"),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
    )
    assert list(mapper(Sample)) == ["time_jd", "duration_min"]


def test_scientific_rules_do_not_match_only_one_arm_of_a_union():
    class Mixed(BaseModel):
        time: UnixTimeField | str
        coordinate: SkyCoordField | int
        quantity: TimeQuantityField | float

    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[
            QuantityColumnRule(physical_type="time", unit="s"),
            TimeColumnRule(format="mjd"),
            SkyCoordColumnRule(attrs=["ra", "dec"], unit="deg"),
            ScalarColumnRule(),
        ],
    )
    assert mapper(Mixed) == {}


def test_rule_expansion_collisions_across_fields_raise():
    def same_columns(name, annotation, column_name):
        return {"same": ModelColumn(name)}

    with pytest.raises(ValueError, match="multiple projections"):
        ModelColumnMapper(name_rules=[(r".+", r"\g<0>")], type_rules=[same_columns])(
            Reading
        )
