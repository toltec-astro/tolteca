"""Shared selector grammar, normalization, and metadata selection."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from tolteca_datamodels.core.obs_spec import FieldSelection, ValueBounds
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec


@pytest.mark.parametrize(
    ("text", "constraints"),
    [
        (123456, {"obsnum": 123456}),
        (
            "ics-018596-32-0",
            {"master": "ics", "obsnum": 18596, "subobsnum": 32, "scannum": 0},
        ),
        ("tcs-1000", {"master": "tcs", "obsnum": 1000}),
        ("tcs-1", {"master": "tcs", "obsnum": 1}),
        (
            "tcs-1-1-1",
            {"master": "tcs", "obsnum": 1, "subobsnum": 1, "scannum": 1},
        ),
        ("1", {"obsnum": 1}),
        ("1/2", {"obsnum": 1, "interface": 2}),
        ("*-1", {"subobsnum": 1}),
        ("tcs-*-1", {"master": "tcs", "subobsnum": 1}),
        ("tcs-*", {"master": "tcs"}),
        ("tcs-*/tel", {"master": "tcs", "interface": "tel"}),
        (
            "tcs-[-1]/(1,5)",
            {
                "master": "tcs",
                "obsnum": FieldSelection(None, slice(-1, None)),
                "interface": ValueBounds(1, 5),
            },
        ),
        (
            "clip-1000-2-3-7",
            {
                "master": "clip",
                "obsnum": 1000,
                "subobsnum": 2,
                "scannum": 3,
                "interface": 7,
            },
        ),
        ("1000/7", {"obsnum": 1000, "interface": 7}),
        ("1000-2/7", {"obsnum": 1000, "subobsnum": 2, "interface": 7}),
        ("1000/2/7", {"obsnum": 1000, "scannum": 2, "interface": 7}),
        ("1000/1/2/7", {"obsnum": 1000, "subobsnum": 1, "scannum": 2, "interface": 7}),
        ("1000/toltec7", {"obsnum": 1000, "interface": "toltec7"}),
        ("1000/toltec13", {"obsnum": 1000, "interface": "toltec13"}),
        ("1000/13", {"obsnum": 1000, "interface": 13}),
        ("simu-1000", {"master": "simu", "obsnum": 1000}),
        ("1000/hwpr", {"obsnum": 1000, "interface": "hwpr"}),
        ("1000-{2,0,2}", {"obsnum": 1000, "subobsnum": (0, 2)}),
        ("1000-(0,10)", {"obsnum": 1000, "subobsnum": ValueBounds(0, 10)}),
        ("1000-(5,)", {"obsnum": 1000, "subobsnum": ValueBounds(5)}),
        ("1000-(,5)", {"obsnum": 1000, "subobsnum": ValueBounds(None, 5)}),
        (
            "1000-2-3/[-1]",
            {
                "obsnum": 1000,
                "subobsnum": 2,
                "scannum": 3,
                "interface": FieldSelection(None, slice(-1, None)),
            },
        ),
        (
            "{0,1000}/(0,13)[::2]",
            {
                "obsnum": (0, 1000),
                "interface": FieldSelection(ValueBounds(0, 13), slice(None, None, 2)),
            },
        ),
        ("*", {}),
        ("[]", {}),
        ("{}", {}),
        ("[:]", {}),
        ("1000-[]-[]-1", {"obsnum": 1000, "interface": 1}),
    ],
)
def test_parse_and_roundtrip(text, constraints):
    spec = ObsSpec.parse(text)
    assert dict(spec.constraints()) == constraints
    assert ObsSpec.parse(str(spec)) == spec
    assert ObsSpec.parse(spec) is spec
    assert hash(ObsSpec.parse(str(spec))) == hash(spec)


@pytest.mark.parametrize(
    ("data", "text"),
    [
        ({}, "*"),
        ({"master": "tcs"}, "tcs-*"),
        ({"obsnum": 1000}, "1000"),
        ({"obsnum": 1000, "interface": 7}, "1000/7"),
        ({"obsnum": 1000, "scannum": 2, "interface": 7}, "1000/2/7"),
        ({"obsnum": 1000, "subobsnum": 2, "interface": 7}, "1000-2/7"),
        ({"obsnum": 1000, "scannum": 2}, "1000-*-2"),
        ({"subobsnum": 2}, "*-2"),
        ({"scannum": 2}, "*-*-2"),
        ({"interface": "tel"}, "*/tel"),
        ({"master": "tcs", "interface": "tel"}, "tcs-*/tel"),
        ({"subobsnum": 2, "interface": 7}, "*-2/7"),
        ({"scannum": 2, "interface": 7}, "*/2/7"),
        ({"obsnum": 0, "interface": 0}, "0/0"),
        (
            {"master": "tcs", "obsnum": 1000, "interface": "toltec7"},
            "tcs-1000/toltec7",
        ),
        (
            {"obsnum": 1000, "subobsnum": 1, "scannum": 2, "interface": 7},
            "1000-1-2-7",
        ),
        (
            {
                "obsnum": 1000,
                "scannum": FieldSelection(None, slice(-1, None)),
                "interface": ("tel", 3, 1),
            },
            "1000/*[-1:]/{1,3,tel}",
        ),
    ],
)
def test_normalized_text_uses_validated_field_positions(data, text):
    spec = ObsSpec(**data)
    assert str(spec) == text
    assert dict(ObsSpec.parse(text).constraints()) == dict(spec.constraints())
    assert str(ObsSpec.parse(text)) == text


@pytest.mark.parametrize(
    ("text", "constraints", "canonical"),
    [
        ("1000-*-*-7", {"obsnum": 1000, "interface": 7}, "1000/7"),
        (
            "1000-2-*-7",
            {"obsnum": 1000, "subobsnum": 2, "interface": 7},
            "1000-2/7",
        ),
        ("1000-*-2-*", {"obsnum": 1000, "scannum": 2}, "1000-*-2"),
    ],
)
def test_text_gap_normalization_preserves_field_positions(text, constraints, canonical):
    spec = ObsSpec.parse(text)
    assert dict(spec.constraints()) == constraints
    assert str(spec) == canonical
    assert ObsSpec.parse(canonical) == spec


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "  ics-018596-32-0  ",
        None,
        True,
        -1,
        "abc",
        "1000//7",
        "1000-",
        "1000/",
        "1000-1-2-3-4",
        "1000-1/2/3/4",
        "1000/1-2",
        "1/foo-2",
        "1000-{1,no}",
        "1000-{1,}",
        "1000-[1:2:3:4]",
        "1000-[1:2:0]",
        "[-1]/[-1]",
        "123*",
        "tcs:1",
        "@tcs-1",
        "@*-1",
    ],
)
def test_invalid_never_becomes_unconstrained(text):
    with pytest.raises((ValueError, TypeError)):
        ObsSpec.parse(text)


@pytest.mark.parametrize("text", ["tcs-1", "{tcs,ics}-1/{tel,1}", "[-1]/(1,5)"])
@pytest.mark.parametrize("whitespace", [" ", "\t", "\n", "\r", "\u00a0"])
def test_whitespace_is_rejected_including_prefix_and_interface(text, whitespace):
    for position in range(len(text) + 1):
        with pytest.raises(ValueError):
            ObsSpec.parse(text[:position] + whitespace + text[position:])


def test_immutable_normalized_sets_and_validation():
    spec = ObsSpec(obsnum=(2, 1, 2))
    assert spec.obsnum == (1, 2)
    with pytest.raises(FrozenInstanceError):
        # Exercise the runtime guard despite the statically frozen attribute.
        spec.obsnum = 3  # ty: ignore[invalid-assignment]
    for kwargs in (
        {"obsnum": -1},
        {"obsnum": True},
        {"obsnum": ()},
    ):
        with pytest.raises(ValueError):
            # Invalid constructor inputs deliberately bypass static constraints.
            ObsSpec(**kwargs)


def test_primary_wildcard_is_not_a_master_wildcard():
    table = pd.DataFrame(
        {"master": ["tcs", "ics", "tcs"], "obsnum": [1, 2, 1], "subobsnum": [0, 1, 1]}
    )
    assert ObsSpec.parse("1").select(table).index.tolist() == [0, 2]
    assert ObsSpec.parse("*-1").select(table).index.tolist() == [1, 2]
    assert ObsSpec.parse("tcs-*-1").select(table).index.tolist() == [2]


def test_value_bounds_are_distinct_from_positional_slices():
    table = pd.DataFrame(
        {"obsnum": [0, 1000000, 1000001, None], "subobsnum": [0, 10002, 10002, 10002]}
    )
    spec = ObsSpec.parse("(1000000,)[::2]-(10000,)")
    assert spec.select(table).index.tolist() == [1]
    assert ObsSpec.parse("0").select(table).index.tolist() == [0]
    assert ObsSpec.parse("[1000000:]").select(table).empty
    assert ObsSpec.parse("[9:1]").select(table).empty
    with pytest.raises(ValueError, match="lacks selector field"):
        ObsSpec.parse("1/7").select(table)
