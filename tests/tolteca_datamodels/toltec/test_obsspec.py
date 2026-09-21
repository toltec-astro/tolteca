"""Shared selector grammar, normalization, and metadata selection."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from tolteca_datamodels.toltec.obsspec import IdRange, ObsSpec


@pytest.mark.parametrize(
    ("text", "constraints"),
    [
        (123456, {"obsnum": 123456}),
        (
            "  ics-018596-32-0  ",
            {"master": "ics", "obsnum": 18596, "subobsnum": 32, "scannum": 0},
        ),
        ("tcs-1000", {"master": "tcs", "obsnum": 1000}),
        (
            "clip-1000-2-3-7",
            {
                "master": "clip",
                "obsnum": 1000,
                "subobsnum": 2,
                "scannum": 3,
                "roach": 7,
            },
        ),
        ("1000/7", {"obsnum": 1000, "roach": 7}),
        ("1000-2/7", {"obsnum": 1000, "subobsnum": 2, "roach": 7}),
        ("1000/2/7", {"obsnum": 1000, "scannum": 2, "roach": 7}),
        ("1000/1/2/7", {"obsnum": 1000, "subobsnum": 1, "scannum": 2, "roach": 7}),
        ("1000/toltec7", {"obsnum": 1000, "roach": 7}),
        ("1000/hwpr", {"obsnum": 1000, "interface": "hwpr"}),
        ("1000-{2,0,2}", {"obsnum": 1000, "subobsnum": (0, 2)}),
        ("1000-[0:10:2]", {"obsnum": 1000, "subobsnum": IdRange(0, 10, 2)}),
        ("1000-[5:]", {"obsnum": 1000, "subobsnum": IdRange(5)}),
        ("1000-[:5]", {"obsnum": 1000, "subobsnum": IdRange(0, 5)}),
        (
            "[1000]-[2]-[3]/[7]",
            {"obsnum": 1000, "subobsnum": 2, "scannum": 3, "roach": 7},
        ),
        ("{0,1000}/[0:13:2]", {"obsnum": (0, 1000), "roach": IdRange(0, 13, 2)}),
        ("*", {}),
        ("[]", {}),
        ("{}", {}),
        ("[:]", {}),
        ("1000-[]-[]-1", {"obsnum": 1000, "roach": 1}),
    ],
)
def test_parse_and_roundtrip(text, constraints):
    spec = ObsSpec.parse(text)
    assert dict(spec.constraints()) == constraints
    assert ObsSpec.parse(str(spec)) == spec
    assert ObsSpec.parse(spec) is spec
    assert hash(ObsSpec.parse(str(spec))) == hash(spec)


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        None,
        True,
        -1,
        "abc",
        "simu-1000",
        "1000//7",
        "1000-",
        "1000/",
        "1000-1-2-3-4",
        "1000-1/2/3/4",
        "1000/1-2",
        "1000-{1,no}",
        "1000-{1,}",
        "1000-[1:2:3:4]",
        "1000-[1:2:0]",
        "1000/13",
        "1000/toltec13",
        "123*",
    ],
)
def test_invalid_never_becomes_unconstrained(text):
    with pytest.raises((ValueError, TypeError)):
        ObsSpec.parse(text)


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
        {"roach": 7, "interface": "hwpr"},
        {"master": "simu"},
    ):
        with pytest.raises(ValueError):
            # Invalid constructor inputs deliberately bypass static constraints.
            ObsSpec(**kwargs)  # ty: ignore[invalid-argument-type]


def test_catalog_ranges_are_identifier_values_not_positions():
    catalog = pd.DataFrame(
        {"obsnum": [0, 1000000, 1000001, None], "subobsnum": [0, 10002, 10002, 10002]}
    )
    spec = ObsSpec.parse("[1000000::2]-[10000:]")
    assert spec.select(catalog).index.tolist() == [1]
    assert ObsSpec.parse("0").select(catalog).index.tolist() == [0]
    assert ObsSpec.parse("[9:1]").select(catalog).empty
    assert spec.matches({"obsnum": 1000000, "subobsnum": 10002})
    assert not spec.matches({"obsnum": 1000000})
    with pytest.raises(ValueError, match="lacks selector field"):
        ObsSpec.parse("1/7").select(catalog)
