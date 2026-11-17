"""LMT selector fields do not depend on TolTEC ROACH conventions."""

import pytest

from tolteca_datamodels.core.obs_spec import FieldSelection, ValueBounds
from tolteca_datamodels.core.obs_spec import base as obs_spec
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec


def test_lmt_identifiers_and_interface_slot():
    spec = ObsSpec.parse("tcs-145609/2/tel")
    assert dict(spec.constraints()) == {
        "master": "tcs",
        "obsnum": 145609,
        "scannum": 2,
        "interface": "tel",
    }
    assert ObsSpec.parse(str(spec)) == spec
    assert ObsSpec.parse("145609/toltec7").interface == "toltec7"


def test_shared_layout_accepts_label_and_numeric_interface_syntax():
    assert ObsSpec.parse("145609/2/7").scannum == 2
    assert ObsSpec.parse("145609/2/7").interface == 7
    assert ObsSpec.parse("145609/toltec7").interface == "toltec7"
    assert ObsSpec.parse("145609/7").interface == 7


def test_constructor_uses_one_interface_field():
    assert ObsSpec("tcs", 145609, 0, 2, 7) == ObsSpec.parse("tcs-145609-0-2/7")
    assert ObsSpec("tcs", 145609, 0, 2, "tel") == ObsSpec.parse("tcs-145609-0-2/tel")


def test_instrument_master_policy_does_not_leak_to_shared_fields():
    assert ObsSpec.parse("simu-145609/tel").master == "simu"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tcs-1", {"master": "tcs", "obsnum": 1}),
        (
            "{tcs,ics}-(100,200)[-1]",
            {
                "master": ("ics", "tcs"),
                "obsnum": FieldSelection(ValueBounds(100, 200), slice(-1, None)),
            },
        ),
        ("tcs-*/tel", {"master": "tcs", "interface": "tel"}),
        ("*-1", {"subobsnum": 1}),
        ("{1,2}-3", {"obsnum": (1, 2), "subobsnum": 3}),
    ],
)
def test_numeric_primary_is_identified_independently_of_master_syntax(text, expected):
    spec = ObsSpec.parse(text)
    assert dict(spec.constraints()) == expected
    assert ObsSpec.parse(str(spec)) == spec


def test_all_tokens_use_the_shared_field_parser_once(monkeypatch):
    calls = []
    parse_selector = obs_spec._parse_selector

    def parse(token):
        calls.append(token)
        return parse_selector(token)

    monkeypatch.setattr(obs_spec, "_parse_selector", parse)
    spec = ObsSpec.parse("{tcs,ics}-(100,200)[-1]/tel")
    assert calls == ["{tcs,ics}", "(100,200)[-1]", "tel"]
    assert spec.master == ("ics", "tcs")
    assert spec.obsnum == FieldSelection(ValueBounds(100, 200), slice(-1, None))
    assert spec.interface == "tel"


@pytest.mark.parametrize("token", ["*", "{}", "{,}", "[]", "[:]"])
def test_unconstrained_tokens_anchor_the_primary_field(token):
    assert ObsSpec.parse(f"{token}-1") == ObsSpec(subobsnum=1)
    assert ObsSpec.parse(f"tcs-{token}-1") == ObsSpec(master="tcs", subobsnum=1)


@pytest.mark.parametrize("token", ["()", "(,)"])
def test_unbounded_numeric_tokens_anchor_and_constrain_the_primary_field(token):
    assert ObsSpec.parse(f"{token}-1") == ObsSpec(obsnum=ValueBounds(), subobsnum=1)


@pytest.mark.parametrize(
    "text", ["tcs", "{tcs,ics}", "tcs/tel", "_", "foo(1,2)", "1.0", "?", "@"]
)
def test_missing_primary_is_rejected(text):
    with pytest.raises(ValueError, match="primary field"):
        ObsSpec.parse(text)


@pytest.mark.parametrize(
    "text",
    [
        "tcs/1",
        "tcs-ics-1",
        "tcs-1-2-3-4-5",
        "tcs-1/2-3",
        "{tcs,}-1",
        "{tcs,1}-2",
        "foo(1,2)bar-1",
        "tcs-(1:2)-3",
        "tcs-{1,}-3",
        "tcs-foo(1,2)bar-3",
    ],
)
def test_primary_recognition_never_skips_invalid_fields(text):
    with pytest.raises(ValueError):
        ObsSpec.parse(text)


@pytest.mark.parametrize(
    ("token", "constraint"),
    [
        ("0123", 123),
        ("{1,3}[-1]", FieldSelection((1, 3), slice(-1, None))),
        ("(1,5)[+1:3:2]", FieldSelection(ValueBounds(1, 5), slice(1, 3, 2))),
        ("[-1,-3]", FieldSelection(None, (-3, -1))),
    ],
)
def test_primary_constraint_is_independent_of_master_prefix(token, constraint):
    assert dict(ObsSpec.parse(token).constraints()) == {"obsnum": constraint}
    assert dict(ObsSpec.parse(f"tcs-{token}").constraints()) == {
        "master": "tcs",
        "obsnum": constraint,
    }


@pytest.mark.parametrize("text", ["[\u0661]", "tcs-[\u0661]", "\uff11"])
def test_primary_values_and_positions_require_ascii_digits(text):
    with pytest.raises(ValueError):
        ObsSpec.parse(text)


@pytest.mark.parametrize("token", ["::", "{1,}", "[::0]", "(1:2)"])
def test_malformed_primary_selectors_are_rejected(token):
    with pytest.raises(ValueError):
        ObsSpec.parse(token)
