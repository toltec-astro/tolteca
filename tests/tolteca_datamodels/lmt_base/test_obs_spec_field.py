"""Observation selectors in Pydantic configuration fields."""

import json

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from tolteca_datamodels.lmt_base.obs_spec import ObsSpec, ObsSpecField


class SelectionConfig(BaseModel):
    obs_spec: ObsSpecField


@pytest.mark.parametrize(
    ("value", "normalized"),
    [
        (145609, "145609"),
        ("ics-018596-32-0", "ics-18596-32-0"),
        ("1/foo(100,200)[-1]bar", "1/foo(100,200)[-1:]bar"),
        ("{ics,tcs,ics}-1/{tel,3,1}", "{ics,tcs}-1/{1,3,tel}"),
        ("1000-*-2-7", "1000/2/7"),
        ("*[:]-[]", "*"),
    ],
)
def test_field_validation_and_dumps_share_the_normalized_string(value, normalized):
    config = SelectionConfig.model_validate({"obs_spec": value})
    spec = config.obs_spec
    assert isinstance(spec, ObsSpec)
    assert str(spec) == normalized
    assert ObsSpec.parse(normalized) == spec
    assert config.model_dump() == {"obs_spec": normalized}
    assert config.model_dump(mode="json") == {"obs_spec": normalized}
    assert json.loads(config.model_dump_json()) == {"obs_spec": normalized}
    assert SelectionConfig.model_validate(config.model_dump()) == config
    assert SelectionConfig.model_validate_json(config.model_dump_json()) == config
    assert SelectionConfig.model_validate({"obs_spec": spec}).obs_spec is spec


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        1.5,
        [],
        {},
        -1,
        "",
        "bad-selector",
        "[-1]/[-1]",
    ],
)
def test_invalid_config_values_raise_validation_error(value):
    with pytest.raises(ValidationError):
        SelectionConfig.model_validate({"obs_spec": value})


def test_field_schema_and_optional_containers():
    adapter = TypeAdapter(ObsSpecField)
    assert adapter.json_schema() == {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert adapter.json_schema(mode="serialization") == {"type": "string"}
    collection = TypeAdapter(list[ObsSpecField | None])
    selectors = collection.validate_python([None, "1/tel", 2])
    assert selectors == [None, ObsSpec.parse("1/tel"), ObsSpec(obsnum=2)]
    assert collection.dump_python(selectors) == [None, "1/tel", "2"]
    assert collection.validate_json(collection.dump_json(selectors)) == selectors
