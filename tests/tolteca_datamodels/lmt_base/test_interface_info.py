"""Complete shared interface identity and telescope filename specialization."""

import astropy.units as u
import pytest
from astropy.time import Time
from pydantic import ValidationError

from tolteca_datamodels.lmt.types import LmtInfo
from tolteca_datamodels.lmt_base.metadata import InterfaceInfoMixin, LmtRawObsMetadata
from tolteca_datamodels.toltec.types import ToltecInfo


@pytest.mark.parametrize("model", [InterfaceInfoMixin, LmtRawObsMetadata])
@pytest.mark.parametrize("field", ["instru", "producer", "interface_kind", "interface"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "null"])
def test_shared_interface_identity_requires_every_field(model, field, missing):
    data: dict[str, object] = {
        "instru": "toltec",
        "producer": "lmt",
        "interface_kind": "tel",
        "interface": "tel",
    }
    if model is LmtRawObsMetadata:
        data |= {
            "master": "tcs",
            "obsnum": 145609,
            "subobsnum": 0,
            "scannum": 2,
            "t0": Time("2025-10-31T00:00:00"),
            "t1": Time("2025-10-31T00:00:01"),
            "t_int": 1 * u.s,
        }
    model.model_validate(data)
    if missing:
        del data[field]
    else:
        data[field] = None
    with pytest.raises(ValidationError) as exc:
        model.model_validate(data)
    assert [(e["loc"], e["type"]) for e in exc.value.errors()] == [
        ((field,), "missing" if missing else "string_type")
    ]


def test_instrument_descriptor_completes_telescope_filename_identity():
    partial = LmtInfo.interface_interface_info["tel"]
    scoped = ToltecInfo.interface_interface_info["tel"]
    assert partial.instru is None
    assert isinstance(scoped, InterfaceInfoMixin)
    assert scoped.instru == "toltec"
    assert partial.model_dump(exclude={"instru"}) == scoped.model_dump(
        exclude={"instru"}
    )
    assert InterfaceInfoMixin.model_validate_json(scoped.model_dump_json()) == scoped
    with pytest.raises(ValidationError, match="frozen"):
        scoped.instru = "other"
