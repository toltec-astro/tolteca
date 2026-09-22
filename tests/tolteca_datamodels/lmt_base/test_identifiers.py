"""Shared observation identifiers and their concrete metadata contracts."""

import dataclasses
import json

import astropy.units as u
import pytest
from astropy.time import Time
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.lmt_base import LmtObsQuartetMixin, LmtRawObsMetadata
from tolteca_datamodels.toltec import ToltecRawObsMetadata
from tolteca_datamodels.toltec.toltecdb import ToltecDBRawObsMaster


@pytest.fixture
def identified_metadata():
    return LmtRawObsMetadata(
        master="simu",
        obsnum=145609,
        subobsnum=0,
        scannum=2,
        instru="lmt",
        interface="tel",
        t0=Time("2025-10-31T09:27:56"),
        t1=Time("2025-10-31T09:28:56"),
        t_exp=60 * u.s,
    )


@pytest.mark.parametrize("master", ["tcs", "ics", "clip", "simu"])
def test_quartet_names(master):
    meta = LmtObsQuartetMixin(master=master, obsnum=145609, subobsnum=0, scannum=2)
    assert meta.master == master
    assert meta.master_str == master
    assert meta.quartet == f"{master}-145609-0-2"


@pytest.mark.parametrize("master", [None, 0, 99])
def test_unavailable_master_name(identified_metadata, master):
    meta = dataclasses.replace(identified_metadata, master=master)
    assert meta.master == master
    assert meta.master_str is None
    assert meta.quartet is None
    assert meta.uid_raw_obs is None
    assert meta.uid_raw_obs_file is None


def test_identifiers_are_derived_cached_and_serialized(identified_metadata):
    meta = identified_metadata
    expected = {
        "master_str": "simu",
        "quartet": "simu-145609-0-2",
        "uid_raw_obs": "simu-145609-0-2",
        "uid_raw_obs_file": "simu-145609-0-2-tel",
    }
    assert not expected.keys() & meta.__dict__.keys()
    for name, value in expected.items():
        assert getattr(meta, name) == value
        assert getattr(meta, name) is meta.__dict__[name]
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(meta, name, "changed")
    assert meta.uid_raw_obs is meta.quartet

    adapter = TypeAdapter(LmtRawObsMetadata)
    payload = adapter.dump_python(meta, mode="json")
    assert expected.items() <= payload.items()
    schema = adapter.json_schema(mode="serialization")["properties"]
    assert all(schema[name]["readOnly"] for name in expected)
    assert not expected.keys() & adapter.json_schema()["properties"].keys()

    inputs = adapter.dump_json(meta, round_trip=True)
    restored = adapter.validate_json(inputs)
    assert not expected.keys() & restored.__dict__.keys()
    assert adapter.dump_python(restored, mode="json") == payload
    # Derived values supplied in a serialized object never override the inputs.
    payload.update(dict.fromkeys(expected, "incorrect"))
    restored = adapter.validate_json(json.dumps(payload))
    assert restored.uid_raw_obs_file == expected["uid_raw_obs_file"]


def test_replacing_metadata_recomputes_identifiers(identified_metadata):
    meta = identified_metadata
    assert meta.uid_raw_obs_file == "simu-145609-0-2-tel"
    other_interface = dataclasses.replace(meta, interface="tel2")
    assert other_interface.quartet == meta.quartet
    assert other_interface.uid_raw_obs == meta.uid_raw_obs
    assert other_interface.uid_raw_obs_file == "simu-145609-0-2-tel2"
    other_scan = dataclasses.replace(meta, master="tcs", scannum=3)
    assert "quartet" not in other_scan.__dict__
    assert other_scan.uid_raw_obs_file == "tcs-145609-0-3-tel"
    assert meta.uid_raw_obs_file == "simu-145609-0-2-tel"


def test_master_stays_required():
    with pytest.raises(ValidationError, match="master"):
        TypeAdapter(LmtObsQuartetMixin).validate_json(
            '{"obsnum": 145609, "subobsnum": 0, "scannum": 2}'
        )


def test_toltec_inherits_interface_identifiers(identified_metadata):
    fields = dataclasses.asdict(identified_metadata)
    fields.update(master=ToltecDBRawObsMaster.ICS, interface="toltec0", roach=12)
    meta = ToltecRawObsMetadata(**fields)
    assert meta.master_str == "ics"
    assert meta.quartet == "ics-145609-0-2"
    assert meta.uid_raw_obs == meta.quartet
    # Interface is authoritative even when the ROACH field differs.
    assert meta.uid_raw_obs_file == "ics-145609-0-2-toltec0"


@pytest.mark.parametrize("master", [None, "tcs", 99])
def test_toltec_master_requires_a_recorded_code(identified_metadata, master):
    fields = dataclasses.asdict(identified_metadata)
    fields["master"] = master
    with pytest.raises(ValidationError, match="master"):
        ToltecRawObsMetadata(**fields)
