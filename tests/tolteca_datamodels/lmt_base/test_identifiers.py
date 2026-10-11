"""Shared observation identifiers and their concrete metadata contracts."""

import json

import astropy.units as u
import pandas as pd
import pytest
from astropy.time import Time
from pydantic import TypeAdapter, ValidationError

from tolteca_datamodels.lmt_base import funcs
from tolteca_datamodels.lmt_base.metadata import (
    LmtObsQuartetMixin,
    LmtRawObsMetadata,
)
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata
from tolteca_datamodels.toltec.toltecdb import ToltecDBRawObsMasterCode


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        (
            None,
            ("tcs-145610-0-2", "tcs-145610", "tcs-145610-0-2", "tcs-145610-0-2-tel"),
        ),
        ("master", ("145610-0-2", "145610", "145610-0-2", "145610-0-2-tel")),
        ("obsnum", (None, None, None, None)),
        ("subobsnum", (None, "tcs-145610", None, None)),
        ("scannum", ("tcs-145610-0", "tcs-145610", None, None)),
        ("interface", ("tcs-145610-0-2", "tcs-145610", "tcs-145610-0-2", None)),
    ],
)
def test_uid_columns_preserve_inputs_and_row_order(missing, expected):
    table = pd.DataFrame(
        {
            "master": pd.array(["tcs"] * 3, dtype="string"),
            "obsnum": pd.array([145609, 145610, 145611], dtype="Int64"),
            "subobsnum": pd.array([0] * 3, dtype="Int64"),
            "scannum": pd.array([2] * 3, dtype="Int64"),
            "interface": pd.array(["tel"] * 3, dtype="string"),
            "batch": ["first", "second", "third"],
        },
        index=pd.Index([8, 3, 8], name="input_row"),
    )
    if missing is not None:
        table.loc[3, missing] = pd.NA
    before = table.copy()
    result = funcs.make_uid_columns(
        table.master, table.obsnum, table.subobsnum, table.scannum, table.interface
    )
    pd.testing.assert_frame_equal(table, before)
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            [
                (
                    "tcs-145609-0-2",
                    "tcs-145609",
                    "tcs-145609-0-2",
                    "tcs-145609-0-2-tel",
                ),
                expected,
                (
                    "tcs-145611-0-2",
                    "tcs-145611",
                    "tcs-145611-0-2",
                    "tcs-145611-0-2-tel",
                ),
            ],
            index=table.index,
            columns=["quartet", "uid_obs", "uid_raw_obs", "uid_raw_obs_file"],
            dtype="string",
        ),
    )


def test_uid_columns_empty_inputs_have_nullable_identifier_columns():
    table = pd.DataFrame(
        columns=["master", "obsnum", "subobsnum", "scannum", "interface"]
    )
    result = funcs.make_uid_columns(
        table.master, table.obsnum, table.subobsnum, table.scannum, table.interface
    )
    assert result.empty
    pd.testing.assert_index_equal(result.index, table.index)
    for name in ("quartet", "uid_obs", "uid_raw_obs", "uid_raw_obs_file"):
        assert str(result[name].dtype) == "string"


@pytest.mark.parametrize("master", ["tcs", None])
@pytest.mark.parametrize(
    ("numbers", "suffix"),
    [
        ((1, 2, 3), "1-2-3"),
        ((1, 2, None), "1-2"),
        ((1, None, None), "1"),
        ((0, 0, 0), "0-0-0"),
        ((1, None, 3), None),
        ((None, 2, 3), None),
        ((None, None, 3), None),
        ((None, 2, None), None),
        ((None, None, None), None),
    ],
)
def test_quartet_preserves_numeric_positions(master, numbers, suffix):
    expected = (
        f"{master}-{suffix}" if master is not None and suffix is not None else suffix
    )
    assert funcs.make_quartet(master, *numbers) == expected


@pytest.mark.parametrize(
    ("master", "numbers", "uid_obs", "uid_raw_obs"),
    [
        ("tcs", (1, 1, 1), "tcs-1", "tcs-1-1-1"),
        (None, (1, 1, 1), "1", "1-1-1"),
        ("tcs", (1, None, None), "tcs-1", None),
        (None, (1, None, None), "1", None),
        ("tcs", (1, 1, None), "tcs-1", None),
        ("tcs", (1, None, 1), "tcs-1", None),
        ("tcs", (None, 1, 1), None, None),
        (None, (None, None, None), None, None),
        (None, (0, 0, 0), "0", "0-0-0"),
    ],
)
def test_uid_scope_requires_its_own_observation_numbers(
    master, numbers, uid_obs, uid_raw_obs
):
    assert funcs.make_uid_obs(master, numbers[0]) == uid_obs
    assert funcs.make_uid_raw_obs(master, *numbers) == uid_raw_obs
    assert funcs.make_uid_raw_obs_file(master, *numbers, "tel") == (
        None if uid_raw_obs is None else f"{uid_raw_obs}-tel"
    )


@pytest.mark.parametrize("column", ["obsnum", "subobsnum", "scannum", "interface"])
@pytest.mark.parametrize("index", [[3, 8, 8], [8, 3], [8, 3, 9]])
def test_uid_columns_reject_misaligned_inputs(column, index):
    data = {
        name: pd.Series([value] * 3, index=[8, 3, 8])
        for name, value in (
            ("master", "tcs"),
            ("obsnum", 145609),
            ("subobsnum", 0),
            ("scannum", 2),
            ("interface", "tel"),
        )
    }
    data[column] = pd.Series([data[column].iloc[0]] * len(index), index=index)
    with pytest.raises(ValueError, match="equal indices in the same order"):
        funcs.make_uid_columns(**data)


def test_file_uid_uses_observation_fields_and_shared_constructor(monkeypatch):
    values = ("tcs", 145609, 0, 2, "tel")
    assert funcs.make_uid_raw_obs_file(*values) == "tcs-145609-0-2-tel"

    def make_uid(*args):
        assert args == values[:4]
        return "observation"

    monkeypatch.setattr(funcs, "make_uid_raw_obs", make_uid)
    assert funcs.make_uid_raw_obs_file(*values) == "observation-tel"


@pytest.mark.parametrize("master", ["tcs", None])
def test_file_uid_requires_interface(master):
    assert funcs.make_uid_raw_obs_file(master, 145609, 0, 2, None) is None


@pytest.fixture
def identified_metadata():
    return LmtRawObsMetadata(
        master="simu",
        obsnum=145609,
        subobsnum=0,
        scannum=2,
        instru="toltec",
        producer="lmt",
        interface="tel",
        interface_kind="tel",
        t0=Time("2025-10-31T09:27:56"),
        t1=Time("2025-10-31T09:28:56"),
        t_int=60 * u.s,
    )


@pytest.mark.parametrize("master", ["tcs", "ics", "clip", "simu"])
def test_quartet_names(master):
    meta = LmtObsQuartetMixin(master=master, obsnum=145609, subobsnum=0, scannum=2)
    assert meta.master == master
    assert meta.quartet == f"{master}-145609-0-2"


def test_nullable_master_produces_unqualified_identifiers(identified_metadata):
    meta = type(identified_metadata).model_validate(
        identified_metadata.model_dump(round_trip=True) | {"master": None}
    )
    assert meta.master is None
    assert meta.quartet == "145609-0-2"
    assert meta.uid_obs == "145609"
    assert meta.uid_raw_obs == "145609-0-2"
    assert meta.uid_raw_obs_file == "145609-0-2-tel"


@pytest.mark.parametrize("master", [0, 99, True, 0.5])
def test_numeric_masters_are_not_labels(identified_metadata, master):
    with pytest.raises(ValidationError, match="master"):
        type(identified_metadata).model_validate(
            identified_metadata.model_dump(round_trip=True) | {"master": master}
        )


def test_identifiers_are_derived_cached_and_serialized(identified_metadata):
    meta = identified_metadata
    expected = {
        "quartet": "simu-145609-0-2",
        "uid_obs": "simu-145609",
        "uid_raw_obs": "simu-145609-0-2",
        "uid_raw_obs_file": "simu-145609-0-2-tel",
    }
    assert not expected.keys() & meta.__dict__.keys()
    for name, value in expected.items():
        assert getattr(meta, name) == value
        assert getattr(meta, name) is meta.__dict__[name]
    assert meta.uid_raw_obs == meta.quartet

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
    other_interface = type(meta).model_validate(
        meta.model_dump(round_trip=True) | {"interface": "tel2"}
    )
    assert other_interface.quartet == meta.quartet
    assert other_interface.uid_obs == meta.uid_obs
    assert other_interface.uid_raw_obs == meta.uid_raw_obs
    assert other_interface.uid_raw_obs_file == "simu-145609-0-2-tel2"
    other_scan = type(meta).model_validate(
        meta.model_dump(round_trip=True) | {"master": "tcs", "scannum": 3}
    )
    assert "quartet" not in other_scan.__dict__
    assert other_scan.uid_raw_obs_file == "tcs-145609-0-3-tel"
    assert meta.uid_raw_obs_file == "simu-145609-0-2-tel"


def test_master_stays_required():
    with pytest.raises(ValidationError, match="master"):
        TypeAdapter(LmtObsQuartetMixin).validate_json(
            '{"obsnum": 145609, "subobsnum": 0, "scannum": 2}'
        )


def test_toltec_inherits_interface_identifiers(identified_metadata, kids_filepath_orig):
    fields = identified_metadata.model_dump(round_trip=True)
    fields.update(
        master=ToltecDBRawObsMasterCode.get_master(ToltecDBRawObsMasterCode.ICS),
        master_code=ToltecDBRawObsMasterCode.ICS,
        instru="toltec",
        producer="toltec",
        interface="toltec0",
        readout_code=12,
        roach=12,
        filepath_orig=kids_filepath_orig,
    )
    meta = ToltecRawObsMetadata(
        **fields, n_blocks_max=1, n_times=1000, n_chans=256, f_smp=500 * u.Hz
    )
    assert meta.master == "ics"
    assert meta.quartet == "ics-145609-0-2"
    assert meta.uid_obs == "ics-145609"
    assert meta.uid_raw_obs == meta.quartet
    # Interface is authoritative even when the ROACH field differs.
    assert meta.uid_raw_obs_file == "ics-145609-0-2-toltec0"


@pytest.mark.parametrize("master_code", [None, "tcs", "0", True, 0.5])
def test_toltec_master_requires_a_recorded_code(
    identified_metadata, kids_filepath_orig, master_code
):
    fields = identified_metadata.model_dump(round_trip=True)
    fields["master_code"] = master_code
    fields["master"] = ToltecDBRawObsMasterCode.get_master(ToltecDBRawObsMasterCode.TCS)
    fields["readout_code"] = 0
    fields["instru"] = "toltec"
    fields["filepath_orig"] = kids_filepath_orig
    with pytest.raises(ValidationError, match="master_code"):
        ToltecRawObsMetadata(
            **fields, n_blocks_max=1, n_times=1000, n_chans=256, f_smp=500 * u.Hz
        )
