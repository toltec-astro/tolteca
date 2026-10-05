"""Observation timestamps retain the precision recorded by the acquisition."""

from pydantic import TypeAdapter

from tolteca_datamodels.lmt import open_lmt_tel
from tolteca_datamodels.lmt.metadata import LmtTelMetadata


def test_recorded_telescope_timestamps_roundtrip_exactly(data_lmt_path):
    paths = list((data_lmt_path / "tel").glob("tel*.nc"))
    assert paths
    adapter = TypeAdapter(LmtTelMetadata)
    for path in paths:
        with open_lmt_tel(path) as data:
            meta = data.lmt_tel.meta
            dumped = adapter.dump_python(meta, mode="json", round_trip=True)
            assert dumped["t0"] == meta.t0.unix
            assert dumped["t1"] == meta.t1.unix
            restored = adapter.validate_json(adapter.dump_json(meta, round_trip=True))
            assert restored.t0 == meta.t0
            assert restored.t1 == meta.t1
            assert restored.t_elapsed == meta.t_elapsed
