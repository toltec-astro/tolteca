"""Full raw metadata requires the recorded observation type."""

import pytest
from tollan.accessor.xarray import ensure_datatree

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kids import BlockCreate


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_missing_raw_observation_type_is_not_guessed(data_lmt_path, suffix):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    with open_toltec_kids(path) as data:
        incomplete = data.drop_vars("Header.Toltec.ObsType")
        # Shape can identify a family, but cannot distinguish tune from VNA/TARG.
        kind_expected = (
            ToltecDataKind.raw_sweep if suffix else ToltecDataKind.raw_timestream
        )
        assert incomplete.toltec_kids.data_kind == kind_expected
        with pytest.raises(ValueError, match="Header.Toltec.ObsType"):
            _ = incomplete.toltec_kids.meta
        for name in ("Data.Toltec.Is", "Data.Toltec.Qs", "Data.Toltec.LoFreq"):
            if name in data:
                assert not data[name].variable._in_memory


def test_processed_type_header_remains_optional(data_lmt_path):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
    with open_toltec_kids(path) as data:
        # Deliberately exercise a processed-header encoding, not a claim about
        # a real processed acquisition in the reference fixture inventory.
        processed = data.drop_vars("Header.Toltec.ObsType").assign_attrs(
            {"Header.Kids.kind": "processed_sweep"}
        )
        assert processed.toltec_kids.data_kind == ToltecDataKind.reduced_sweep
        assert (
            BlockCreate()(ensure_datatree(processed))
            .toltec_kids.blocks[0]
            .meta.obs_type_code
            is None
        )


@pytest.mark.parametrize("code", [-1, 5])
def test_unrecognized_observation_type_has_unknown_data_kind(data_lmt_path, code):
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc")
    with open_toltec_kids(path) as data:
        invalid = data.assign({"Header.Toltec.ObsType": code})
        assert invalid.toltec_kids.data_kind == ToltecDataKind.unknown
        with pytest.raises(ValueError, match="raw KIDs acquisition"):
            _ = invalid.toltec_kids.meta
