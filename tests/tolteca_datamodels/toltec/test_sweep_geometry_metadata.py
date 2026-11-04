"""Raw sweep counts come from recorded headers and native dimensions."""

import pytest

from tolteca_datamodels.toltec import open_toltec_kids


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune"])
@pytest.mark.parametrize(
    ("field", "error_type"),
    [
        ("Header.Toltec.NumSweepSteps", KeyError),
        ("Header.Toltec.NumSamplesPerSweepStep", ValueError),
        ("numSweeps", KeyError),
        ("time", KeyError),
    ],
)
def test_raw_sweep_requires_recorded_geometry(data_lmt_path, suffix, field, error_type):
    root = data_lmt_path / "toltec/tcs/toltec0"
    (path,) = root.glob(f"*1456*_{suffix}.nc")
    with open_toltec_kids(path) as data:
        if field == "time":
            # Keep arrays/channel dimensions; deliberately remove the mapped
            # time dimension rather than destroying I/Q with drop_dims().
            changed = data.rename_dims({"time": "unrecognized_sample_dimension"})
        elif field in data.dims:
            changed = data.drop_dims(field)
        else:
            changed = data.drop_vars(field)
        with pytest.raises(error_type, match=field):
            _ = changed.toltec_kids.block_index
        assert not data["Data.Toltec.Is"].variable._in_memory
