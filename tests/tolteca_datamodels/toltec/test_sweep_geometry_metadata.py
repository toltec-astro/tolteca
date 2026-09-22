"""Raw sweep counts come from recorded headers and native dimensions."""

import pytest

from tolteca_datamodels.toltec import open_toltec


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
    (path,) = root.glob(f"*_{suffix}.nc")
    with open_toltec(path) as data:
        if field == "time":
            # Keep arrays/channel dimensions; deliberately remove the mapped
            # time dimension rather than destroying I/Q with drop_dims().
            changed = data.rename_dims({"time": "unrecognized_sample_dimension"})
        elif field in data.dims:
            changed = data.drop_dims(field)
        else:
            changed = data.drop_vars(field)
        with pytest.raises(error_type, match=field):
            _ = changed.toltec_kids.meta
        assert not data["Data.Toltec.Is"].variable._in_memory


def test_processed_sweep_does_not_require_raw_geometry(real_targsweep_file):
    with open_toltec(real_targsweep_file) as data:
        reps = int(data["Header.Toltec.NumSamplesPerSweepStep"].item())
        # Synthetic one-value-per-step layout, not a numerical reduction oracle.
        changed = (
            data.isel(time=slice(None, None, reps))
            .rename({"time": "sweep"})
            .drop_dims("numSweeps")
            .drop_vars("Header.Toltec.NumSamplesPerSweepStep")
            .assign_attrs({"Header.Kids.kind": "processed_sweep"})
        )
        meta = changed.toltec_kids.meta
        assert meta.n_sweepreps is None
        assert meta.n_blocks_max == meta.n_blocks == 1
        assert meta.n_times == meta.n_sweepsteps == changed.sizes["sweep"]
