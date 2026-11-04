"""Complete raw metadata requires the acquisition's recorded readout settings."""

import pytest

from tolteca_datamodels.toltec import open_toltec_kids


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
@pytest.mark.parametrize(
    "header",
    [
        "RoachIndex",
        "DriveAtten",
        "SenseAtten",
        "TargSweepObsNum",
        "TargSweepSubObsNum",
        "TargSweepScanNum",
        "MaxNumTones",
        "Filename",
        "RepeatLevel",
    ],
)
def test_raw_metadata_requires_readout_headers(data_lmt_path, suffix, header):
    root = data_lmt_path / "toltec/tcs/toltec0"
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    name = f"Header.Toltec.{header}"
    with open_toltec_kids(path) as data:
        assert name in data
        missing = data.drop_vars(name)
        error_type = KeyError if header == "Filename" else ValueError
        with pytest.raises(error_type, match=name):
            _ = missing.toltec_kids.meta
        assert not data["Data.Toltec.Is"].variable._in_memory
