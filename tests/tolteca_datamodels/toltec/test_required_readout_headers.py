"""Complete raw metadata requires the acquisition's recorded readout settings."""

import pytest

from tolteca_datamodels.toltec import open_toltec


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
@pytest.mark.parametrize(
    "header",
    [
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
    pattern = f"*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = root.glob(pattern)
    name = f"Header.Toltec.{header}"
    with open_toltec(path) as data:
        assert name in data
        missing = data.drop_vars(name)
        error_type = KeyError if header == "Filename" else ValueError
        with pytest.raises(error_type, match=name):
            _ = missing.toltec_kids.meta
        assert not data["Data.Toltec.Is"].variable._in_memory


def test_processed_metadata_can_omit_readout_settings(real_targsweep_file):
    with open_toltec(real_targsweep_file) as data:
        # Deliberately synthetic processed encoding, not an acquisition fixture.
        processed = data.drop_vars(
            [
                "Header.Toltec.DriveAtten",
                "Header.Toltec.SenseAtten",
                "Header.Toltec.TargSweepObsNum",
                "Header.Toltec.TargSweepSubObsNum",
                "Header.Toltec.TargSweepScanNum",
                "Header.Toltec.MaxNumTones",
                "Header.Toltec.Filename",
                "Header.Toltec.RepeatLevel",
            ],
        ).assign_attrs({"Header.Kids.kind": "processed_sweep"})
        meta = processed.toltec_kids.meta
        for field in (
            "atten_drive",
            "atten_sense",
            "cal_obsnum",
            "cal_subobsnum",
            "cal_scannum",
            "n_chans_max",
            "filepath_orig",
            "repeat",
        ):
            assert getattr(meta, field) is None
