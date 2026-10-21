"""Flat file-table construction and row-based selection."""

from tolteca_datamodels.toltec.file import (
    guess_info_from_sources,
)

PATHS = [
    "tel_toltec_2025-10-31_145609_00_0002.nc",
    "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
]


def test_filename_uids_allow_an_unknown_master():
    frame = guess_info_from_sources(PATHS)
    assert frame.uid_obs.tolist() == ["tcs-145609", "145605"]
    assert frame.uid_raw_obs.tolist() == ["tcs-145609-0-2", "145605-0-1"]
    assert frame.uid_raw_obs_file.tolist() == [
        "tcs-145609-0-2-tel",
        "145605-0-1-toltec8",
    ]
    assert frame.toltec_file.query("uid_obs == '145605'").source.tolist() == [PATHS[1]]


def test_selected_rows_expose_projected_information():
    frame = guess_info_from_sources(PATHS)
    selected = frame.toltec_file.query("master == 'tcs' & obsnum == 145609")
    (row,) = selected.itertuples(index=False)
    assert row.source == PATHS[0]
    assert row.uid_raw_obs_file == "tcs-145609-0-2-tel"
    assert row.file_prefix == "tel_toltec"
    assert row.filepath_info is frame.filepath_info.iloc[0]
    assert row.filepath_info.prefix == row.file_prefix
