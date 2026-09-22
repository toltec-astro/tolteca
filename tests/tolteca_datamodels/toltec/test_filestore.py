"""Observation discovery in the recorded LMT and TolTEC directory layouts."""

from pathlib import Path

import pytest
import xarray as xr

from tolteca_datamodels.toltec.filestore import resolve_sources


@pytest.mark.parametrize(
    ("spec", "names"),
    [
        (
            "145609/tel",
            [
                "tel_toltec_2025-10-31_145609_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0002.nc",
            ],
        ),
        ("145609/2/tel", ["tel_toltec_2025-10-31_145609_00_0002.nc"]),
        (
            "{145605,145609}/tel",
            [
                "tel_toltec_2025-10-31_145605_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0002.nc",
            ],
        ),
        (
            "[145605:145610:4]/tel",
            [
                "tel_toltec_2025-10-31_145605_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0001.nc",
                "tel_toltec_2025-10-31_145609_00_0002.nc",
            ],
        ),
    ],
)
def test_telescope_selectors_use_recorded_layout_without_opening(
    data_lmt_path, monkeypatch, spec, names
):
    expected = [data_lmt_path / "tel" / name for name in names]
    assert all(path.is_file() for path in expected)

    def unexpected(*args, **kwargs):
        pytest.fail("Filename discovery opened a source")

    monkeypatch.setattr(Path, "open", unexpected)
    monkeypatch.setattr(xr, "open_dataset", unexpected)
    catalog = resolve_sources([spec], data_root=data_lmt_path)
    assert catalog["source"].tolist() == [str(path) for path in expected]
    assert catalog["interface"].tolist() == ["tel"] * len(expected)
    assert catalog["master"].isna().all()
    assert catalog["data_kind"].tolist() == ["LmtTel"] * len(expected)


def test_unqualified_and_roach_selectors_keep_toltec_scope(data_lmt_path):
    names = [
        "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
        "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
    ]
    expected = [str(data_lmt_path / "toltec/tcs/toltec0" / name) for name in names]
    for spec in ("145609", "145609/0", "145609/toltec0", "tcs-145609/0"):
        catalog = resolve_sources([spec], data_root=data_lmt_path)
        assert catalog["source"].tolist() == expected
        assert catalog["interface"].tolist() == ["toltec0", "toltec0"]


def test_telescope_selector_does_not_invent_master(data_lmt_path):
    # Master is a header value; the telescope directory/name does not supply it.
    with pytest.raises(FileNotFoundError, match="No input files matched"):
        resolve_sources(["tcs-145609/tel"], data_root=data_lmt_path)


def test_telescope_selector_and_explicit_path_deduplicate(data_lmt_path, tmp_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    alias = tmp_path / path.name
    alias.symlink_to(path)
    catalog = resolve_sources([alias, "145609/2/tel"], data_root=data_lmt_path)
    assert catalog["source"].tolist() == [str(alias)]
