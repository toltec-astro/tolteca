"""Observation discovery in the recorded LMT and TolTEC directory layouts."""

from pathlib import Path

import pytest
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper, NetCDF4DataStore

from tolteca_datamodels.file import guess_info_from_source
from tolteca_datamodels.filestore import HeaderReadMode
from tolteca_datamodels.toltec.filestore import build_input_catalog, resolve_sources


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


@pytest.mark.parametrize("mode", list(HeaderReadMode))
@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize(
    "relative_path",
    [
        "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        "toltec/tcs/toltec0/toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
    ],
)
def test_header_opening_modes(  # noqa: PLR0913, PLR0917 -- fixtures and case matrix
    data_lmt_path, tmp_path, monkeypatch, mode, renamed, relative_path
):
    source = data_lmt_path / relative_path
    path = tmp_path / "downloaded.nc" if renamed else source
    if renamed:
        path.symlink_to(source)
    expected = guess_info_from_source(source)
    assert expected is not None
    should_open = mode is HeaderReadMode.always or (
        mode is HeaderReadMode.fallback and renamed
    )
    opened = []
    closed = []
    time_reads = []
    open_dataset = xr.open_dataset
    close_store = NetCDF4DataStore.close
    getitem = NetCDF4ArrayWrapper._getitem

    def open_data(*args, **kwargs):
        assert should_open, "Filename-only catalog opened a source"
        opened.append(args[0])
        return open_dataset(*args, **kwargs)

    def close(data, **kwargs):
        close_store(data, **kwargs)
        closed.append(data)

    def read(array, key):
        if array.variable_name.startswith("Data."):
            assert expected.interface == "tel"
            assert array.variable_name == "Data.TelescopeBackend.TelTime"
            assert key in ((0,), (array.shape[0] - 1,))
            time_reads.append(key)
        return getitem(array, key)

    monkeypatch.setattr(xr, "open_dataset", open_data)
    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read)
    catalog = build_input_catalog([path], header_mode=mode)
    record = catalog.toltec_file.to_info_list()[0]
    assert record.filepath == path
    assert len(opened) == len(closed) == int(should_open)
    assert len(time_reads) == (2 if should_open and expected.interface == "tel" else 0)
    if should_open:
        assert record.metadata is not None
        assert record.master == "tcs"
        assert record.filepath_orig is not None
        assert record.filepath_info_orig is not None
        assert record.filepath_info == record.filepath_info_orig
    else:
        assert record.metadata is None
        assert record.filepath_orig is None
    if renamed and not should_open:
        assert record.obsnum is None
        assert record.filepath_info is None
    else:
        assert record.obsnum == expected.obsnum
        assert record.data_kind == expected.data_kind


def test_default_catalog_enriches_all_reference_headers(data_lmt_path):
    paths = [
        *sorted((data_lmt_path / "tel").glob("*.nc")),
        *sorted((data_lmt_path / "toltec/tcs/toltec0").glob("*.nc")),
    ]
    assert len(paths) == 8
    catalog = build_input_catalog(paths)
    assert catalog["master"].tolist() == ["tcs"] * 8
    assert len(catalog.toltec_file.query("interface == 'tel' & master == 'tcs'")) == 4
    assert len(catalog.toltec_file.query("atten_drive == 7.25")) == 4
    assert len(catalog.toltec_file.query("obs_pgm == metadata.obs_pgm")) == 4
    for info in catalog.toltec_file.to_info_list():
        assert info.filepath_orig is not None
        assert info.filepath_info_orig is not None
        assert info.metadata is not None
        assert info.obsnum == info.filepath_info_orig.obsnum


def test_catalog_failure_closes_the_open_file(data_lmt_path, monkeypatch):
    from tolteca_datamodels.toltec import filestore

    source = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    closed = []
    close_store = NetCDF4DataStore.close

    def close(data, **kwargs):
        close_store(data, **kwargs)
        closed.append(data)

    def fail(*args, **kwargs):
        msg = "Test header failure"
        raise ValueError(msg)

    monkeypatch.setattr(NetCDF4DataStore, "close", close)
    monkeypatch.setattr(filestore, "identify_source", fail)
    with pytest.raises(ValueError, match="Test header failure"):
        build_input_catalog([source])
    assert len(closed) == 1


def test_invalid_header_mode_rejected_before_discovery():
    with pytest.raises(ValueError, match="HeaderReadMode"):
        build_input_catalog(["nonexistent.nc"], header_mode="unknown")
