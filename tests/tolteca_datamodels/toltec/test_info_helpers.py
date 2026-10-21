"""Single-row and batch file-information helpers share the factory behavior."""

import pandas as pd
import pytest
import xarray as xr

from tolteca_datamodels.toltec.file import (
    ToltecFileAccessor,
    get_info_from_source,
    guess_info_from_sources,
    guess_or_read_info_from_sources,
    read_info_from_sources,
)


@pytest.mark.parametrize(
    ("helper", "mode"),
    [
        (guess_info_from_sources, "guess"),
        (guess_or_read_info_from_sources, "guess_or_read"),
        (read_info_from_sources, "read"),
    ],
)
@pytest.mark.parametrize("on_error", ["raise", "record"])
def test_batch_helper_delegates_without_consuming_sources(
    monkeypatch, helper, mode, on_error
):
    sources = iter(["unidentified.nc"])
    expected = pd.DataFrame()

    def build(cls, supplied, **kwargs):
        assert supplied is sources
        assert kwargs == {"mode": mode, "on_error": on_error}
        return expected

    monkeypatch.setattr(ToltecFileAccessor, "from_sources", classmethod(build))
    assert helper(sources, on_error=on_error) is expected
    assert list(sources) == ["unidentified.nc"]


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_single_cached_source_matches_batch(data_lmt_path, tmp_path, mode):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    cache = tmp_path / "download.nc"
    cache.symlink_to(path)
    source_loc = "archive:/unknown.nc"
    row = get_info_from_source(cache, source_loc=source_loc, mode=mode)
    table = ToltecFileAccessor.from_sources(
        [{"source": cache, "source_loc": source_loc}], mode=mode
    )
    columns = [
        name for name in table if isinstance(name, str) and not name.startswith("_")
    ]
    pd.testing.assert_series_equal(
        pd.Series(row._asdict(), dtype=object).reindex(columns),
        table.iloc[0].reindex(columns),
        check_names=False,
        check_exact=True,
    )
    assert row.source is cache
    assert pd.isna(row.file_prefix)
    if mode == "guess":
        assert pd.isna(row.obsnum)
    else:
        assert row.obsnum == 145609
        assert row.obs_goal == "pointing"


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
def test_single_unstaged_remote_returns_partial_row(monkeypatch, mode):
    def unexpected(*args, **kwargs):
        pytest.fail("Unstaged remote information attempted I/O")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    source = "archive:/unidentified.nc"
    row = get_info_from_source(source, mode=mode)
    assert row.source is source
    assert pd.isna(row.obsnum)
    assert pd.isna(row.file_prefix)


def test_single_borrowed_source_remains_open(data_lmt_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    with xr.open_dataset(path) as data:
        row = get_info_from_source(data, source_loc=path, mode="read")
        assert row.source is data
        assert row.obs_goal == "pointing"
        assert data._close is not None
        assert data.lmt_tel.meta.obsnum == row.obsnum


def test_single_invalid_mode_precedes_source_validation():
    with pytest.raises(ValueError, match="literal_error"):
        get_info_from_source(object(), mode="invalid")  # ty: ignore[invalid-argument-type]


def test_single_local_read_error_propagates(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_info_from_source(tmp_path / "missing.nc", mode="read", on_error="raise")
