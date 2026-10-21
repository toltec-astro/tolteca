"""Format policies and typed opening ownership."""

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from tollan.utils.fileloc import fileloc
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.core.file import FileAccessor
from tolteca_datamodels.core.source import (
    SourceOpener,
    SourceReaderRules,
    source_opener,
)
from tolteca_datamodels.core.source import opener as source_module
from tolteca_datamodels.lmt import open_lmt_tel
from tolteca_datamodels.toltec import open_toltec_hwpr, open_toltec_kids
from tolteca_dummy_instru.data import open_dummy_instru


def test_opener_rules_use_projected_columns_and_location_fallback(monkeypatch):
    def specific(path):
        pytest.fail("Selecting an opener must not open the source")

    policy = SourceOpener(
        rules=(({"ext": "nc", "suffix": "tune"}, specific), *source_opener.rules)
    )
    monkeypatch.setattr(FileAccessor, "_source_opener", policy)
    table = pd.DataFrame(
        {
            "file_ext": pd.array(["nc", "nc", None, "nc.gz"], dtype="string"),
            "file_suffix": pd.array(["tune", None, None, "tune"], dtype="string"),
            "source_loc": ["a.nc", "b.nc", "archive:/renamed.ecsv", "c.nc.gz"],
            "data_kind": ["unrelated"] * 4,
        },
        index=[5, 3, 5, 1],
    )
    table["source"] = table.source_loc
    before = table.copy()
    choices = FileAccessor(table, accessor_types=())._get_opener_choices()
    assert [row[FileAccessor] for row in choices] == [
        specific,
        source_module._open_netcdf,
        source_module._open_ecsv,
        None,
    ]
    pd.testing.assert_frame_equal(table, before)


def test_scalar_open_accepts_externally_parsed_record_and_actual_cached_path(tmp_path):
    from tolteca_datamodels.filepath import FilepathInfo

    location = "archive:/toltec8_145605_000_0001_2025_10_31_09_20_06_tune.txt"
    info = FilepathInfo.from_source(location)
    policy = SourceOpener(rules=(({"ext": "txt", "suffix": "tune"}, Path.open),))
    cached = tmp_path / "cached-without-extension"
    cached.write_text("model parameters")
    with pytest.raises(ValueError, match="No source opener"):
        policy.open({}, source=cached)
    with policy.open(info, source=cached) as data:
        assert data.read() == "model parameters"


def test_missing_filename_attribute_is_not_an_explicit_null():
    policy = SourceOpener(rules=(({"ext": "nc", "suffix": None}, Path.open),))
    assert policy({"ext": "nc"}) is None
    assert policy({"ext": "nc", "suffix": None}) is Path.open
    assert (
        FileAccessor(
            FileAccessor.from_sources([], accessor_types=()), accessor_types=()
        )._get_opener_choices()
        == []
    )


@pytest.mark.parametrize("explicit_parent", [False, True])
def test_open_preserves_validated_fileloc_resolution(
    tmp_path, monkeypatch, explicit_parent
):
    monkeypatch.chdir(tmp_path)
    parent = Path("observations") if explicit_parent else None
    source = fileloc("input.nc", local_parent_path=parent)
    expected = (
        tmp_path / "observations/input.nc" if explicit_parent else tmp_path / "input.nc"
    )
    monkeypatch.chdir(tmp_path.parent)
    opened = []

    def opener(path, **kwargs):
        opened.append((path, kwargs))
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_dataset", opener)
    with source_opener.open({"ext": "nc"}, source=source, chunks={}):
        pass
    assert opened == [(expected, {"chunks": {}})]
    assert source.root.local_parent_path == parent


@pytest.mark.parametrize("source", ["", "ftp://example.org/input.nc"])
def test_invalid_source_does_not_reach_opener(source, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("An invalid location reached the opener")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    with pytest.raises(ValueError):
        source_opener.open({"ext": "nc"}, source=source)


@pytest.mark.parametrize("source_factory", [str, fileloc])
@pytest.mark.parametrize(
    "source",
    [
        "archive:/data/input.nc",
        "file://archive/data/input.nc",
        "https://example.org/input.nc",
        "http://localhost/input.nc",
    ],
)
def test_default_policy_rejects_remote_locations_before_opening(
    source_factory, source, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("Remote location was passed to the opener")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    with pytest.raises(ValueError, match="staged locally"):
        source_opener.open({"ext": "nc"}, source=source_factory(source))


@pytest.mark.parametrize(
    "opener", [open_lmt_tel, open_toltec_kids, open_toltec_hwpr, open_dummy_instru]
)
@pytest.mark.parametrize("source_factory", [str, fileloc])
def test_concrete_openers_reject_remote_locations_before_local_backend(
    opener, source_factory, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("Remote location was passed to the local backend")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    with pytest.raises(ValueError, match="staged locally"):
        opener(source=source_factory("file://archive/data/input.nc"))


@pytest.mark.parametrize(
    "opener", [open_lmt_tel, open_toltec_kids, open_toltec_hwpr, open_dummy_instru]
)
def test_concrete_openers_close_unrecognized_datasets(opener, monkeypatch):
    data = xr.Dataset()
    closed = []
    data.set_close(lambda: closed.append(True))
    monkeypatch.setattr(xr, "open_dataset", lambda *args, **kwargs: data)
    with pytest.raises(ValueError):
        opener("unrecognized.nc")
    assert closed == [True]


@pytest.mark.parametrize("family", ["lmt", "toltec", "dummy_instru"])
@pytest.mark.parametrize(
    "source_factory",
    [str, Path, Path.as_uri, fileloc],
    ids=["str", "path", "url", "fileloc"],
)
def test_concrete_openers_preserve_identity_options_and_lazy_reads(
    family, source_factory, data_lmt_path, monkeypatch
):
    def unexpected_selection(*args, **kwargs):
        pytest.fail("A typed opener must bind its known reader without selection")

    monkeypatch.setattr(SourceReaderRules, "identify", unexpected_selection)
    monkeypatch.setattr(SourceReaderRules, "__call__", unexpected_selection)
    if family == "lmt":
        path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
        data = xr.open_dataset(path)
        opener, accessor_name = open_lmt_tel, "lmt_tel"
    elif family == "toltec":
        path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
        data = xr.open_dataset(path)
        opener, accessor_name = (
            open_toltec_kids,
            "toltec_kids",
        )
    else:
        # Deliberately incomplete headers verify that opening only identifies.
        path = data_lmt_path / "missing-dummy.nc"
        data = xr.Dataset(attrs={"instru": "dummy_instru"})
        opener, accessor_name = (
            open_dummy_instru,
            "dummy_instru",
        )
    opened, closed, reads = [], [], []
    close = data._close
    getitem = NetCDF4ArrayWrapper._getitem

    def track_close():
        closed.append(True)
        if close is not None:
            close()

    def track_open(source, **kwargs):
        opened.append((source, kwargs))
        return data

    def read_classification_only(array, key):
        assert family == "toltec" and array.variable_name == "Header.Toltec.ObsType"
        reads.append(array.variable_name)
        return getitem(array, key)

    data.set_close(track_close)
    with data:
        monkeypatch.setattr(xr, "open_dataset", track_open)
        monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", read_classification_only)
        with opener(source=source_factory(path), chunks={}) as result:
            assert result is data
            reader = getattr(result, accessor_name)
            assert getattr(result, accessor_name) is reader
            assert "meta" not in reader.__dict__
            assert not closed
    assert closed == [True]
    options = {"chunks": {}}
    if family == "toltec":
        options.update(decode_coords="all", decode_times=True)
    assert opened == [(path, options)]
    assert reads == (["Header.Toltec.ObsType"] if family == "toltec" else [])


def test_toltec_open_preserves_caller_decoding_options(data_lmt_path, monkeypatch):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145606*_targsweep.nc"))
    opened = []
    with xr.open_dataset(path) as data:

        def track_open(source, **kwargs):
            opened.append((source, kwargs))
            return data

        monkeypatch.setattr(xr, "open_dataset", track_open)
        with open_toltec_kids(path, decode_coords=False, decode_times=False):
            pass
    assert opened == [(path, {"decode_coords": False, "decode_times": False})]
