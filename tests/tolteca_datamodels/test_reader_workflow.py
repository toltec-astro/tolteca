"""Shared identification policy, opening ownership and concrete entry points."""

from io import StringIO
from pathlib import Path

import pytest
import xarray as xr
from tollan.utils.fileloc import fileloc
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.lmt import identify_lmt_tel, open_lmt_tel
from tolteca_datamodels.source import identify_reader, open_source
from tolteca_datamodels.toltec import open_toltec
from tolteca_datamodels.toltec.kids.core import identify_toltec_kids
from tolteca_dummy_instru.data import identify_dummy_instru, open_dummy_instru


class _Reader:
    def __init__(self, payload):
        self.payload = payload

    def read_metadata(self):
        pytest.fail("Identification or opening read full metadata")


@pytest.mark.parametrize("strict", [False, True])
def test_shared_identification_preserves_payload_and_defers_metadata(strict):
    payload = xr.Dataset()
    reader = identify_reader(
        payload,
        source_types=(xr.Dataset,),
        reader_factory=_Reader,
        raise_on_error=strict,
    )
    assert reader.payload is payload


def test_unsupported_type_does_not_invoke_factory():
    def unexpected(data):
        pytest.fail("Unsupported payload reached the factory")

    options = {"source_types": (xr.Dataset,), "reader_factory": unexpected}
    assert identify_reader("file.nc", **options) is None
    with pytest.raises(TypeError, match="Expected Dataset, got str"):
        identify_reader("file.nc", raise_on_error=True, **options)


def test_value_error_policy_preserves_original_exception_in_strict_mode():
    error = ValueError("Unsupported layout")

    def reject(data):
        raise error

    options = {"source_types": (xr.Dataset,), "reader_factory": reject}
    assert identify_reader(xr.Dataset(), **options) is None
    with pytest.raises(ValueError) as exc:
        identify_reader(xr.Dataset(), raise_on_error=True, **options)
    assert exc.value is error


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("error_type", [TypeError, KeyError, RuntimeError, OSError])
def test_other_factory_errors_always_propagate(strict, error_type):
    error = error_type("Reader failure")

    def fail(data):
        raise error

    with pytest.raises(error_type) as exc:
        identify_reader(
            xr.Dataset(),
            source_types=(xr.Dataset,),
            reader_factory=fail,
            raise_on_error=strict,
        )
    assert exc.value is error


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, KeyboardInterrupt])
def test_shared_open_closes_on_identification_failure(error_type):
    data = xr.Dataset()
    closed = []
    data.set_close(lambda: closed.append(True))
    error = error_type("Identification failed")

    def identify(payload, *, raise_on_error=False):
        assert payload is data
        assert raise_on_error is True
        raise error

    with pytest.raises(error_type) as exc:
        open_source("input.nc", opener=lambda _: data, identifier=identify)
    assert exc.value is error
    assert closed == [True]


def test_shared_open_does_not_identify_after_backend_failure():
    error = OSError("Backend failure")

    def fail(source):
        raise error

    def unexpected(*args, **kwargs):
        pytest.fail("Identification followed an unsuccessful open")

    with pytest.raises(OSError) as exc:
        open_source("input.nc", opener=fail, identifier=unexpected)
    assert exc.value is error


def test_shared_open_without_identifier_leaves_resource_to_caller():
    data = xr.Dataset()
    closed = []
    data.set_close(lambda: closed.append(True))
    with open_source("unrecognized.nc", opener=lambda _: data) as opened:
        assert opened is data
        assert not closed
    assert closed == [True]


def test_shared_open_supports_non_xarray_resources(tmp_path):
    path = tmp_path / "observation.txt"
    path.write_text("observation\n")
    with open_source(path, opener=Path.open, encoding="utf-8") as opened:
        assert opened.read() == "observation\n"
    assert opened.closed


def test_shared_open_closes_non_xarray_resource_on_identification_failure():
    data = StringIO("unrecognized")

    def reject(payload, *, raise_on_error):
        assert payload is data and raise_on_error
        msg = "Unsupported text"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="Unsupported text"):
        open_source("input.txt", opener=lambda _: data, identifier=reject)
    assert data.closed


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

    def identify(data, *, raise_on_error):
        assert raise_on_error is True
        return _Reader(data)

    with open_source(source, opener=opener, identifier=identify, chunks={}):
        pass
    assert opened == [(expected, {"chunks": {}})]
    assert source.root.local_parent_path == parent


@pytest.mark.parametrize("source", ["", "ftp://example.org/input.nc"])
def test_invalid_source_does_not_reach_opener_or_identifier(source):
    def unexpected(*args, **kwargs):
        pytest.fail("An invalid location reached the opener or identifier")

    with pytest.raises(ValueError):
        open_source(source, opener=unexpected, identifier=unexpected)


@pytest.mark.parametrize("opener", [open_lmt_tel, open_toltec, open_dummy_instru])
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
def test_remote_locations_do_not_reach_local_backend(
    opener, source_factory, source, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("Remote location was passed to the local backend")

    monkeypatch.setattr(xr, "open_dataset", unexpected)
    with pytest.raises(ValueError, match="staged locally"):
        opener(source=source_factory(source))


@pytest.mark.parametrize(
    ("opener", "identifier"),
    [
        (open_lmt_tel, identify_lmt_tel),
        (open_toltec, identify_toltec_kids),
        (open_dummy_instru, identify_dummy_instru),
    ],
)
def test_concrete_openers_close_unrecognized_datasets(opener, identifier, monkeypatch):
    data = xr.Dataset()
    closed = []
    data.set_close(lambda: closed.append(True))
    assert identifier(data) is None
    assert not closed
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
    if family == "lmt":
        path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
        data = xr.open_dataset(path)
        opener, identifier, accessor_name = open_lmt_tel, identify_lmt_tel, "lmt_tel"
    elif family == "toltec":
        path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_tune.nc"))
        data = xr.open_dataset(path)
        opener, identifier, accessor_name = (
            open_toltec,
            identify_toltec_kids,
            "toltec_kids",
        )
    else:
        # Deliberately incomplete headers verify that opening only identifies.
        path = data_lmt_path / "missing-dummy.nc"
        data = xr.Dataset(attrs={"instru": "dummy_instru"})
        opener, identifier, accessor_name = (
            open_dummy_instru,
            identify_dummy_instru,
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
            assert identifier(result, raise_on_error=True) is reader
            assert "meta" not in reader.__dict__
            assert not closed
    assert closed == [True]
    options = {"chunks": {}}
    if family == "toltec":
        options.update(decode_coords="all", decode_times=True)
    assert opened == [(path, options)]
    assert reads == (["Header.Toltec.ObsType"] if family == "toltec" else [])


def test_toltec_open_preserves_caller_decoding_options(data_lmt_path, monkeypatch):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*_targsweep.nc"))
    opened = []
    with xr.open_dataset(path) as data:

        def track_open(source, **kwargs):
            opened.append((source, kwargs))
            return data

        monkeypatch.setattr(xr, "open_dataset", track_open)
        with open_toltec(path, decode_coords=False, decode_times=False):
            pass
    assert opened == [(path, {"decode_coords": False, "decode_times": False})]
