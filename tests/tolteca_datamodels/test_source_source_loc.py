"""Standalone source preparation, rule opening, and resource lifetime."""

import subprocess
import sys
from io import StringIO

import pytest
import xarray as xr
from astropy.table import QTable
from pydantic import ValidationError

from tolteca_datamodels.core.file import SourceSourceLoc
from tolteca_datamodels.core.filepath import FilepathInfo
from tolteca_datamodels.core.source import SourceOpener


def test_pair_validation_is_strict_and_location_is_authoritative():
    resource = object()
    with pytest.raises(ValidationError):
        SourceSourceLoc.model_validate(resource)
    pair = {"source": resource, "source_loc": "archive:/unrecognized.nc"}
    item = SourceSourceLoc.model_validate(pair)
    assert item.source is resource
    assert item.source_loc_host == "archive"
    assert not item.source_loc_is_local
    assert type(item.filepath_info) is FilepathInfo
    assert item.filepath_info.ext == "nc"
    assert item.filepath_info.data_models == frozenset()
    assert not hasattr(item, "source_trace")
    assert pair == {"source": resource, "source_loc": "archive:/unrecognized.nc"}
    with pytest.raises(ValidationError, match="frozen"):
        item.source = "replacement"  # ty: ignore[invalid-assignment]


def test_parsing_is_eager_and_prepared_instances_do_not_reparse(monkeypatch):
    parsed = FilepathInfo(ext="nc", data_models=frozenset())
    locations = []

    def parse(location, *, raise_on_error):
        locations.append(location)
        assert not raise_on_error
        return parsed

    monkeypatch.setattr(FilepathInfo, "from_source", parse)
    item = SourceSourceLoc.model_validate("local.nc")
    assert locations == [item.source_loc]
    assert item.filepath_info is parsed
    assert item.filepath_info.data_models == frozenset()
    assert SourceSourceLoc.model_validate(item) is item
    assert len(locations) == 1


def test_filename_parse_failure_raises_for_standalone_use():
    from tolteca_datamodels.lmt import filepath  # noqa: F401 -- load telescope parser

    with pytest.raises(ValidationError, match="does not match format"):
        SourceSourceLoc.model_validate("tel_toltec_2025-99-31_145609_00_0002.nc")


def test_open_uses_parsed_record_but_passes_original_source_and_owns_resource():
    from tolteca_datamodels.toltec import filepath  # noqa: F401 -- load TolTEC parser

    calls = []

    def acquire(source, **kwargs):
        calls.append((source, kwargs))
        return StringIO("contents")

    opener = SourceOpener(rules=(({"ext": "nc", "interface": "toltec8"}, acquire),))

    item = SourceSourceLoc.model_validate(
        {
            "source": "cached-resource",
            "source_loc": "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
        }
    )
    context = item.open(opener=opener, option=1)
    assert calls == []
    with context as resource:
        assert calls == [("cached-resource", {"option": 1})]
        assert resource.read() == "contents"
    assert resource.closed


def test_open_closes_owned_objects_and_propagates_body_failure():
    class Resource:
        closed = False

        def close(self):
            self.closed = True

    resource = Resource()

    opener = SourceOpener(rules=(({}, lambda source: resource),))

    item = SourceSourceLoc.model_validate("unrecognized.nc")

    def use_resource():
        with item.open(opener=opener) as opened:
            assert opened is resource
            msg = "caller failed"
            raise ValueError(msg)

    with pytest.raises(ValueError, match="caller failed"):
        use_resource()
    assert resource.closed


def test_borrowed_object_is_not_reopened_or_closed():
    opener = SourceOpener(rules=())

    with StringIO("contents") as resource:
        item = SourceSourceLoc.model_validate(
            {"source": resource, "source_loc": "renamed.nc"}
        )
        with item.open(opener=opener) as opened:
            assert opened is resource
        assert not resource.closed


@pytest.mark.parametrize("suffix", ["nc", "ecsv"])
def test_default_opener_handles_unrecognized_local_filenames(tmp_path, suffix):
    path = tmp_path / f"renamed.{suffix}"
    if suffix == "nc":
        xr.Dataset({"signal": ("sample", [1, 2])}).to_netcdf(path)
    else:
        QTable({"signal": [1, 2]}).write(path, format="ascii.ecsv")
    item = SourceSourceLoc.model_validate(path)
    assert type(item.filepath_info) is FilepathInfo
    assert item.filepath_info.ext == suffix
    assert item.filepath_info.data_models == frozenset()
    with item.open() as opened:
        assert list(opened["signal"]) == [1, 2]


@pytest.mark.parametrize("failure", ["selection", "opening"])
def test_open_failures_raise(failure):
    def broken(source):
        msg = "missing resource"
        raise OSError(msg)

    opener = SourceOpener(rules=() if failure == "selection" else (({}, broken),))

    item = SourceSourceLoc.model_validate("unrecognized.nc")
    with (
        pytest.raises(ValueError if failure == "selection" else OSError),
        item.open(opener=opener),
    ):
        pytest.fail("Failed opening yielded an object")


def test_root_entry_loads_builtins_and_core_uses_existing_registrations():
    script = """
import sys
from tolteca_datamodels.core.file import SourceSourceLoc

source = "tel_toltec_2025-10-31_145609_00_0002.nc"
info = SourceSourceLoc.model_validate(source).filepath_info
assert info.ext == "nc" and info.data_models == frozenset()
assert "tolteca_datamodels.lmt" not in sys.modules
assert "tolteca_datamodels.toltec" not in sys.modules
from tolteca_datamodels.source import SourceSourceLoc as Assembled
assert Assembled is SourceSourceLoc
assert Assembled.model_validate(source).filepath_info.data_models == {'lmt', 'toltec'}
assert "tolteca_datamodels.core.data_model" not in sys.modules
"""
    subprocess.run(  # noqa: S603 -- current interpreter and literal test code
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
