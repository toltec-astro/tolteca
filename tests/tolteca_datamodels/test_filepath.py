"""Universal filename information and specialized acquisition layouts."""

from typing import Annotated

import pytest
from pydantic import ValidationError

from tolteca_datamodels.core.filepath import (
    FilepathInfo,
    MetadataField,
    register_filepath_parser,
)


def test_metadata_marker_is_inherited_without_changing_field_values():
    class Reading(FilepathInfo):
        value: Annotated[int, MetadataField]

    class Inherited(Reading):
        pass

    assert Reading.model_fields["value"].metadata == [MetadataField]
    assert Inherited.model_fields["value"].metadata[0] is MetadataField
    data = {"ext": "nc", "data_models": frozenset(), "value": 1}
    assert Reading.model_validate(data).model_dump() == data
    with pytest.raises(ValidationError, match="valid integer"):
        Reading.model_validate(data | {"value": "1"})
    assert Reading.model_json_schema()["properties"]["value"]["type"] == "integer"


@pytest.mark.parametrize(
    ("source", "ext"),
    [
        ("foo.nc", "nc"),
        ("archive:/foo.ecsv", "ecsv"),
        ("https://archive.example/foo.nc", "nc"),
        ("foo.tar.gz", "gz"),
        ("foo", ""),
        (".hidden", ""),
    ],
)
def test_generic_filenames_provide_format_without_claiming_compatibility(source, ext):
    info = FilepathInfo.from_source(source)
    assert type(info) is FilepathInfo
    assert info.ext == ext
    assert info.data_models == frozenset()


def test_specialized_layout_keeps_its_extension_and_validation():
    from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo

    info = FilepathInfo.from_source(
        "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.tar.gz"
    )
    assert type(info) is ToltecFilepathInfo
    assert info.ext == "tar.gz"
    assert info.data_models == {"toltec"}
    with pytest.raises(ValueError):
        FilepathInfo.from_source("toltec8_145605_000_0001_2025_99_31_09_20_06_tune.nc")


def test_location_without_filename_can_be_reported_or_suppressed():
    with pytest.raises(ValueError, match="No filename parser matches"):
        FilepathInfo.from_source("/")
    assert FilepathInfo.from_source("/", raise_on_error=False) is None


def test_basic_filename_information_comes_from_a_registration(monkeypatch):
    from tolteca_datamodels.core import filepath

    registered = filepath._filepath_parsers
    monkeypatch.setattr(
        filepath,
        "_filepath_parsers",
        [entry for entry in registered if entry.filepath_info_cls is not FilepathInfo],
    )
    assert FilepathInfo.from_source("foo.nc", raise_on_error=False) is None
    monkeypatch.setattr(filepath, "_filepath_parsers", registered)
    first = FilepathInfo.from_source("foo.nc")
    second = FilepathInfo.from_source("bar.ecsv")
    assert first is not None and second is not None
    assert (first.ext, second.ext) == ("nc", "ecsv")
    assert first.data_models == second.data_models == frozenset()


@pytest.mark.parametrize("reverse", [False, True])
def test_factory_limits_each_call_to_self_and_descendants(monkeypatch, reverse):
    from tolteca_datamodels.core import filepath

    monkeypatch.setattr(
        filepath, "_filepath_parsers", filepath._filepath_parsers.copy()
    )

    class A(FilepathInfo):
        number: int

    class B(A):
        pass

    class C(B):
        pass

    class D(C):
        pass

    models = (A, B, C, D)
    converted = []

    def convert(captures, path):
        converted.append(path.name)
        return {"data_models": []}

    entries = list(enumerate(models))
    for level, model in reversed(entries) if reverse else entries:
        register_filepath_parser(
            pattern=rf"^chain-(?P<number>[{level}-3])\.(?P<ext>nc)$",
            parsers=(convert,),
            filepath_info_cls=model,
        )

    for number, expected in enumerate(models):
        source = f"chain-{number}.nc"
        assert type(FilepathInfo.from_source(source)) is expected
        for level, caller in enumerate(models):
            converted.clear()
            if level <= number:
                result = caller.from_source(source)
                assert result is not None
                assert type(result) is expected
                assert result.number == number
                assert converted == [source]
            else:
                assert caller.from_source(source, raise_on_error=False) is None
                with pytest.raises(ValueError, match=f"matches {caller.__name__}"):
                    caller.from_source(source)
                assert converted == []

    class Sibling(FilepathInfo):
        number: int

    register_filepath_parser(
        pattern=r"^chain-(?P<number>3)\.(?P<ext>nc)$",
        parsers=(convert,),
        filepath_info_cls=Sibling,
    )
    assert type(A.from_source("chain-3.nc")) is D
    assert type(Sibling.from_source("chain-3.nc")) is Sibling
    # The universal factory still rejects unrelated overlapping results.
    with pytest.raises(RuntimeError, match="Ambiguous acquisition filename"):
        FilepathInfo.from_source("chain-3.nc", raise_on_error=False)


def test_concrete_factory_does_not_fall_back_to_ancestor_or_sibling():
    from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
    from tolteca_datamodels.toltec.filepath import (
        ToltecFilepathInfo,
        ToltecLmtFilepathInfo,
    )

    tel = "tel_toltec_2025-10-31_145609_00_0002.nc"
    other_tel = "tel_other_2025-10-31_145609_00_0002.nc"
    kids = "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc"
    assert type(LmtFilepathInfo.from_source(tel)) is ToltecLmtFilepathInfo
    assert type(LmtFilepathInfo.from_source(other_tel)) is LmtFilepathInfo
    assert type(ToltecFilepathInfo.from_source(kids)) is ToltecFilepathInfo
    for caller, source in (
        (LmtFilepathInfo, "renamed.nc"),
        (LmtFilepathInfo, kids),
        (ToltecFilepathInfo, tel),
        (ToltecLmtFilepathInfo, other_tel),
    ):
        assert caller.from_source(source, raise_on_error=False) is None
        with pytest.raises(ValueError, match=f"matches {caller.__name__}"):
            caller.from_source(source)

    with pytest.raises(ValueError, match="does not match format"):
        LmtFilepathInfo.from_source(
            tel.replace("2025-10-31", "2025-99-31"), raise_on_error=False
        )
