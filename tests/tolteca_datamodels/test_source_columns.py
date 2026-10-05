"""Resource identity and explicit filename locations in file tables."""

from io import BytesIO
from typing import ClassVar

import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError
from tollan.utils.fileloc import FileLoc, fileloc

from tolteca_datamodels.file import FileAccessor, FilepathInfo, _SourceSourceLoc
from tolteca_datamodels.model_table import ModelColumn
from tolteca_datamodels.toltec.file import ToltecFileAccessor

_SOURCE_COLUMN_DTYPES = {
    "source": "object",
    "source_loc": "object",
    "source_loc_is_local": "boolean",
    "source_loc_host": "string",
    "source_loc_scheme": "string",
}


@pytest.mark.parametrize("mode", ["guess", "guess_or_read", "read"])
@pytest.mark.parametrize("sources", [[], ["unknown.nc"]])
def test_base_factory_only_records_resources(monkeypatch, sources, mode):
    def unexpected(*args, **kwargs):
        pytest.fail("Source-only accessor attempted filename parsing or I/O")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    monkeypatch.setattr(xr, "open_dataset", unexpected)
    monkeypatch.setattr(FileAccessor, "_source_opener", xr.open_dataset)
    monkeypatch.setattr(FileLoc, "exists", unexpected)
    table = FileAccessor.from_sources(sources, mode=mode)
    assert list(table) == list(_SOURCE_COLUMN_DTYPES)
    assert table.dtypes.astype(str).to_dict() == _SOURCE_COLUMN_DTYPES
    assert table.source.tolist() == sources


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("guess", []), ("guess_or_read", [1, 3]), ("read", [0, 1, 2, 3])],
)
def test_shared_factory_selects_metadata_rows_after_guessing(mode, expected):
    read_positions = []
    sources = ["first.nc", "cached.nc", "first.nc", "other.nc"]

    class Files(FileAccessor):
        def _guess_columns(self, *, diagnostics):
            self._obj["value"] = [10, 20, 30, 40]
            return {1, 3}

        def _read_metadata(self, rows, *, diagnostics):
            for position, row in rows:
                read_positions.append(position)
                assert row.source == sources[position]
                assert row.value == (position + 1) * 10
                self._obj.iloc[position, list(self._obj).index("value")] = -1

    table = Files.from_sources(iter(sources), mode=mode)
    assert read_positions == expected
    assert table.source.tolist() == sources
    assert table.value.tolist() == [
        -1 if position in expected else (position + 1) * 10
        for position in range(len(sources))
    ]


@pytest.mark.parametrize("inplace", [False, True])
def test_shared_metadata_wrapper_preserves_positions_and_copy_contract(inplace):
    table = FileAccessor.from_sources(["first.nc", "second.nc", "first.nc"])
    table.index = pd.Index([8, 3, 8], name="input_row")
    table.attrs["batch"] = "observations"
    before = table.copy()
    instances, rows_read = [], []

    class Files(FileAccessor):
        def __init__(self, data):
            super().__init__(data)
            instances.append(self)

        def _read_metadata(self, rows, *, diagnostics):
            rows_read.extend(rows)
            self._obj["ordinal"] = [position for position, _ in rows_read]

    accessor = Files(table)
    result = accessor.read_metadata(inplace=inplace)
    if inplace:
        assert result is None
        assert instances == [accessor]
        result = table
    else:
        assert result is not table
        assert len(instances) == 2
        pd.testing.assert_frame_equal(table, before)
    pd.testing.assert_frame_equal(result[before.columns], before)
    assert result.attrs == before.attrs
    assert result.ordinal.tolist() == [0, 1, 2]
    assert [row.source for _, row in rows_read] == before.source.tolist()


@pytest.mark.parametrize("mode", ["unknown", None, 1, b"read"])
def test_factory_invalid_metadata_policy_rejected_before_input(mode):
    def sources():
        pytest.fail("Invalid read policy consumed the sources")
        yield

    with pytest.raises(ValueError, match="literal_error"):
        FileAccessor.from_sources(sources(), mode=mode)


def test_single_iterable_preserves_sources_and_validates_locations(tmp_path):
    sources = [
        "relative.nc",
        tmp_path / "cached.nc",
        fileloc("archive:/original.nc"),
    ]
    table = FileAccessor.from_sources(iter(sources))
    accessor = FileAccessor(table)
    assert list(table) == list(_SOURCE_COLUMN_DTYPES)
    assert table.dtypes.astype(str).to_dict() == _SOURCE_COLUMN_DTYPES
    for row, source in zip(table.itertuples(index=False), sources, strict=True):
        assert row.source is source
        assert isinstance(row.source_loc, FileLoc)
        assert row.source_loc == fileloc(source)
    assert accessor.source.iloc[2] is sources[2]
    assert accessor.source_loc.iloc[2] is sources[2]
    updated = fileloc("updated.nc")
    table["source_loc"] = pd.Series([updated] * len(table), dtype=object)
    assert accessor.source_loc.iloc[0] is updated


@pytest.mark.parametrize("explicit_none", [False, True])
def test_pair_validation_does_not_mutate_input_or_copy_fileloc(tmp_path, explicit_none):
    source = fileloc("relative.nc", local_parent_path=tmp_path)
    entry = {"source": source}
    if explicit_none:
        entry["source_loc"] = None
    before = entry.copy()
    table = FileAccessor.from_sources([entry])
    assert entry == before
    assert table.source.iloc[0] is source
    assert table.source_loc.iloc[0] is source
    assert table.source.iloc[0].root.local_parent_path == tmp_path


def test_mixed_sources_and_explicit_locations_keep_pair_alignment():
    resource = BytesIO(b"example")
    location = fileloc("archive:/original.nc")
    entries = ["first.nc", {"source": resource, "source_loc": location}, "last.nc"]
    table = FileAccessor.from_sources(entry for entry in entries)
    table.index = [4, 4, 2]
    selected = FileAccessor(table.iloc[[1, 0]])
    assert selected.source.iloc[0] is resource
    assert selected.source_loc.iloc[0] is location
    assert selected.source.iloc[1] is entries[0]
    assert selected.source_loc.iloc[1] == fileloc(entries[0])
    assert not resource.closed


@pytest.mark.parametrize("entry_form", ["bare", "missing", "none"])
def test_open_handle_requires_explicit_location(entry_form):
    with BytesIO(b"example") as resource:
        entry = {
            "bare": resource,
            "missing": {"source": resource},
            "none": {"source": resource, "source_loc": None},
        }[entry_form]
        with pytest.raises(TypeError, match="unknown file loc data type"):
            FileAccessor.from_sources([entry])
        assert not resource.closed


def test_open_handle_accepts_location_string():
    with BytesIO(b"example") as resource:
        table = FileAccessor.from_sources(
            [{"source": resource, "source_loc": "archive:/original.nc"}]
        )
        row = next(table.itertuples(index=False))
        assert row.source is resource
        assert isinstance(row.source_loc, FileLoc)
        assert row.source_loc.as_rsync_arg() == "archive:/original.nc"
        assert table.dtypes.astype(str).to_dict() == _SOURCE_COLUMN_DTYPES
        assert not resource.closed


def test_source_location_columns_and_queries(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Location projection attempted file access")

    monkeypatch.setattr(FileLoc, "exists", unexpected)
    monkeypatch.setattr(xr, "open_dataset", unexpected)
    monkeypatch.setattr(ToltecFileAccessor, "_source_opener", xr.open_dataset)
    locations = [
        "relative.nc",
        "file://localhost/data/local.nc",
        "archive:/data/remote.nc",
        "https://archive:8443/data/remote.nc",
        "http://localhost/data/remote.nc",
    ]
    table = FileAccessor.from_sources(locations)
    assert table.source_loc_is_local.tolist() == [True, True, False, False, False]
    assert table.source_loc_host.tolist() == ["", "", "archive", "archive", "localhost"]
    assert table.source_loc_scheme.tolist() == ["file", "file", "file", "https", "http"]
    accessor = FileAccessor(table)
    assert accessor.query("source_loc_is_local").index.tolist() == [0, 1]
    assert accessor.query(
        'not source_loc_is_local and source_loc_host == "archive"'
    ).index.tolist() == [2, 3]
    assert accessor.query('source_loc_scheme == "https"').index.tolist() == [3]


@pytest.mark.parametrize("borrowed", [False, True])
def test_location_columns_describe_identification_not_opening_source(
    tmp_path, borrowed
):
    with BytesIO(b"example") as handle:
        source = handle if borrowed else tmp_path / "cached.nc"
        location = fileloc("archive:/data/original.nc")
        table = FileAccessor.from_sources([{"source": source, "source_loc": location}])
        row = next(table.itertuples(index=False))
        assert row.source is source
        assert row.source_loc is location
        assert not row.source_loc_is_local
        assert row.source_loc_host == "archive"
        assert row.source_loc_scheme == "file"
        assert not handle.closed


def test_invalid_location_does_not_fall_back_to_valid_source():
    with pytest.raises(TypeError, match="unknown file loc data type"):
        FileAccessor.from_sources([{"source": "valid.nc", "source_loc": object()}])


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {},
        {"source_loc": "unknown.nc"},
        {"source": None},
        {"source": "file.nc", "misspelled_location": "other.nc"},
    ],
)
def test_invalid_pairs_fail_at_construction(entry):
    with pytest.raises(ValidationError):
        FileAccessor.from_sources([entry])


def test_all_pairs_are_validated_before_filename_projection(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Filename projection ran before pair validation completed")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    with pytest.raises(ValidationError):
        ToltecFileAccessor.from_sources(["file.nc", {"source_loc": "missing.nc"}])


def test_builtin_columns_do_not_depend_on_subclass_projection():
    class Files(FileAccessor):
        columns: ClassVar = {"other": ModelColumn("other")}

    for sources in ([], ["file.nc"]):
        table = Files.from_sources(sources)
        assert list(table) == list(_SOURCE_COLUMN_DTYPES)
        assert table.dtypes.astype(str).to_dict() == _SOURCE_COLUMN_DTYPES


def test_builtin_column_names_and_removed_path_conversion():
    table = FileAccessor.from_sources(["example.nc"])
    assert list(table) == list(_SOURCE_COLUMN_DTYPES)
    assert list(FileAccessor._source_columns) == list(table)
    assert ToltecFileAccessor.IO_OBJ_COLUMN == "io_obj"
    assert not hasattr(FileAccessor, "filepaths")


def test_validated_source_input_projects_attributes_without_serializing(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Source projection attempted model serialization")

    monkeypatch.setattr(_SourceSourceLoc, "model_dump", unexpected)
    monkeypatch.setattr(FileLoc, "model_dump", unexpected)
    with BytesIO(b"example") as source:
        location = fileloc("archive:/original.nc")
        model = _SourceSourceLoc(source=source, source_loc=location)
        assert not model.source_loc_is_local
        assert model.source_loc_host == "archive"
        assert model.source_loc_scheme == "file"

        table = FileAccessor.from_sources([model])
        row = next(table.itertuples(index=False))
        assert row.source is source
        assert row.source_loc is location
        assert row.source_loc_is_local == model.source_loc_is_local
        assert row.source_loc_host == model.source_loc_host
        assert row.source_loc_scheme == model.source_loc_scheme
        assert table.dtypes.astype(str).to_dict() == _SOURCE_COLUMN_DTYPES
        assert not source.closed


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("source_loc_is_local", True),
        ("source_loc_host", "different-host"),
        ("source_loc_scheme", "https"),
    ],
)
def test_location_query_attributes_cannot_be_supplied_or_assigned(name, value):
    data = {"source": "archive:/original.nc"}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FileAccessor.from_sources([data | {name: value}])
    model = _SourceSourceLoc.model_validate(data)
    with pytest.raises(AttributeError):
        setattr(model, name, value)


def test_source_loc_is_the_only_filename_input(monkeypatch):
    source = fileloc("tel_toltec_2025-10-31_145609_00_0002.nc")
    location = fileloc("archive:/unrecognized.nc")
    calls = []
    parse = FilepathInfo.from_source

    def trace(value, **kwargs):
        calls.append(value)
        return parse(value, **kwargs)

    monkeypatch.setattr(FilepathInfo, "from_source", trace)
    table = ToltecFileAccessor.from_sources(
        [
            {"source": source, "source_loc": location},
            source,
        ]
    )
    assert calls[0] is location and calls[1] is source
    assert pd.isna(table.iloc[0].obsnum)
    assert table.iloc[1].obsnum == 145609


@pytest.mark.parametrize("as_tree", [False, True])
def test_open_sources_are_borrowed_and_identified_by_source_loc(
    data_lmt_path,
    monkeypatch,
    as_tree,
):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    location = fileloc(f"archive:/data_lmt/tel/{path.name}")
    with xr.open_dataset(path) as raw:
        source = xr.DataTree(raw) if as_tree else raw
        table = ToltecFileAccessor.from_sources(
            [
                {"source": source, "source_loc": location},
            ]
        )
        assert table.source.iloc[0] is source
        assert table.source_loc.iloc[0] is location
        assert table.file_prefix.iloc[0] == "tel_toltec"
        location_columns = table.filter(regex=r"^source_loc_")

        def unexpected(*args, **kwargs):
            pytest.fail("A borrowed source was opened or closed by the accessor")

        with monkeypatch.context() as patch:
            patch.setattr(xr, "open_dataset", unexpected)
            patch.setattr(ToltecFileAccessor, "_source_opener", xr.open_dataset)
            patch.setattr(xr.Dataset, "close", unexpected)
            patch.setattr(xr.DataTree, "close", unexpected)
            enriched = table.toltec_file.read_metadata()
            assert enriched.source.iloc[0] is source
            assert enriched.source_loc.iloc[0] is location
            pd.testing.assert_frame_equal(
                enriched[location_columns.columns], location_columns
            )
            assert enriched.obsnum.iloc[0] == raw.lmt_tel.meta.obsnum
            with table.toltec_file.open() as opened:
                assert opened.io_obj.iloc[0] is source
                assert opened.source.iloc[0] is source
                assert opened.source_loc.iloc[0] is location
                pd.testing.assert_frame_equal(
                    opened[location_columns.columns], location_columns
                )
                assert (
                    opened.uid_raw_obs_file.iloc[0] == raw.lmt_tel.meta.uid_raw_obs_file
                )
            assert "io_obj" not in opened
        assert table.source.iloc[0] is source


def test_direct_tables_require_both_builtin_columns():
    with pytest.raises(AttributeError, match="source and source_loc"):
        FileAccessor(pd.DataFrame({"source": ["file.nc"]}))
