"""Typed filename metadata and vectorized catalog selection."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal
from pydantic import ValidationError
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.file import (
    FilepathInfo,
    SourceInfoModel,
    ToltecFileAccessor,
    guess_info_from_source,
    guess_info_from_sources,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
            (
                "toltec8",
                145605,
                1,
                datetime(2025, 10, 31, 9, 20, 6, tzinfo=UTC),
                "vnasweep",
            ),
        ),
        (
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            ("tel_toltec", 145609, 2, datetime(2025, 10, 31, tzinfo=UTC), None),
        ),
        (
            "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc",
            (
                "toltec8",
                145609,
                1,
                datetime(2025, 10, 31, 9, 26, 59, tzinfo=UTC),
                "tune",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "prefix", ["/recorded", "host:/recorded", "https://example.org"]
)
def test_filename_info(name, expected, prefix, monkeypatch):
    file_prefix, obsnum, scannum, ut, suffix = expected

    def unexpected(*args, **kwargs):
        pytest.fail("Filename parsing accessed the filesystem")

    monkeypatch.setattr(Path, "exists", unexpected)
    info = FilepathInfo.from_path(f"{prefix}/{name}")
    assert info is not None
    assert (info.prefix, info.obsnum, info.subobsnum, info.scannum) == (
        file_prefix,
        obsnum,
        0,
        scannum,
    )
    assert (info.ut, info.suffix, info.ext) == (ut, suffix, "nc")
    assert type(info).model_validate_json(info.model_dump_json(round_trip=True)) == info
    with pytest.raises(ValidationError, match="frozen"):
        info.obsnum = 1  # ty: ignore[invalid-assignment]


def test_unknown_filename_and_invalid_date():
    assert FilepathInfo.from_path("renamed.nc", raise_on_error=False) is None
    with pytest.raises(ValueError, match="Unrecognized"):
        FilepathInfo.from_path("renamed.nc")
    with pytest.raises(ValueError, match="does not match format"):
        FilepathInfo.from_path(
            "tel_toltec_2025-99-31_145609_00_0002.nc", raise_on_error=False
        )


def test_original_filename_is_cached_and_independent_of_source(monkeypatch):
    original = "/recorded/toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
    record = SourceInfoModel(source="downloaded.nc", filepath_orig=original)
    parsed = record.filepath_info_orig
    assert parsed is not None and parsed.suffix == "vnasweep"

    def unexpected(*args, **kwargs):
        pytest.fail("The cached original filename was parsed again")

    with monkeypatch.context() as patch:
        patch.setattr(FilepathInfo, "from_path", unexpected)
        assert record.filepath_info_orig is parsed
        assert record.model_dump()["filepath_info_orig"]["ut"] == parsed.ut
        assert "filepath_info_orig" not in repr(record)
        ToltecFileAccessor.from_info_list([record])

    restored = SourceInfoModel.model_validate_json(
        record.model_dump_json(round_trip=True)
    )
    assert restored.filepath_info_orig == parsed
    assert restored.filepath_orig == original
    updated = SourceInfoModel.model_validate(
        {
            **record.model_dump(round_trip=True),
            "filepath_orig": (
                "/recorded/toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
            ),
        }
    )
    assert updated.filepath_info_orig is not None
    assert updated.filepath_info_orig.suffix == "tune"
    assert record.filepath_info_orig.suffix == "vnasweep"
    discovered = guess_info_from_source(Path(original).name)
    assert discovered is not None
    assert discovered.filepath_orig is None
    assert discovered.filepath_info_orig is None


@pytest.fixture
def records(tmp_path, lmtmc_records, toltecdb_records):
    result = [
        SourceInfoModel(
            source=tmp_path / "downloaded.nc",
            obsnum=145605,
            filepath_orig="/data_lmt/toltec/tcs/toltec8/toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        ),
        SourceInfoModel(
            source=tmp_path / "tel.nc",
            obsnum=145606,
            filepath_orig="/data_lmt/tel/tel_toltec_2025-10-31_145606_00_0001.nc",
        ),
        SourceInfoModel(source=tmp_path / "unknown.nc"),
        SourceInfoModel(
            source=tmp_path / "tune.nc",
            obsnum=145609,
            filepath_orig="/data_lmt/toltec/tcs/toltec8/toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc",
        ),
    ]
    recorded_paths = {row["FileName"] for row in [*lmtmc_records, *toltecdb_records]}
    assert all(
        item.filepath_orig is None or item.filepath_orig in recorded_paths
        for item in result
    )
    return result


def test_catalog_scalar_and_struct_columns(records):
    catalog = ToltecFileAccessor.from_info_list(iter(records))
    scalars = pd.DataFrame(
        [
            record.model_dump(
                mode="json",
                context={"flat": True},
                exclude={"filepath_info", "filepath_info_orig"},
            )
            for record in records
        ],
        columns=[
            name for name in SourceInfoModel.model_fields if name != "filepath_info"
        ],
    )
    assert_frame_equal(
        catalog.drop(columns=["filepath_info", "filepath_info_orig"]), scalars
    )
    column = catalog["filepath_info_orig"]
    assert isinstance(column.dtype, pd.ArrowDtype)
    assert pa.types.is_struct(column.dtype.pyarrow_dtype)
    assert column.dtype.pyarrow_dtype.field("ut").type == pa.timestamp("us", "UTC")
    assert column.isna().tolist() == [False, False, True, False]
    assert column.struct.field("suffix").isna().tolist() == [False, True, True, False]
    assert catalog.toltec_file.to_info_list() == records


@pytest.mark.parametrize(
    ("expression", "indices"),
    [
        ('filepath_info_orig.suffix in ["vnasweep", "tune"]', [0, 3]),
        ('filepath_info_orig.suffix != "tune"', [0, 1, 2]),
        ('obsnum >= 145606 & filepath_info_orig.suffix == "tune"', [3]),
        ('obsnum < 145606 | filepath_info_orig.prefix == "tel_toltec"', [0, 1]),
        ('filepath_info_orig.ut >= "2025-10-31T00:00:00Z"', [0, 1, 3]),
        ("filepath_info_orig.ut.isna()", [2]),
        ("filepath_info_orig.interface.isna()", [2]),
        ('filepath_info_orig.data_kind == "VnaSweep"', [0]),
        ('filepath_info_orig.data_kind == "LmtTel"', [1]),
    ],
)
@pytest.mark.parametrize("short_names", [False, True])
def test_dotted_selection(records, expression, indices, short_names):
    if short_names:
        for name in ("suffix", "ut", "interface", "prefix", "data_kind"):
            expression = expression.replace(
                f"filepath_info_orig.{name}", f"file_{name}_orig"
            )
    catalog = ToltecFileAccessor.from_info_list(records)
    selected = catalog.toltec_file.query(expression)
    assert selected.index.tolist() == indices
    assert isinstance(selected["filepath_info_orig"].dtype, pd.ArrowDtype)
    assert catalog.toltec_file.to_info_list(expression) == [records[i] for i in indices]


@pytest.mark.parametrize("field", ["filepath_info_orig.suffix", "file_suffix_orig"])
def test_selection_uses_current_rows_without_reconstructing_models(
    records, monkeypatch, field
):
    catalog = ToltecFileAccessor.from_info_list(records)

    def unexpected(*args, **kwargs):
        pytest.fail("Selection traversed row models")

    monkeypatch.setattr(FilepathInfo, "from_path", unexpected)
    monkeypatch.setattr(SourceInfoModel, "model_validate", unexpected)
    reordered = catalog.iloc[[3, 0, 1]]
    reordered.index = ["duplicate", "duplicate", "last"]
    selected = reordered.toltec_file.query(f'{field} == "vnasweep"')
    assert selected["source"].tolist() == [
        records[0].model_dump(
            mode="json",
            context={"flat": True},
            exclude={"filepath_info", "filepath_info_orig"},
        )["source"]
    ]
    assert selected.index.tolist() == ["duplicate"]
    # The accessor must not retain child columns from an earlier frame state.
    reordered["filepath_info_orig"] = reordered["filepath_info_orig"].array.take(
        [0, 0, 0]
    )
    assert reordered.toltec_file.query(f'{field} == "vnasweep"').empty


@pytest.mark.parametrize("size", [0, 2])
def test_empty_and_all_null_structs(size):
    catalog = guess_info_from_sources(["unknown.nc"] * size)
    assert isinstance(catalog["filepath_info_orig"].dtype, pd.ArrowDtype)
    assert catalog.toltec_file.query('filepath_info_orig.suffix == "tune"').empty
    assert catalog.toltec_file.query('file_suffix_orig == "tune"').empty
    assert catalog.toltec_file.query('file_suffix == "tune"').empty
    assert (
        len(catalog.toltec_file.query("file_ut.isna() & file_ut_orig.isna()")) == size
    )
    assert len(catalog.toltec_file.to_info_list()) == size


def test_unknown_field_is_not_silently_ignored(records):
    catalog = ToltecFileAccessor.from_info_list(records)
    with pytest.raises(AttributeError, match="not_a_field"):
        catalog.toltec_file.query('filepath_info_orig.not_a_field == "tune"')
    with pytest.raises(pd.errors.UndefinedVariableError):
        catalog.toltec_file.query("missing > 0")
    with pytest.raises(pd.errors.UndefinedVariableError):
        catalog.toltec_file.query("file_not_a_field_orig > 0")


def test_effective_path_aliases_and_parquet(tmp_path, monkeypatch):
    original = (
        "/data_lmt/toltec/tcs/toltec8/"
        "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    # Deliberate conflicting paths test precedence without inventing filenames.
    current = "/copied/toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc"
    record = guess_info_from_source(current, filepath_orig=original)
    assert record is not None
    assert record.filepath_info is not None
    assert record.file_ut == record.filepath_info.ut
    assert record.file_suffix == record.filepath_info.suffix == "vnasweep"
    assert record.file_ext == record.filepath_info.ext == "nc"
    assert record.filepath_info_orig is not None
    assert record.filepath_info_orig.suffix == "tune"
    fallback = guess_info_from_source(
        tmp_path / "downloaded.nc", filepath_orig=original
    )
    assert fallback is not None
    unknown = SourceInfoModel(source=tmp_path / "unknown.nc")
    catalog = ToltecFileAccessor.from_info_list([record, fallback, unknown])
    path = tmp_path / "effective-paths.parquet"
    catalog.to_parquet(path, index=False)
    restored = pd.read_parquet(path, dtype_backend="pyarrow")
    assert restored.toltec_file.to_info_list() == [record, fallback, unknown]

    def unexpected(*args, **kwargs):
        pytest.fail("Selection parsed a filename or rebuilt a row model")

    monkeypatch.setattr(FilepathInfo, "from_path", unexpected)
    monkeypatch.setattr(SourceInfoModel, "model_validate", unexpected)
    for frame in (catalog, restored):
        assert frame.toltec_file.query("file_suffix == 'vnasweep'").index.tolist() == [
            0
        ]
        assert frame.toltec_file.query("file_suffix_orig == 'tune'").index.tolist() == [
            0,
            1,
        ]
        assert frame.toltec_file.query("file_ut < file_ut_orig").index.tolist() == [0]
        assert frame.toltec_file.query(
            "file_ut >= '2025-10-31T09:26:00Z'"
        ).index.tolist() == [1]
        assert frame.toltec_file.query(
            "file_ext == 'nc' & file_master_str_orig == 'tcs'"
        ).index.tolist() == [0, 1]
        assert frame.toltec_file.query("file_ut.isna()").index.tolist() == [2]
        assert frame.toltec_file.query(
            "filepath_info.ut == file_ut_orig"
        ).index.tolist() == [1]


def test_aliases_project_only_requested_fields(records, monkeypatch):
    catalog = ToltecFileAccessor.from_info_list(records)
    accessor_cls = type(catalog["filepath_info_orig"].struct)
    field = accessor_cls.field
    projections = []

    def traced(accessor, name):
        projections.append(name)
        return field(accessor, name)

    monkeypatch.setattr(accessor_cls, "field", traced)
    catalog.toltec_file.query("obsnum == 145609")
    assert projections == []
    catalog.toltec_file.query("file_suffix_orig == 'tune'")
    assert projections and set(projections) == {"suffix"}


def test_unregistered_master_catalog_roundtrip(tmp_path):
    original = (
        "/data_lmt/toltec/simu/toltec8/"
        "toltec8_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    )
    record = guess_info_from_source(tmp_path / "downloaded.nc", filepath_orig=original)
    assert record is not None
    catalog = ToltecFileAccessor.from_info_list([record])
    path = tmp_path / "simulated.parquet"
    catalog.to_parquet(path, index=False)
    restored = pd.read_parquet(path, dtype_backend="pyarrow")
    for frame in (catalog, restored):
        assert frame.toltec_file.query(
            "master == 'simu' & file_master_str_orig == 'simu'"
        ).index.tolist() == [0]
        assert frame.toltec_file.to_info_list() == [record]


def test_nested_struct_resolver_preserves_index():
    dtype = pd.ArrowDtype(pa.struct([("inner", pa.struct([("value", pa.int64())]))]))
    catalog = pd.DataFrame({"source": ["a", "b"]})
    catalog["nested"] = pd.Series(
        [{"inner": {"value": 1}}, {"inner": {"value": 2}}], dtype=dtype
    )
    catalog = catalog.iloc[::-1]
    assert catalog.toltec_file.query("nested.inner.value == 1")["source"].tolist() == [
        "a"
    ]


@pytest.mark.parametrize("empty", [False, True])
def test_parquet_roundtrip(records, tmp_path, empty):
    catalog = ToltecFileAccessor.from_info_list([] if empty else records)
    path = tmp_path / "catalog.parquet"
    catalog.to_parquet(path, index=False)
    restored = pd.read_parquet(path, dtype_backend="pyarrow")
    assert restored["filepath_info_orig"].dtype == catalog["filepath_info_orig"].dtype
    assert (
        restored["filepath_info_orig"].tolist()
        == catalog["filepath_info_orig"].tolist()
    )
    assert restored.toltec_file.to_info_list() == ([] if empty else records)
    selected = restored.toltec_file.query(
        'filepath_info_orig.suffix in ["vnasweep", "tune"]'
    )
    assert selected.index.tolist() == ([] if empty else [0, 3])


def test_reference_original_filename_catalog(data_lmt_path, tmp_path):
    records = []
    for directory, header in [
        ("tel", "Header.File.Name"),
        ("toltec", "Header.Toltec.Filename"),
    ]:
        for path in sorted((data_lmt_path / directory).rglob("*.nc")):
            with xr.open_dataset(path) as data:
                value = data[header].item()
                original = (
                    value.decode().strip()
                    if isinstance(value, bytes)
                    else str(value).strip()
                )
            records.append(
                SourceInfoModel(source=fileloc(path), filepath_orig=original)
            )
    assert records
    catalog = ToltecFileAccessor.from_info_list(records)
    assert not catalog["filepath_info_orig"].isna().any()
    path = tmp_path / "reference-catalog.parquet"
    catalog.to_parquet(path, index=False)
    restored = pd.read_parquet(path, dtype_backend="pyarrow")
    assert restored.toltec_file.to_info_list() == records
    assert (
        restored.toltec_file.query('filepath_info_orig.prefix == "tel_toltec"').shape[0]
        > 0
    )
