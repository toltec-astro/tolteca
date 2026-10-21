"""Declared attribute columns and file-table projection."""

from dataclasses import MISSING
from types import SimpleNamespace

import pandas as pd
import pytest
import xarray as xr
from pydantic import (
    BaseModel,
    create_model,
)
from tollan.accessor.pandas.model_table import (
    ModelColumn,
    ModelColumnMapper,
    ModelTableAccessor,
    ScalarColumnRule,
)

from tolteca_datamodels.core.file._resolution import _SourceReaderResolver
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.lmt.filepath import LmtFilepathInfo
from tolteca_datamodels.lmt.types import LmtInfo, LmtRawObsMasterCode
from tolteca_datamodels.lmt_base.types import LabelT, SeqNumT
from tolteca_datamodels.toltec.file import (
    ToltecFileAccessor,
    get_info_from_source,
)
from tolteca_datamodels.toltec.filepath import (
    ToltecFilepathInfo,
    ToltecLmtFilepathInfo,
)
from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor


@pytest.mark.parametrize(
    ("annotation", "dtype"), [(LabelT, "string"), (SeqNumT, "Int64")]
)
def test_column_mapper_selects_lmt_scalar_aliases(annotation, dtype):
    model = create_model("Scalar", value=(annotation, ...))
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(model)
    assert list(columns) == ["value"]
    assert columns["value"].attr == "value"
    assert columns["value"].dtype == dtype
    assert columns["value"].default is MISSING


def test_scalar_columns_are_compatible_between_labels_codes_and_primitives():
    class Plain(BaseModel):
        master: str
        master_code: int

    class Constrained(BaseModel):
        master: LmtInfo.MasterT
        master_code: LmtRawObsMasterCode

    mapper = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")], type_rules=[ScalarColumnRule()]
    )
    columns = mapper(Plain)
    assert columns == mapper(Constrained)
    table = ModelTableAccessor.from_models(
        [
            Plain(master="tcs", master_code=0),
            Constrained(master="tcs", master_code=LmtRawObsMasterCode.TCS),
        ],
        columns=columns,
    )
    assert table["master"].tolist() == ["tcs", "tcs"]
    assert table["master_code"].tolist() == [0, 0]
    assert str(table["master"].dtype) == "string"
    assert str(table["master_code"].dtype) == "Int64"


def test_filepath_columns_include_concrete_and_inherited_fields():
    columns = ModelColumnMapper(
        name_rules=[(r".+", r"\g<0>")],
        type_rules=[ScalarColumnRule()],
    )(
        ToltecFilepathInfo,
        name_rules=[(r".+", r"file_\g<0>")],
        default=None,
    )
    assert set(columns) == {
        f"file_{name}"
        for name in ToltecFilepathInfo.model_fields
        if name != "data_models"
    }
    assert {name: ToltecFilepathInfo._columns[name] for name in columns} == columns
    assert all(
        name == f"file_{column.attr.split('.')[-1]}" for name, column in columns.items()
    )
    assert all(column.default is None for column in columns.values())
    assert "file_obsnum" in columns
    assert "file_roach" in columns
    assert "file_proc_suffix" in columns


def test_metadata_columns_are_declared_once_without_reading_counts(
    monkeypatch,
):
    columns = ToltecFileAccessor._metadata_columns
    assert columns["n_blocks"].default is None
    assert columns["block"].default is None
    assert "filepath_info_orig" not in columns
    assert "adc_snap" not in columns

    def unexpected(*args, **kwargs):
        pytest.fail("Repeated field inspection")

    monkeypatch.setattr(ModelColumnMapper, "__call__", unexpected)
    table = ModelTableAccessor.from_models(
        [SimpleNamespace(obsnum=145609)], columns=columns
    )
    assert table["obsnum"].tolist() == [145609]
    assert pd.isna(table["n_blocks"].iloc[0])
    assert ToltecFileAccessor._metadata_columns is columns


def test_filepath_columns_are_declared_once(monkeypatch):
    models = (ToltecFilepathInfo, ToltecLmtFilepathInfo)
    columns = {
        name: column for model in models for name, column in model._columns.items()
    }
    assert columns["file_obsnum"].attr == "obsnum"
    assert columns["obsnum"].attr == "obsnum"

    def unexpected(*args, **kwargs):
        pytest.fail("Repeated filename field inspection")

    monkeypatch.setattr(ModelColumnMapper, "__call__", unexpected)
    table = ToltecFileAccessor.from_sources(["tel_toltec_2025-10-31_145609_00_0002.nc"])
    assert table["file_obsnum"].tolist() == [145609]
    assert table["obsnum"].tolist() == [145609]
    assert {
        name: column for model in models for name, column in model._columns.items()
    } == columns


def test_file_and_current_identifiers_remain_separate(data_lmt_path):
    path = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    location = path.with_name(path.name.replace("145609", "000001"))
    table = ToltecFileAccessor.from_sources(
        [{"source": path, "source_loc": location}], mode="read"
    )
    assert table["obsnum"].tolist() == [145609]
    assert table["file_obsnum"].tolist() == [1]
    assert table.filepath_info.iloc[0].obsnum == 1


def test_membership_columns_are_lists_independent_of_the_filepath_model(monkeypatch):
    info = FilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert isinstance(info, ToltecLmtFilepathInfo)
    monkeypatch.setattr(FilepathInfo, "from_source", lambda *args, **kwargs: info)
    table = ToltecFileAccessor.from_sources(["first.nc", "second.nc"])
    assert table.data_models.dtype == table.file_data_models.dtype == object
    assert not {"producer", "file_producer"} & set(table)
    assert table.data_models.tolist() == [["lmt", "toltec"]] * 2
    assert table.data_models.iloc[0] is not table.data_models.iloc[1]
    assert table.data_models.iloc[0] is not table.file_data_models.iloc[0]
    assert table.data_models.iloc[0] is not info.data_models
    assert table.file_data_models.tolist() == [["lmt", "toltec"]] * 2
    assert info.data_models == ["lmt", "toltec"]


@pytest.mark.parametrize("options", [{"mode": "json"}, {"round_trip": True}])
def test_projection_rejects_dump_options_before_consuming_models(options):
    def models():
        pytest.fail("Invalid options consumed the input")
        yield

    with pytest.raises(TypeError, match="unexpected keyword"):
        ToltecFileAccessor.from_models(models(), columns={}, **options)


def test_file_accessor_delegates_projection_without_schema_inference(monkeypatch):
    # Recorded acquisition name; no file is opened for this projection.
    info = FilepathInfo.from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert info is not None

    def unexpected(*args, **kwargs):
        pytest.fail("Projection inspected a filepath schema or dumped a model")

    assert not hasattr(type(info), "_get_filepath_info_schema")
    monkeypatch.setattr(type(info), "model_dump", unexpected)
    columns = {
        "file_suffix": ModelColumn("suffix", default=None, dtype="string"),
        "obsnum": ModelColumn("obsnum", dtype="Int64"),
    }
    table = ToltecFileAccessor.from_models([info], columns=columns)
    assert list(table) == list(columns)
    assert table["obsnum"].tolist() == [145609]
    assert ModelTableAccessor(table).query("obsnum == 145609").equals(table)


def test_toltec_projection_does_not_convert_undeclared_values():
    info = get_info_from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    assert info is not None
    table = ToltecFileAccessor.from_models(
        [info], columns={"data_kind_toltec": ModelColumn("data_kind_toltec")}
    )
    assert table["data_kind_toltec"].dtype == object
    assert table["data_kind_toltec"].iloc[0] is info.data_kind_toltec


def test_toltec_projection_rejects_implicit_header_columns():
    with pytest.raises(TypeError, match="metadata"):
        ToltecFileAccessor.from_models([], columns={}, metadata=[])  # ty: ignore[unknown-argument]


def test_reference_filename_projection_header_enrichment_and_open(data_lmt_path):
    paths = [
        path
        for directory in ("tel", "toltec")
        for path in sorted((data_lmt_path / directory).rglob("*.nc"))
    ]
    assert len(paths) == 72
    filepath_models = []
    for path in paths:
        info = FilepathInfo.from_source(path.name)
        assert isinstance(info, LmtFilepathInfo | ToltecFilepathInfo)
        filepath_models.append(info)
    table = ModelTableAccessor.from_models(
        filepath_models,
        columns={
            "master": ModelColumn("master", default=None, dtype="string"),
            "obsnum": ModelColumn("obsnum", dtype="Int64"),
            "file_suffix": ModelColumn("suffix", default=None, dtype="string"),
            "roach": ModelColumn("roach", default=None, dtype="Int64"),
        },
    )
    table["source"] = paths
    assert table["master"].isna().any()
    metadata = []
    for path in paths:
        with xr.open_dataset(path) as data:
            identified = _SourceReaderResolver(
                handlers={ToltecFileAccessor: ToltecFileAccessor._reader_rules}
            )(data)
            assert identified is not None
            reader, _ = identified
            metadata.append(reader.meta)
    header = ModelTableAccessor.from_models(
        metadata,
        columns={"master": ModelColumn("master", dtype="string")},
    )
    table["master"] = header["master"]
    selected = ModelTableAccessor(table).query(
        'master == "tcs" & file_suffix in ["vnasweep", "targsweep", "tune"]'
    )
    assert len(selected) == 47
    assert set(selected.roach) == {0, 7, 11}
    for (source,) in selected[["source"]].itertuples(index=False, name=None):
        with xr.open_dataset(source) as data:
            assert ToltecKidsAccessor(data).meta.master == "tcs"
