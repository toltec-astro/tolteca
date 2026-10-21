"""Column-wise bit predicates and TolTEC flag names in catalog queries."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.toltec.file import guess_info_from_sources
from tolteca_datamodels.toltec.types import ToltecDataKind


class _SharedDataModel(DataModelInfo):
    name = "shared"


class SharedFileAccessor(FileAccessor[_SharedDataModel]):
    pass


@pytest.fixture(params=["int64", "int64[pyarrow]"])
def catalog(request):
    # These are bitmask unit cases, including combinations and zero, not claims
    # that acquisition files carry every combination of these flags.
    kinds = [
        ToltecDataKind.vnasweep,
        ToltecDataKind.targsweep,
        ToltecDataKind.tune,
        ToltecDataKind.raw_timestream,
        ToltecDataKind.lmt_tel,
        ToltecDataKind.vnasweep | ToltecDataKind.tune,
        ToltecDataKind.raw_sweep,
        ToltecDataKind.raw_kids_data,
        ToltecDataKind.unknown,
        ToltecDataKind(0),
    ]
    return pd.DataFrame(
        {
            "source": [None] * len(kinds),
            "source_loc": [None] * len(kinds),
            "data_kind_toltec": pd.Series(
                [kind.value for kind in kinds], dtype=request.param
            ),
            "row_id": range(len(kinds)),
        }
    )


@pytest.mark.parametrize(
    ("expression", "indices"),
    [
        ("bits_any_set(data_kind_toltec, raw_sweep)", [0, 1, 2, 5, 6, 7]),
        ("bits_all_set(data_kind_toltec, raw_sweep)", [6, 7]),
        (
            "bits_any_set(data_kind_toltec, raw_sweep | raw_timestream)",
            [0, 1, 2, 3, 5, 6, 7],
        ),
        ("bits_all_set(data_kind_toltec, raw_sweep | raw_timestream)", [7]),
        ("bits_any_set(data_kind_toltec, raw_kids_data & ~tune)", [0, 1, 3, 5, 6, 7]),
        ("~bits_any_set(data_kind_toltec, raw_sweep)", [3, 4, 8, 9]),
        ("bits_any_set(data_kind_toltec, tune & vnasweep)", []),
        ("bits_all_set(data_kind_toltec, tune & vnasweep)", list(range(10))),
        ("bits_any_set(data_kind_toltec, raw_sweep) & row_id < 2", [0, 1]),
        ("bits_all_set(data_kind_toltec, tune) | row_id == 4", [2, 4, 5, 6, 7]),
        ("bits_any_set(data_kind_toltec, 7)", [0, 1, 2, 5, 6, 7]),
        ("bits_all_set(data_kind_toltec, 5)", [5, 6, 7]),
        ("bits_any_set(data_kind_toltec, 0)", []),
        ("bits_all_set(data_kind_toltec, 0)", list(range(10))),
    ],
)
def test_bit_query(catalog, expression, indices):
    original = catalog.copy(deep=True)
    selected = catalog.toltec_file.query(expression)
    assert_frame_equal(selected, catalog.iloc[indices])
    assert_frame_equal(catalog, original)


@pytest.mark.parametrize("dtype", ["Int64", "int64[pyarrow]"])
@pytest.mark.parametrize(
    ("expression", "indices"),
    [
        ("bits_any_set(data_kind_toltec, vnasweep)", [0]),
        ("bits_all_set(data_kind_toltec, vnasweep)", [0]),
        ("~bits_any_set(data_kind_toltec, vnasweep)", [1]),
        ("~bits_all_set(data_kind_toltec, vnasweep)", [1]),
        ("bits_any_set(data_kind_toltec, 0)", []),
        ("bits_all_set(data_kind_toltec, 0)", [0, 1]),
    ],
)
def test_null_bits_remain_unknown(dtype, expression, indices):
    catalog = pd.DataFrame(
        {
            "source": [None] * 3,
            "source_loc": [None] * 3,
            "data_kind_toltec": pd.Series([1, 0, None], dtype=dtype),
        }
    )
    assert catalog.toltec_file.query(expression).index.tolist() == indices


@pytest.mark.parametrize("predicate", ["bits_any_set", "bits_all_set"])
def test_empty_catalog(predicate):
    catalog = guess_info_from_sources([])
    assert_frame_equal(
        catalog.toltec_file.query(f"{predicate}(data_kind_toltec, raw_sweep)"), catalog
    )


def test_predicates_are_generic_but_constants_are_toltec_specific(catalog):
    accessor = SharedFileAccessor(catalog)
    assert accessor.query("bits_any_set(data_kind_toltec, 7)").index.tolist() == [
        0,
        1,
        2,
        5,
        6,
        7,
    ]
    assert accessor.query("bits_all_set(data_kind_toltec, 7)").index.tolist() == [6, 7]
    with pytest.raises(pd.errors.UndefinedVariableError, match="raw_sweep"):
        accessor.query("bits_any_set(data_kind_toltec, raw_sweep)")


@pytest.mark.parametrize("name", ["raw_sweep", "bits_any_set", "bits_all_set"])
def test_existing_column_names_take_precedence(catalog, name):
    catalog[name] = catalog["row_id"]
    assert catalog.toltec_file.query(f"{name} == 2").index.tolist() == [2]


def test_all_named_flags_are_available(catalog):
    for name, flag in ToltecDataKind.__members__.items():
        expected = catalog.loc[(catalog["data_kind_toltec"] & flag.value) != 0]
        assert_frame_equal(
            catalog.toltec_file.query(f"bits_any_set(data_kind_toltec, {name})"),
            expected,
        )


def test_real_filenames_support_bit_queries_and_row_iteration(toltecdb_records):
    catalog = guess_info_from_sources(row["FileName"] for row in toltecdb_records)
    expression = "bits_any_set(data_kind_toltec, raw_sweep)"
    expected = catalog.loc[
        catalog["data_kind_toltec"].isin(
            [kind.value for kind in ToltecDataKind.raw_sweep]
        )
    ]
    assert not expected.empty
    assert_frame_equal(catalog.toltec_file.query(expression), expected)
    assert list(catalog.toltec_file.query(expression).itertuples(index=False)) == (
        list(expected.itertuples(index=False))
    )


@pytest.mark.parametrize("predicate", ["bits_any_set", "bits_all_set"])
@pytest.mark.parametrize("column", ["file_obsnum", "exposure"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(("operator", "indices"), [("&", [1]), ("|", [0, 1])])
def test_built_table_combines_flags_with_nullable_columns(
    predicate, column, reverse, operator, indices
):
    paths = [
        "tel_toltec_2025-10-31_145609_00_0002.nc",
        "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        "unrecognized.nc",  # Deliberate unidentified-source/null-query case.
    ]
    table = guess_info_from_sources(paths)
    table["exposure"] = pd.array([1.0, 1.0, None], dtype="Float64")
    records = list(table.itertuples(index=False))
    table.index = ["duplicate", "duplicate", "unknown"]
    original = table.copy(deep=True)
    clauses = [f"{predicate}(data_kind_toltec, vnasweep)", f"{column} > 0"]
    if reverse:
        clauses.reverse()
    selected = table.toltec_file.query(f" {operator} ".join(clauses))
    assert_frame_equal(selected, table.iloc[indices])
    assert list(selected.itertuples(index=False)) == [records[i] for i in indices]
    assert_frame_equal(table, original)
