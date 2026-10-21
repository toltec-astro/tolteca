"""Classification operates on the final projected filename columns."""

import pandas as pd
import pytest
from tollan.accessor.pandas.model_table import RecordRuleMapper

from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.toltec import file as files
from tolteca_datamodels.toltec.file import (
    get_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_dummy_instru.file import DummyInstruFileAccessor
from tolteca_dummy_instru.filepath import DummyInstruFilepathInfo
from tolteca_dummy_instru.types import DummyInstruDataKind


def test_batch_classification_once_and_no_model_dump(monkeypatch):
    paths = [
        "tel_toltec_2025-10-31_145609_00_0002.nc",
        "toltec8_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
        "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
        "chan01-tcs-1-0-2-20251031T120000.nc",
        "unrecognized.nc",
    ] * 20
    assert isinstance(FilepathInfo.from_source(paths[3]), DummyInstruFilepathInfo)
    handlers = (files.ToltecFileAccessor, DummyInstruFileAccessor)
    mappers = {id(cls._data_kind_mapper): cls._data_kind_column for cls in handlers}
    calls = []
    classify = RecordRuleMapper.map_table

    def trace(mapper, table, **kwargs):
        calls.append((mappers[id(mapper)], len(table)))
        assert {"file_ext", "file_prefix", "source"} <= set(table)
        return classify(mapper, table, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("Filename ingestion serialized an intermediate model")

    monkeypatch.setattr(RecordRuleMapper, "map_table", trace)
    monkeypatch.setattr(FilepathInfo, "model_dump", unexpected)
    table = FileAccessor.from_sources(iter(paths), accessor_types=handlers)
    assert calls == [("data_kind_toltec", 40), ("data_kind_dummy_instru", 40)]
    assert (
        table["data_kind_toltec"].tolist()
        == [
            ToltecDataKind.lmt_tel.value,
            ToltecDataKind.vnasweep.value,
            ToltecDataKind.unknown.value,
            ToltecDataKind.unknown.value,
            ToltecDataKind.unknown.value,
        ]
        * 20
    )
    assert (
        table["data_kind_dummy_instru"].tolist()
        == [
            DummyInstruDataKind.unknown.value,
            DummyInstruDataKind.unknown.value,
            DummyInstruDataKind.telescope.value,
            DummyInstruDataKind.counts.value,
            DummyInstruDataKind.unknown.value,
        ]
        * 20
    )


def test_explicit_null_is_not_empty_suffix_and_queries_are_columnar():
    table = guess_info_from_sources(
        [
            "toltec8_145609_000_0002_2025_10_31_09_27_57.nc",
        ]
        * 3
    )
    table["file_suffix"] = pd.array([None, "", "not_known"], dtype="string")
    assert table.toltec_file._guess_data_kinds() is None
    assert table["data_kind_toltec"].tolist() == [
        ToltecDataKind.raw_timestream.value,
        ToltecDataKind.unknown.value,
        ToltecDataKind.unknown.value,
    ]


@pytest.mark.parametrize(
    ("suffix", "ext", "kind"),
    [
        ("vnasweep", "nc", ToltecDataKind.vnasweep),
        ("targsweep", "nc", ToltecDataKind.targsweep),
        ("tune", "nc", ToltecDataKind.tune),
        ("tune_processed", "nc", ToltecDataKind.reduced_targsweep),
        ("tune_kids_find", "ecsv", ToltecDataKind.kids_prop_table),
        ("tune_targamps", "dat", ToltecDataKind.targ_amps_dat),
        ("vnasweep", "txt", ToltecDataKind.kids_model_params_table),
        ("not_known", "nc", ToltecDataKind.unknown),
        ("vnasweep", "nc.gz", ToltecDataKind.unknown),
    ],
)
def test_filename_kind_rules(suffix, ext, kind):
    # Unsupported suffix/extensions deliberately test classification fallbacks.
    path = f"toltec8_145605_000_0001_2025_10_31_09_20_06_{suffix}.{ext}"
    row = get_info_from_source(path)
    assert row.data_kind_toltec == kind.value
    assert row.file_suffix == suffix


def test_unknown_and_empty_tables():
    empty = guess_info_from_sources([])
    assert empty.empty and str(empty["data_kind_toltec"].dtype) == "Int64"
    unknown = guess_info_from_sources(["unrecognized.nc"])
    assert pd.isna(unknown.iloc[0].obsnum)
    assert unknown.iloc[0].data_kind_toltec == ToltecDataKind.unknown.value


def test_classifier_preserves_positions_with_duplicate_labels():
    table = guess_info_from_sources(
        [
            "tel_toltec_2025-10-31_145609_00_0002.nc",
            "tel_toltec2_2025-10-31_145609_00_0002.nc",
        ]
    )
    table.index = [4, 4]
    assert table.toltec_file._guess_data_kinds() is None
    result = table["data_kind_toltec"]
    assert result.index.tolist() == [4, 4]
    assert result.tolist() == [
        ToltecDataKind.lmt_tel.value,
        ToltecDataKind.lmt_tel2.value,
    ]
