"""Product naming grammar applied to recorded raw input names.

The reference fixture contains raw sweeps, not these output tables. Generated
names follow v2 FileStoreConfigMixin.make_data_path (input stem + output suffix)
and the product tags declared in v2 tolteca_datamodels.toltec.file.
"""

import pandas as pd
import pytest

from tolteca_datamodels.file import (
    FileIdentifier,
    SourceInfoModel,
    guess_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.file import RoachFilepathInfo
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.fixture(params=["vnasweep", "targsweep", "tune"])
def raw_sweep_path(request, data_lmt_path):
    paths = list((data_lmt_path / "toltec/tcs/toltec0").glob(f"*_{request.param}.nc"))
    assert len(paths) == 1
    return paths[0]


@pytest.mark.parametrize(
    ("proc_suffix", "ext", "kind"),
    [
        ("targamps", "dat", ToltecDataKind.TargAmpsDat),
        ("kids_find", "ecsv", ToltecDataKind.KidsPropTable),
        ("kids_fit", "ecsv", ToltecDataKind.KidsPropTable),
        ("chan_prop", "ecsv", ToltecDataKind.KidsPropTable),
        ("adrv", "ecsv", ToltecDataKind.KidsPropTable),
    ],
)
def test_processing_segment_preserves_the_complete_suffix(
    raw_sweep_path, proc_suffix, ext, kind
):
    raw = RoachFilepathInfo.from_path(raw_sweep_path)
    assert isinstance(raw, RoachFilepathInfo)
    assert raw.proc_suffix is None
    name = f"{raw_sweep_path.stem}_{proc_suffix}.{ext}"
    info = guess_info_from_source(name)
    assert info is not None
    parsed = info.filepath_info
    assert isinstance(parsed, RoachFilepathInfo)
    assert parsed.suffix == f"{raw.suffix}_{proc_suffix}"
    assert parsed.proc_suffix == proc_suffix
    assert info.file_suffix == parsed.suffix
    assert info.data_kind is kind
    assert parsed.ut == raw.ut
    assert (parsed.obsnum, parsed.subobsnum, parsed.scannum) == (
        raw.obsnum,
        raw.subobsnum,
        raw.scannum,
    )
    assert (
        SourceInfoModel.model_validate_json(info.model_dump_json(round_trip=True))
        == info
    )


@pytest.mark.parametrize(
    "suffix", ["unknown", "vnasweep_", "kids_find", "targetsweep_kids_find"]
)
def test_unknown_suffix_is_not_classified_as_a_product(raw_sweep_path, suffix):
    # Deliberately renamed files test rejection by classification, not the grammar.
    stem = raw_sweep_path.stem.rsplit("_", 1)[0]
    info = guess_info_from_source(f"{stem}_{suffix}.ecsv")
    assert info is not None
    assert info.file_suffix == suffix
    assert info.data_kind is ToltecDataKind.Unknown


def test_processing_tag_is_available_in_vectorized_catalogs(
    raw_sweep_path, data_lmt_path, tmp_path, monkeypatch
):
    product = f"{raw_sweep_path.stem}_kids_find.ecsv"
    telescope = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    frame = guess_info_from_sources([product, raw_sweep_path, telescope])
    records = frame.toltec_file.to_info_list()
    path = tmp_path / "catalog.parquet"
    frame.to_parquet(path)
    restored = pd.read_parquet(path, dtype_backend="pyarrow")
    assert restored.toltec_file.to_info_list() == records

    def unexpected(*args, **kwargs):
        pytest.fail("Catalog selection attempted to re-parse filenames")

    monkeypatch.setattr(FileIdentifier, "filename_pattern_parser", {})
    monkeypatch.setattr(RoachFilepathInfo, "from_path", unexpected)
    for expression in (
        'filepath_info.proc_suffix == "kids_find"',
        'file_proc_suffix == "kids_find"',
    ):
        assert restored.toltec_file.query(expression).index.tolist() == [0]
    full_suffix = records[0].file_suffix
    assert restored.toltec_file.query(
        f'file_suffix == "{full_suffix}"'
    ).index.tolist() == [0]
