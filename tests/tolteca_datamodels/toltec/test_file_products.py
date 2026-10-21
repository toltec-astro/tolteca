"""Product naming grammar applied to recorded raw input names.

The reference fixture contains raw sweeps, not these output tables. Generated
names follow v2 FileStoreConfigMixin.make_data_path (input stem + output suffix)
and the product tags declared in v2 tolteca_datamodels.toltec.file.
"""

import pytest

from tolteca_datamodels.filepath import FilepathInfo
from tolteca_datamodels.toltec.file import (
    get_info_from_source,
    guess_info_from_sources,
)
from tolteca_datamodels.toltec.filepath import ToltecFilepathInfo
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.fixture(params=["vnasweep", "targsweep", "tune"])
def raw_sweep_path(request, data_lmt_path):
    paths = list(
        (data_lmt_path / "toltec/tcs/toltec0").glob(f"*1456*_{request.param}.nc")
    )
    assert len(paths) == 1
    return paths[0]


def test_processing_tag_is_queryable_without_reparsing(
    raw_sweep_path, data_lmt_path, monkeypatch
):
    product = f"{raw_sweep_path.stem}_kids_find.ecsv"
    telescope = data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc"
    frame = guess_info_from_sources([product, raw_sweep_path, telescope])
    records = list(frame.itertuples(index=False))

    def unexpected(*args, **kwargs):
        pytest.fail("Table selection attempted to re-parse filenames")

    monkeypatch.setattr(FilepathInfo, "from_source", unexpected)
    assert frame.toltec_file.query(
        'file_proc_suffix == "kids_find"'
    ).index.tolist() == [0]
    full_suffix = frame["file_suffix"].iloc[0]
    assert frame.toltec_file.query(
        f'file_suffix == "{full_suffix}"'
    ).index.tolist() == [0]
    assert list(frame.itertuples(index=False)) == records


@pytest.mark.parametrize(
    ("tag", "ext", "kind"),
    [
        ("kids_find", "ecsv", ToltecDataKind.kids_prop_table),
        ("kids_fit", "ecsv", ToltecDataKind.kids_prop_table),
        ("targamps", "dat", ToltecDataKind.targ_amps_dat),
        ("unknown", "ecsv", ToltecDataKind.unknown),
    ],
)
def test_processing_segment_preserves_full_suffix(raw_sweep_path, tag, ext, kind):
    product = raw_sweep_path.with_name(f"{raw_sweep_path.stem}_{tag}.{ext}")
    parsed = FilepathInfo.from_source(product)
    assert isinstance(parsed, ToltecFilepathInfo)
    row = get_info_from_source(product)
    assert row.file_proc_suffix == parsed.proc_suffix == tag
    assert (
        row.file_suffix
        == parsed.suffix
        == f"{raw_sweep_path.stem.rsplit('_', 1)[1]}_{tag}"
    )
    assert row.data_kind_toltec == kind.value
