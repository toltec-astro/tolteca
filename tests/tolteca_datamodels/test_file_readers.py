"""Concrete accessors select readers and selectors for the rows they handle."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from astropy import units as u
from astropy.time import Time

from tolteca_datamodels.core.file import (
    FileAccessorBase,
    FileAccessorView,
    FileAccessorWorkflow,
)
from tolteca_datamodels.core.obs_spec import FieldConstraint, ObsSpecBase, Selector
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.lmt.file import LmtFileAccessor
from tolteca_datamodels.lmt.tel import LmtTelAccessor
from tolteca_datamodels.lmt_base.obs_spec import ObsSpec
from tolteca_datamodels.toltec.file import ToltecFileAccessor
from tolteca_datamodels.toltec.hwpr import ToltecHwprAccessor
from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor
from tolteca_dummy_instru.data import DummyInstruAccessor
from tolteca_dummy_instru.file import (
    DummyInstruCounterFileAccessor,
    DummyInstruFileAccessor,
    DummyInstruTelescopeFileAccessor,
)
from tolteca_dummy_instru.metadata import DummyInstruMetadata
from tolteca_dummy_instru.testing import make_dataset


def _check_reader_selection(file_accessor, expected):
    """Compare value-based, content-based and declared metadata reader support."""
    with file_accessor.open(on_error="raise") as table:
        assert [
            position
            for position, reader in enumerate(table.accessor_obj)
            if reader is not None
        ] == [position for position, _, _ in expected]
        for position, accessor_cls, reader_cls in expected:
            workflow_cls = accessor_cls._workflow_cls
            rules = workflow_cls._source_reader_rules
            assert list(rules.map_table(table.iloc[[position]])) == [reader_cls]
            assert rules.identify(table.io_obj.iloc[position]) is reader_cls
            reader = table.accessor_obj.iloc[position]
            assert type(reader) is reader_cls
            assert type(reader.meta) in workflow_cls._metadata_columns


def test_toltec_and_lmt_readers_agree_with_contents(data_lmt_path):
    paths = [
        data_lmt_path / "tel/tel_toltec_2025-10-31_145609_00_0002.nc",
        *(
            data_lmt_path / "toltec/tcs" / name.split("_")[0] / name
            for name in (
                "toltec0_145609_000_0002_2025_10_31_09_27_57.nc",
                "toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc",
                "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
                "toltec0_145605_000_0001_2025_10_31_09_20_06_vnasweep.nc",
                "hwpr_156755_000_0002_2026_05_16_12_33_28.nc",
            )
        ),
    ]
    _check_reader_selection(
        FileAccessor.from_sources(paths),
        [
            (0, ToltecFileAccessor, LmtTelAccessor),
            *((i, ToltecFileAccessor, ToltecKidsAccessor) for i in range(1, 5)),
            (5, ToltecFileAccessor, ToltecHwprAccessor),
        ],
    )
    _check_reader_selection(
        LmtFileAccessor.from_sources(paths[:1]),
        [(0, LmtFileAccessor, LmtTelAccessor)],
    )


def test_dummy_readers_agree_with_contents(tmp_path, dummy_tel_path):
    counter_path = tmp_path / "chan01-tcs-1-0-2-20251031T120000.nc"
    meta = DummyInstruMetadata(
        data_models=frozenset({"dummy_instru"}),
        instru="dummy_instru",
        interface="chan01",
        interface_kind="counter",
        channel=1,
        master="tcs",
        obsnum=1,
        subobsnum=0,
        scannum=2,
        filepath_orig=str(counter_path),
        t0=Time(1761912000.0, format="unix"),
        t1=Time(1761912003.0, format="unix"),
        t_int=3 * u.s,
    )
    make_dataset(meta, [1, 2, 4, 8]).to_netcdf(counter_path)
    _check_reader_selection(
        DummyInstruFileAccessor.from_sources([counter_path, dummy_tel_path]),
        [
            (0, DummyInstruCounterFileAccessor, DummyInstruAccessor),
            (1, DummyInstruTelescopeFileAccessor, LmtTelAccessor),
        ],
    )


_obs_spec_sources = [
    "toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
    "toltec7_145606_000_0001_2025_10_31_09_22_14_targsweep.nc",
    "tel_toltec_2025-10-31_145606_00_0001.nc",
    "chan01-tcs-145606-0-1-20251031T120000.nc",
]


@pytest.mark.parametrize(
    ("accessor_cls", "spec", "interfaces"),
    [
        (ToltecFileAccessor, "145606/7", ["toltec7"]),
        (ToltecFileAccessor, "145606/tel", ["tel"]),
        (LmtFileAccessor, "145606/7", []),
        (DummyInstruCounterFileAccessor, "145606/1", ["chan01"]),
        (
            DummyInstruCounterFileAccessor,
            145606,
            ["toltec0", "toltec7", "tel", "chan01"],
        ),
    ],
)
def test_query_obs_spec_uses_the_accessor_interface_numbering(
    accessor_cls, spec, interfaces
):
    direct = accessor_cls.from_sources(_obs_spec_sources)
    partial = FileAccessorView.from_sources(
        _obs_spec_sources, accessor_types=(accessor_cls,)
    )
    for file_accessor in (direct, partial):
        assert file_accessor.query_obs_spec(spec).interface.tolist() == interfaces
        assert (
            file_accessor.query_obs_spec(ObsSpec.parse(spec)).interface.tolist()
            == interfaces
        )


@dataclass(frozen=True)
class _ObsnumOnlySpec(ObsSpecBase):
    obsnum: Selector = None
    _field_constraints: ClassVar = {"obsnum": FieldConstraint(kind="integer")}
    _primary_field: ClassVar = "obsnum"


def test_single_workflow_rejects_incompatible_parsed_layout_directly_and_in_view():
    direct = LmtFileAccessor.from_sources(_obs_spec_sources)
    partial = FileAccessorView.from_sources(
        _obs_spec_sources, accessor_types=(LmtFileAccessor,)
    )
    for file_accessor in (direct, partial):
        with pytest.raises(TypeError, match="Expected an ObsSpec"):
            file_accessor.query_obs_spec(_ObsnumOnlySpec.parse(145606))


def test_composite_obs_spec_applies_position_once_over_combined_interfaces():
    file_accessor = FileAccessorView.from_sources(
        _obs_spec_sources,
        accessor_types=(ToltecFileAccessor, DummyInstruCounterFileAccessor),
    )
    for spec in ("145606/(0,13)[-1]", ObsSpec.parse("145606/(0,13)[-1]")):
        assert file_accessor.query_obs_spec(spec).interface.tolist() == ["toltec7"]


def test_composite_with_different_layouts_requires_a_parsed_selector():
    class ObsnumWorkflow(FileAccessorWorkflow):
        _obs_spec_cls = _ObsnumOnlySpec

    class ObsnumFileAccessor(FileAccessorBase[ObsnumWorkflow]):
        pass

    file_accessor = FileAccessorView.from_sources(
        _obs_spec_sources, accessor_types=(ToltecFileAccessor, ObsnumFileAccessor)
    )
    with pytest.raises(ValueError, match="Pass a parsed ObsSpec"):
        file_accessor.query_obs_spec(145606)
    result = file_accessor.query_obs_spec(_ObsnumOnlySpec.parse(145606))
    assert result.index.tolist() == file_accessor.table.index.tolist()
