"""Dummy-instrument classification rules."""

from tollan.accessor.pandas.model_table import RecordRuleMapper

from tolteca_datamodels.lmt.types import LmtInfo

from .types import DummyInstruDataKind, DummyInstruInfo

_instru_data_kind_value = {
    DummyInstruInfo.instru: DummyInstruDataKind.counts.value,
    LmtInfo.instru: DummyInstruDataKind.telescope.value,
}
_data_kind_mapper = RecordRuleMapper[int](
    rules=[
        ({"instru": instru}, kind) for instru, kind in _instru_data_kind_value.items()
    ],
    default=DummyInstruDataKind.unknown.value,
)
