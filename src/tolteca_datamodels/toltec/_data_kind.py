"""Filename classification rules shared by TolTEC file workflows."""

from tollan.accessor.pandas.model_table import RecordRuleMapper

from .types import ToltecDataKind

_data_kind_mapper = RecordRuleMapper[int](
    rules=[
        *(
            (
                {
                    "instru": "toltec",
                    "interface_kind": "roach",
                    "suffix": suffix,
                    "ext": "nc",
                },
                kind.value,
            )
            for suffix, kind in (
                ("vnasweep", ToltecDataKind.vnasweep),
                ("targsweep", ToltecDataKind.targsweep),
                ("targetsweep", ToltecDataKind.targsweep),
                ("tune", ToltecDataKind.tune),
                ("vnasweep_processed", ToltecDataKind.reduced_vnasweep),
                ("targsweep_processed", ToltecDataKind.reduced_targsweep),
                ("tune_processed", ToltecDataKind.reduced_targsweep),
                ("sweep", ToltecDataKind.reduced_sweep),
                ("timestream_processed", ToltecDataKind.solved_timestream),
                ("timestream", ToltecDataKind.raw_timestream),
                (None, ToltecDataKind.raw_timestream),
            )
        ),
        *(
            (
                {
                    "instru": "toltec",
                    "interface_kind": "roach",
                    "suffix": suffix,
                    "ext": "txt",
                },
                ToltecDataKind.kids_model_params_table.value,
            )
            for suffix in ("vnasweep", "targsweep", "tune")
        ),
        (
            {
                "instru": "toltec",
                "interface_kind": "roach",
                "proc_suffix": "targamps",
                "ext": "dat",
            },
            ToltecDataKind.targ_amps_dat.value,
        ),
        *(
            (
                {
                    "instru": "toltec",
                    "interface_kind": "roach",
                    "proc_suffix": suffix,
                    "ext": "ecsv",
                },
                ToltecDataKind.kids_prop_table.value,
            )
            for suffix in ("kids_find", "kids_fit", "chan_prop", "adrv")
        ),
        *(
            ({"instru": instru, "prefix": prefix, "ext": "nc"}, kind.value)
            for instru, prefix, kind in (
                ("toltec", "hwpr", ToltecDataKind.hwpr),
                ("toltec", "toltec_hk", ToltecDataKind.housekeeping),
                ("toltec", "wyatt", ToltecDataKind.wyatt),
                ("lmt", "tel_toltec", ToltecDataKind.lmt_tel),
                ("lmt", "tel_toltec2", ToltecDataKind.lmt_tel2),
            )
        ),
    ],
    default=ToltecDataKind.unknown.value,
)
