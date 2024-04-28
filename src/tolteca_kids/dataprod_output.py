from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import ConfigDict, Field
from tollan.config.types import AbsDirectoryPath
from tollan.utils.log import timeit

from tolteca_kidsproc.kidsdata.sweep import MultiSweep

from .kids_find import KidsFind
from .output import OutputConfigMixin
from .pipeline import Step, StepConfig, StepContext

ItemType = Literal["tone_prop", "chan_prop", "data_ctx"]


class DataProdOutputConfig(StepConfig, OutputConfigMixin):
    """The kids dataprod output config."""

    _output_subdir_fmt: ClassVar = "{obsnum}-{subobsnum}-{scannum}"
    _output_rootpath_attr: ClassVar = "path"

    path: AbsDirectoryPath = Field(
        default=".",
        description="root path.",
    )


@dataclass(kw_only=True)
class DataProdOutputData:
    """The data class for dataprod output."""

    item_path: dict[ItemType, Path] = ...


class DataProdOutputContext(StepContext["DataProdOutput", DataProdOutputConfig]):
    """The context class for dataprod output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: DataProdOutputData = Field(default_factory=DataProdOutputData)


class DataProdOutput(Step[DataProdOutputConfig, DataProdOutputContext]):
    """Data product output.

    This step write files to dataprod folder.
    """

    @classmethod
    @timeit
    def run(cls, data: MultiSweep, context):
        """Run dataprod output."""
        swp = data
        cfg = context.config
        ctd = context.data
        ctx_kf = KidsFind.get_context(data)
        if not ctx_kf.completed:
            raise ValueError("kids find step has not run yet.")
        item_path = ctd.item_path = {}

        tbl_meta = {
            "Header.Toltec.ObsNum": swp.meta["obsnum"],
            "Header.Toltec.SubObsNum": swp.meta["subobsnum"],
            "Header.Toltec.ScanObsNum": swp.meta["scannum"],
            "Header.Toltec.RoachIndex": swp.meta["roach"],
        }
        tbl_cols = [
            "idx_det",
            "bitmask_det",
            "f_det",
            "Qr",
            "idx_chan",
            "f_chan",
            "amp_tone",
            "dist",
        ]
        tbl_tone_prop = ctx_kf.data.detected_matched[tbl_cols]
        tbl_tone_prop.meta.update(tbl_meta)
        item_path["tone_prop"] = cfg.save_table(
            cfg.make_output_path(data=swp, suffix="_toneprop.ecsv"),
            tbl_tone_prop,
        )

        tbl_chan_prop = ctx_kf.data.chan_matched[tbl_cols]
        tbl_chan_prop.meta.update(tbl_meta)
        item_path["chan_prop"] = cfg.save_table(
            cfg.make_output_path(data=swp, suffix="_chanprop.ecsv"),
            tbl_chan_prop,
        )

        item_path["data_ctx"] = cfg.save_obj_pickle(
            cfg.make_output_path(data=swp, suffix="_ctx.pkl"),
            swp,
        )
        return True
