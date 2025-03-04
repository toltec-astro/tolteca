from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import ConfigDict, Field, constr
from tollan.config.types import AbsDirectoryPath
from tollan.utils.general import rgetattr
from tollan.utils.log import timeit

from tolteca_kidsproc.kidsdata.sweep import MultiSweep

from .filestore import FileStoreConfigMixin
from .kids_find import KidsFind
from .pipeline import Step, StepConfig, StepContext, get_pipeline_contexts

ItemType = Literal["tone_prop", "chan_prop", "data_ctx"]


_obj_name_pattern = r"[A-z][A-z0-9.]*:[A-z][A-z0-9.]*/[A-z][A-z0-9.]*"


class DataProdOutputConfig(StepConfig, FileStoreConfigMixin):
    """The kids dataprod output config."""

    _filestore_path_attr: ClassVar = "path"

    path: AbsDirectoryPath = Field(
        default=".",
        description="root path.",
    )
    dump_context: bool = Field(
        default=False,
        description="whether to dump context pickle file.",
    )
    dump_objects: None | list[constr(pattern=_obj_name_pattern)] = Field(
        default=None,
        description="list of ininternal objects to dump as pickle file.",
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
            "Header.Toltec.ScanNum": swp.meta["scannum"],
            "Header.Toltec.RoachIndex": swp.meta["roach"],
            "uid_raw_obs": swp.meta["uid_raw_obs"],
            "uid_raw_obs_file": swp.meta["uid_raw_obs_file"],
        }
        tbl_cols = [
            "idx_det",
            "bitmask_det",
            "f_det",
            "Qr",
            "dist",
            "d_phi",
            "idx_chan",
            "f_chan",
            "amp_tone",
            "snr_d21",
            "snr_s21",
            "height_d21",
            "height_db_s21",
        ]
        tbl_kids_find = ctx_kf.data.detected_matched[tbl_cols]
        tbl_kids_find.meta.update(tbl_meta)
        item_path["kids_find"] = cfg.save_table(
            cfg.make_data_path(data=swp, suffix="_kids_find.ecsv"),
            tbl_kids_find,
        )

        tbl_chan_prop = ctx_kf.data.chan_matched[tbl_cols]
        tbl_chan_prop.meta.update(tbl_meta)
        item_path["chan_prop"] = cfg.save_table(
            cfg.make_data_path(data=swp, suffix="_chan_prop.ecsv"),
            tbl_chan_prop,
        )
        if cfg.dump_context:
            data_ctx_path = cfg.save_obj_pickle(
                cfg.make_data_path(data=swp, suffix="_ctx.pkl"),
                swp,
            )
        else:
            data_ctx_path = None
        item_path["data_ctx"] = data_ctx_path

        data_obj_names = cfg.dump_objects or []
        if data_obj_names:
            data_obj_dict = cls._resolve_data_objs(swp, data_obj_names)
            data_obj_path = cfg.save_obj_pickle(
                cfg.make_data_path(data=swp, suffix="_objs.pkl"),
                data_obj_dict,
            )
        else:
            data_obj_path = None
        item_path["data_obj"] = data_obj_path

        return True

    @classmethod
    def _resolve_data_objs(cls, data, names):
        ctxs = get_pipeline_contexts(data)
        result = defaultdict(dict)
        for name in names:
            context_key, attr = name.split("/", 1)
            result[context_key][attr] = rgetattr(ctxs[context_key], attr)
        return result
